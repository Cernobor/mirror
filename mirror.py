import json
import tomllib
import typing
import time
import argparse
import sys

import numpy as np
import depthai as dai
import cv2

import pipeline
import host_sync
from rendering import BBox, Renderer
import utils

DISPLAY_SIZE: typing.Tuple[int, int] = (1920, 1080)
IMAGE_SIZE: typing.Tuple[int, int] = (1080, 1080)

def main():
    config = get_conf()
    print('Creating pipeline...')
    pl = pipeline.create_pipeline(IMAGE_SIZE, config)
    print('Saving pipeline to JSON...')
    with open('pipeline.json', 'w') as f:
        json.dump(pl.serializeToJson(), f, indent=2)
    print('Initializing device...')
    with dai.Device(pl) as dev:
        device = typing.cast(dai.Device, dev)
        print('Device initialized.')
        run(device, config)


def get_conf() -> utils.Config:
    ap = argparse.ArgumentParser('mirror')
    ap.add_argument('-c', '--config', type=argparse.FileType(mode='rb'), help='Configuration file in TOML format.', required=True)

    ns = ap.parse_args()
    confDict = tomllib.load(ns.config)

    res = utils.Config(**confDict)

    errors = res.validate()
    if errors:
        print('Errors in configuration:', file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)
    
    return res


def run(device: dai.Device, config: utils.Config):
    print(device.getUsbSpeed())

    if config.special_trigger_file is not None and config.final_trigger_file is not None:
        trigger_mode = 'file'
    elif config.special_trigger_pin is not None and config.final_trigger_pin is not None:
        trigger_mode = 'gpio'
    else:
        raise ValueError('Illegal state.')
    print(f'Trigger mode: {trigger_mode}')
    if trigger_mode == 'gpio':
        import gpiod
        chip = gpiod.chip(3)
        line_mapping = {
            46: 19,
            45: 22,
            44: 25,
            43: 18,
            42: 21
        }
        lines_special_final = chip.get_lines([line_mapping[config.special_trigger_pin], line_mapping[config.final_trigger_pin]])
        line_config = gpiod.line_request()
        line_config.consumer = 'mirror'
        line_config.request_type = gpiod.line_request.DIRECTION_INPUT
        line_config.flags = gpiod.line_request.FLAG_BIAS_PULL_UP
        lines_special_final.request(line_config)
        print('Configured trigger GPIOs.')

    device.setLogLevel(dai.LogLevel.INFO)
    device.setLogOutputLevel(dai.LogLevel.WARN)
    queues = [
        'color',
        'nearest_face',
        'depth'
    ]
    sync = host_sync.HostSync(device, *queues, print_add=False)
    renderer = Renderer(display_size=DISPLAY_SIZE,
                        image_size=utils.scale(IMAGE_SIZE, *config.global_res_scale),
                        global_upscale=tuple(reversed(config.global_res_scale)),
                        screen_rotated=config.screen_rotated,
                        depth=config.depth,
                        halo_common_dir=config.halo_common,
                        halo_special_dir=config.halo_special,
                        halo_common_blow_factor=config.halo_common_blow_factor,
                        halo_special_blow_factor=config.halo_special_blow_factor,
                        halo_position_mixing_coef=config.halo_position_mixing_coef,
                        halo_decay_coef=config.halo_decay_coef,
                        background_stars_no=config.background_stars_no,
                        #common_constellations_js=config.common_constellations,
                        #special_constellations_js=config.special_constellations,
                        common_constellations_dir=config.common_constellations,
                        special_constellations_dir=config.special_constellations,
                        halo_fade_in_time=config.halo_fade_in_time,
                        constellation_fade_in_time=config.constellation_fade_in_time,
                        halo_delay_time=config.halo_delay_time,
                        constellation_delay_time=config.constellation_delay_time,
                        debug=config.debug)
    latency_buffer = np.zeros((50,), dtype=np.float32)
    latency_buffer_idx = 0
    fps_buffer = np.zeros((50,), dtype=np.float32)
    fps_buffer_idx = 0
    last_frame_time = time.time()

    while True:
        time.sleep(0.001)
        msgs, seq = sync.get()
        if msgs is None:
            continue
        
        color_in: dai.ImgFrame = msgs.get('color', None)
        depth_in: dai.ImgFrame = msgs.get('depth', None)
        nearest_face_in: dai.NNData = msgs.get('nearest_face', None)

        latency = (dai.Clock.now() - color_in.getTimestamp()).total_seconds() * 1000
        latency_buffer[latency_buffer_idx] = latency
        latency_buffer_idx = (latency_buffer_idx + 1) % latency_buffer.size
        t = time.time()
        fps = 1 / (t - last_frame_time)
        last_frame_time = t
        fps_buffer[fps_buffer_idx] = fps
        fps_buffer_idx = (fps_buffer_idx + 1) % fps_buffer.size
        #print('Seq: {}, Latency: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}'.format(seq, latency, np.average(latency_buffer), np.std(latency_buffer)))
        #print('Seq: {}, FPS: {:.2f}, Average FPS: {:.2f}, Std: {:.2f}'.format(seq, fps, np.average(fps_buffer), np.std(fps_buffer)))

        color = typing.cast(cv2.Mat, color_in.getCvFrame())
        depth = depth_in.getFrame()
        
        bbox_raw = nearest_face_in.getLayerFp16('bbox')
        bbox = None
        if bbox_raw is not None and len(bbox_raw) == 4:
            rect = dai.Rect(dai.Point2f(bbox_raw[0], bbox_raw[1]),
                            dai.Point2f(bbox_raw[2], bbox_raw[3]))
            rect = rect.denormalize(color_in.getWidth(), color_in.getHeight())
            bbox = BBox(int(rect.topLeft().x),
                        int(rect.topLeft().y),
                        int(rect.bottomRight().x),
                        int(rect.bottomRight().y))
        if renderer.render(color, depth, bbox, seq):
            print('Requested stoppage.')
            break

        if trigger_mode == 'file':
            with open(config.special_trigger_file) as f:
                content = f.read().strip()
                renderer.special = content == '1'
            with open(config.final_trigger_file) as f:
                content = f.read().strip()
                renderer.final_trigger = content == '1'
        elif trigger_mode == 'gpio':
            special, final = lines_special_final.get_values()
            renderer.special = special == 0
            renderer.final_trigger = final == 0
        else:
            raise ValueError('Illegal trigger_mode')
    
    if trigger_mode == 'gpio':
        lines_special_final.release()


if __name__ == '__main__':
    main()

import json
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
    config = parse_args()
    print('Creating pipeline...')
    pl = pipeline.create_pipeline(IMAGE_SIZE,
                                  (config.global_res_scale[0] * config.video_res_scale[0],
                                   config.global_res_scale[1] * config.video_res_scale[1]))
    print('Saving pipeline to JSON...')
    with open('pipeline.json', 'w') as f:
        json.dump(pl.serializeToJson(), f, indent=2)
    print('Initializing device...')
    with dai.Device(pl) as dev:
        device = typing.cast(dai.Device, dev)
        print('Device initialized.')
        run(device, config)


def parse_args() -> utils.Config:
    ap = argparse.ArgumentParser('mirror')
    ap.add_argument('--debug', action='store_true', help='If specified, mirror is rendered in non-fullscreen, smaller window.')
    ap.add_argument('--depth', type=int, default=0, help='Cutoff depth between foreground (person) and background.')
    ap.add_argument('--halo-common', help='Path to a directory containing images of the common halo animation.')
    ap.add_argument('--halo-special', help='Path to a directory containing images of the special halo animation.')
    ap.add_argument('--halo-position-mixing-coef', type=float, default=1, help='Mixing coefficient for halo position. Must be in the range [0, 1].')
    ap.add_argument('--background-stars-no', type=int, default=0, help='Number of randomb background stars.')
    ap.add_argument('--common-constellations', help='Path to a JSON file containing common constellations.')
    ap.add_argument('--special-constellations', help='Path to a JSON file containing special constellations.')
    ap.add_argument('--halo-fade-in-time', type=float, default=1, help='Fade-in time of the halo effect.')
    ap.add_argument('--constellation-fade-in-time', type=float, default=1, help='Fade-in time of the constellation effect.')
    ap.add_argument('--halo-delay-time', type=float, default=0, help='Delay time before the start of the fade-in of the halo effect.')
    ap.add_argument('--constellation-delay-time', type=float, default=0, help='Delay time before the start of the fade-in of the constellation effect.')
    ap.add_argument('--trigger-files', nargs=2, help='Paths to files that will be read for special and final trigger respectively.')
    ap.add_argument('--trigger-gpios', type=int, nargs=2, help='Pin numbers that will be checked (pullup, trigger on LOW) for special and final trigger respectively.')
    ap.add_argument('--screen-rotated', action='store_true', help='Tells the program that the display is already rotated by the system.')
    ap.add_argument('--global-resolution-scale', type=int, nargs=2, default=[1, 1], help='Global scaling of the resolution. The two arguments are numerator and denominator of a fraction by which the standard 1080p resolution will be multiplied.')
    ap.add_argument('--video-resolution-scale', type=int, nargs=2, default=[1, 1], help='Scaling of the resolution of the video. The two arguments are numerator and denominator of a fraction by which the standard 1080p resolution will be multiplied.')

    ns = ap.parse_args()
    print(ns)
    if ns.trigger_files is None and ns.trigger_gpios is None or ns.trigger_files is not None and ns.trigger_gpios is not None:
        ap.print_usage(file=sys.stderr)
        print(f'{sys.argv[0]}: error: exactly one of --trigger-files and --trigger-gpios must be specified')
        sys.exit(1)
    elif ns.trigger_files is not None:
        stf = ns.trigger_files[0]
        ftf = ns.trigger_files[1]
        stp = None
        ftp = None
    elif ns.trigger_gpios is not None:
        stf = None
        ftf = None
        stp = ns.trigger_gpios[0]
        ftp = ns.trigger_gpios[1]
    else:
        raise ValueError('Illegal state.')
    if ns.halo_position_mixing_coef < 0 or ns.halo_position_mixing_coef > 1:
        ap.print_usage(file=sys.stderr)
        print(f'{sys.argv[0]}: error: --halo-position-mixing-coef must be in the range [0, 1]')
        sys.exit(1)
    return utils.Config(
        debug=ns.debug,
        screen_rotated=ns.screen_rotated,
        depth=ns.depth,
        halo_common=ns.halo_common,
        halo_special=ns.halo_special,
        halo_position_mixing_coef=ns.halo_position_mixing_coef,
        background_stars_no=ns.background_stars_no,
        common_constellations=ns.common_constellations,
        special_constellations=ns.special_constellations,
        halo_fade_in_time=ns.halo_fade_in_time,
        constellation_fade_in_time=ns.constellation_fade_in_time,
        halo_delay_time=ns.halo_delay_time,
        constellation_delay_time=ns.constellation_delay_time,
        special_trigger_file=stf,
        final_trigger_file=ftf,
        special_trigger_pin=stp,
        final_trigger_pin=ftp,
        global_res_scale=ns.global_resolution_scale,
        video_res_scale=ns.video_resolution_scale
    )


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
                        halo_position_mixing_coef=config.halo_position_mixing_coef,
                        background_stars_no=config.background_stars_no,
                        common_constellations_js=config.common_constellations,
                        special_constellations_js=config.special_constellations,
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

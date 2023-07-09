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

DISPLAY_SIZE: tuple[int, int] = (1920, 1080)
IMAGE_SIZE: tuple[int, int] = (1080, 1080)

def main():
    config = parse_args()
    print('Creating pipeline...')
    pl = pipeline.create_pipeline()
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
    ap.add_argument('--background-stars-no', type=int, default=0, help='Number of randomb background stars.')
    ap.add_argument('--common-constellations', help='Path to a JSON file containing common constellations.')
    ap.add_argument('--special-constellations', help='Path to a JSON file containing special constellations.')
    ap.add_argument('--trigger-files', nargs=2, help='Paths to files that will be read for special and final trigger respectively.')
    ap.add_argument('--trigger-gpios', type=int, nargs=2, help='Pin numbers that will be checked (pullup, trigger on LOW) for special and final trigger respectively.')
    ap.add_argument('--screen-rotated', action='store_true', help='Tells the program that the display is already rotated by the system.')

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
    return utils.Config(
        debug=ns.debug,
        screen_rotated=ns.screen_rotated,
        depth=ns.depth,
        halo_common=ns.halo_common,
        halo_special=ns.halo_special,
        background_stars_no=ns.background_stars_no,
        common_constellations=ns.common_constellations,
        special_constellations=ns.special_constellations,
        special_trigger_file=stf,
        final_trigger_file=ftf,
        special_trigger_pin=stp,
        final_trigger_pin=ftp
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
        cfg.request_type = gpiod.line_request.DIRECTION_INPUT
        cfg.flags = gpiod.line_request.FLAG_BIAS_PULL_UP
        lines_special_final.request(cfg)
        print('Configured trigger GPIOs.')

    device.setLogLevel(dai.LogLevel.INFO)
    device.setLogOutputLevel(dai.LogLevel.INFO)
    queues = [
        'color',
        'nearest_face',
        'depth'
    ]
    sync = host_sync.HostSync(device, *queues, print_add=False)
    renderer = Renderer(display_size=DISPLAY_SIZE,
                        image_size=IMAGE_SIZE,
                        screen_rotated=config.screen_rotated,
                        depth=config.depth,
                        halo_common_dir=config.halo_common,
                        halo_special_dir=config.halo_special,
                        background_stars_no=config.background_stars_no,
                        common_constellations_js=config.common_constellations,
                        special_constellations_js=config.special_constellations,
                        debug=config.debug)
    latency_buffer = np.zeros((50,), dtype=np.float32)
    latency_buffer_idx = 0

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
        print('Seq: {}, Latency: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}'.format(seq, latency, np.average(latency_buffer), np.std(latency_buffer)))

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
                renderer.final = content == '1'
        elif trigger_mode == 'gpio':
            special, final = lines_special_final.get_values()
            renderer.special = special == 0
            renderer.final = final == 0
        else:
            raise ValueError('Illegal trigger_mode')
    
    if trigger_mode == 'gpio':
        lines_special_final.release()


if __name__ == '__main__':
    main()

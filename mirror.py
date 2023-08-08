import json
import os
import shutil
import tempfile
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

def main():
    config = get_conf()
    cleanup = preprocess(config)
    try:
        print('Creating pipeline...')
        pl = pipeline.create_pipeline((min(DISPLAY_SIZE), min(DISPLAY_SIZE)), config)
        print('Saving pipeline to JSON...')
        with open('pipeline.json', 'w') as f:
            json.dump(pl.serializeToJson(), f, indent=2)
        print('Initializing device...')
        with dai.Device(pl) as dev:
            device = typing.cast(dai.Device, dev)
            print('Device initialized.')
            run(device, config)
    finally:
        cleanup()
        pass


def get_conf() -> utils.Config:
    ap = argparse.ArgumentParser('mirror')
    ap.add_argument('-c', '--config', type=argparse.FileType(mode='rb'), help='Configuration file in TOML format.', required=True)

    ns = ap.parse_args()
    confDict = tomllib.load(ns.config)

    res = utils.Config(**confDict)

    errors = res.validate_sanitize()
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
                        config=config)
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
        
        det_raw = nearest_face_in.getLayerFp16('bbox')
        face = None
        if det_raw is not None and len(det_raw) == 5:
            rect = dai.Rect(dai.Point2f(det_raw[0], det_raw[1]),
                            dai.Point2f(det_raw[2], det_raw[3]))
            rect = rect.denormalize(color_in.getWidth(), color_in.getHeight())
            bbox = BBox(int(rect.topLeft().x),
                        int(rect.topLeft().y),
                        int(rect.bottomRight().x),
                        int(rect.bottomRight().y))
            dist = det_raw[4]
            face = (bbox, dist)
        if renderer.render(color, depth, face, seq):
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


def preprocess(conf: utils.Config) -> typing.Callable:
    cleanups: typing.List[typing.Tuple[str, typing.Callable]] = []

    print('Preprocessing...')
    if conf.blend_mode == 'alpha':
        dn = 'preprocessed-alpha-black'
        if conf.alpha_convert_black_common:
            print('  ...converting common halo from black to alpha', end='')
            p = os.path.join(conf.halo_common_path, dn)
            if not os.path.exists(p):
                os.mkdir(p)
                convert_alpha_black(conf.halo_common_path, p)
            else:
                print(' - already exists, skipping', end='')
            conf.halo_common_path = p
            print()
        if conf.alpha_convert_black_special:
            print('  ...converting special halo from black to alpha', end='')
            p = os.path.join(conf.halo_special_path, dn)
            if not os.path.exists(p):
                os.mkdir(p)
                convert_alpha_black(conf.halo_special_path, p)
            else:
                print(' - already exists, skipping', end='')
            conf.halo_special_path = p
            print()
    elif conf.blend_mode == 'screen':
        dn = 'preprocessed-inverted'
        
        print('  ...inverting common halo', end='')
        p = os.path.join(conf.halo_common_path, dn)
        if not os.path.exists(p):
            os.mkdir(p)
            invert(conf.halo_common_path, p)
        else:
            print(' - already exists, skipping', end='')
        conf.halo_common_path = p
        print()
        
        print('  ...inverting special halo', end='')
        p = os.path.join(conf.halo_special_path, dn)
        if not os.path.exists(p):
            os.mkdir(p)
            invert(conf.halo_special_path, p)
        else:
            print(' - already exists, skipping', end='')
        conf.halo_special_path = p
        print()
    
    def cleanup():
        if len(cleanups) == 0:
            return
        print('Cleaning up...')
        for info, task in cleanups:
            print(f'  ...{info}')
            task()
        print('Cleaned up.')
    
    print('Done preprocessing.')
    return cleanup
    

def convert_alpha_black(src: str, tgt: str):
    with os.scandir(src) as sd:
        for entry in sd:
            if entry.is_dir():
                continue
            if not entry.name.lower().endswith('.png'):
                continue
            img = cv2.imread(entry.path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = img[:, :, :-1].max(axis=2)
            cv2.imwrite(os.path.join(tgt, entry.name), img)


def invert(src: str, tgt: str):
    with os.scandir(src) as sd:
        for entry in sd:
            if entry.is_dir():
                continue
            if not entry.name.lower().endswith('.png'):
                continue
            img = cv2.imread(entry.path)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.bitwise_not(img)
            cv2.imwrite(os.path.join(tgt, entry.name), img)


if __name__ == '__main__':
    main()

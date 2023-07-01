import json
import typing
import time
import argparse

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
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--depth', type=int, default=0)
    ap.add_argument('--halo-common')
    ap.add_argument('--halo-special')
    ap.add_argument('--special-trigger-file')
    ap.add_argument('--final-trigger-file')
    ap.add_argument('--screen-rotated', action='store_true')

    ns = ap.parse_args()
    return utils.Config(
        debug=ns.debug,
        screen_rotated=ns.screen_rotated,
        depth=ns.depth,
        halo_common=ns.halo_common,
        halo_special=ns.halo_special,
        special_trigger_file=ns.special_trigger_file,
        final_trigger_file=ns.final_trigger_file
    )


def run(device: dai.Device, config: utils.Config):
    print(device.getUsbSpeed())
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
                        halo_common=config.halo_common,
                        halo_special=config.halo_special,
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

        with open(config.special_trigger_file) as f:
            content = f.read().strip()
            renderer.special = content == '1'
        with open(config.final_trigger_file) as f:
            content = f.read().strip()
            renderer.final = content == '1'


if __name__ == '__main__':
    main()

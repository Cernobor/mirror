import json
import typing
import time

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
    show_depth = False
    print('Creating pipeline...')
    pl = pipeline.create_pipeline(show_depth)
    print('Saving pipeline to JSON...')
    with open('pipeline.json', 'w') as f:
        json.dump(pl.serializeToJson(), f, indent=2)
    print('Initializing device...')
    with dai.Device(pipeline.create_pipeline(show_depth)) as dev:
        device = typing.cast(dai.Device, dev)
        print('Device initialized.')
        run(device, show_depth)


def run(device: dai.Device, show_depth: bool):
    print(device.getUsbSpeed())
    device.setLogLevel(dai.LogLevel.INFO)
    device.setLogOutputLevel(dai.LogLevel.INFO)
    queues = [
        'color',
        'nearest_face'
    ]
    if show_depth:
        queues.append('depth')
    sync = host_sync.HostSync(device, *queues, print_add=False)
    renderer = Renderer(DISPLAY_SIZE, IMAGE_SIZE)
    latency_buffer = np.zeros((50,), dtype=np.float32)
    latency_buffer_idx = 0

    while True:
        time.sleep(0.001)
        msgs, seq = sync.get()
        if msgs is None:
            continue
        
        #print('Seq', seq, 'lag', sync.get_lag())

        color_in: dai.ImgFrame = msgs.get('color', None)
        nearest_face_in: dai.NNData = msgs.get('nearest_face', None)

        latency = (dai.Clock.now() - color_in.getTimestamp()).total_seconds() * 1000
        latency_buffer[latency_buffer_idx] = latency
        latency_buffer_idx = (latency_buffer_idx + 1) % latency_buffer.size
        #print('Latency: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}'.format(latency, np.average(latency_buffer), np.std(latency_buffer)))

        color_frame = color_in.getCvFrame()
        color = typing.cast(cv2.Mat, color_frame)

        bbox_raw = nearest_face_in.getLayerFp16('bbox')
        bbox = None
        if bbox_raw is not None and len(bbox_raw) == 4:
            rect = dai.Rect(dai.Point2f(1 - bbox_raw[2], bbox_raw[1]),
                            dai.Point2f(1 - bbox_raw[0], bbox_raw[3]))
            rect = rect.denormalize(color_in.getWidth(), color_in.getHeight())
            bbox = BBox(int(rect.topLeft().x),
                        int(rect.topLeft().y),
                        int(rect.bottomRight().x),
                        int(rect.bottomRight().y))
        if renderer.render(color, bbox, seq):
            print('Requested stoppage.')
            break
        
        #cv2.imshow('Color', color)

        #if show_depth:
        #    depth_in: dai.ImgFrame = msgs.get('depth', None)
        #    stereo_cfg_in: dai.StereoDepthConfig = typing.cast(dai.StereoDepthConfig, sync.device.getOutputQueue('stereo_cfg').get())
        #    depth = utils.depth_to_cv_frame(depth_in, stereo_cfg_in)
        #    depth_resized = cv2.resize(depth, DISPLAY_SIZE)
        #    #cv2.imshow('Depth', depth_resized)
        
        #if cv2.waitKey(1) == ord('q'):
        #    break


if __name__ == '__main__':
    main()

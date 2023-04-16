import json
import typing
import pipeline
import depthai as dai
import host_sync
import cv2
import utils
import time
import struct

DISPLAY_SIZE: tuple[int, int] = (500, 500)

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
    device.setLogLevel(dai.LogLevel.INFO)
    device.setLogOutputLevel(dai.LogLevel.INFO)
    queues = [
        'color',
        'nearest_face'
    ]
    if show_depth:
        queues.append('depth')
    sync = host_sync.HostSync(device, *queues, print_add=False)
    while loop(sync, show_depth):
        time.sleep(0.001)


def loop(sync: host_sync.HostSync, show_depth: bool) -> bool:
    msgs, seq = sync.get()
    if msgs is None:
        return True
    
    print('Seq', seq, 'lag', sync.get_lag())

    color_in: dai.ImgFrame = msgs.get('color', None)
    nearest_face_in: dai.NNData = msgs.get('nearest_face', None)

    color = typing.cast(cv2.Mat, color_in.getCvFrame())

    bbox_raw = nearest_face_in.getLayerFp16('bbox')
    if bbox_raw is not None and len(bbox_raw) == 4:
        rect = dai.Rect(dai.Point2f(1 - bbox_raw[2], bbox_raw[1]),
                        dai.Point2f(1 - bbox_raw[0], bbox_raw[3]))
        rect = rect.denormalize(color_in.getWidth(), color_in.getHeight())
        xmin = int(rect.topLeft().x)
        ymin = int(rect.topLeft().y)
        xmax = int(rect.bottomRight().x)
        ymax = int(rect.bottomRight().y)
        bbox_top_left = (xmin, ymin)
        bbox_bottom_right = (xmax, ymax)
        color = cv2.rectangle(color, bbox_top_left, bbox_bottom_right, (255, 255, 255), 5)
    
    color_resized = cv2.resize(color, DISPLAY_SIZE)
    cv2.imshow('Color', color_resized)

    if show_depth:
        depth_in: dai.ImgFrame = msgs.get('depth', None)
        stereo_cfg_in: dai.StereoDepthConfig = typing.cast(dai.StereoDepthConfig, sync.device.getOutputQueue('stereo_cfg').get())
        depth = utils.depth_to_cv_frame(depth_in, stereo_cfg_in)
        depth_resized = cv2.resize(depth, DISPLAY_SIZE)
        cv2.imshow('Depth', depth_resized)
    
    if cv2.waitKey(1) == ord('q'):
        return False
    
    return True


if __name__ == '__main__':
    main()

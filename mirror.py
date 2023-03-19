import pipeline
import depthai as dai
import host_sync
import cv2
import utils
import time

DISPLAY_SIZE = (500, 500)

def main():
    show_depth = False
    print('Initializing device...')
    with dai.Device(pipeline.create_pipeline(show_depth)) as device:
        print('Device initialized.')
        run(device, show_depth)


def run(device: dai.Device, show_depth: bool):
    queues = [
        'color',
        'faces',
        'spatial'
    ]
    if show_depth:
        queues.append('depth')
    sync = host_sync.HostSync(device, *queues, print_add=True)
    while loop(sync, show_depth):
        time.sleep(0.001)


def loop(sync: host_sync.HostSync, show_depth: bool) -> bool:
    msgs, seq = sync.get()
    if msgs is None:
        return True
    
    print('Seq', seq, 'lag', sync.get_lag())

    color_in: dai.ImgFrame = msgs.get('color', None)
    faces_in: dai.ImgDetections = msgs.get('faces', None)
    spatial_in: dai.SpatialLocationCalculatorData = msgs.get('spatial', None)
    #print(seq, color_in, faces_in, spatial_in)

    color = color_in.getCvFrame()

    for i, det in enumerate(faces_in.detections):
        rect = utils.process_detection(color_in, det)
        x0 = int(rect.topLeft().x)
        y0 = int(rect.topLeft().y)
        x1 = int(rect.bottomRight().x)
        y1 = int(rect.bottomRight().y)
        color = cv2.rectangle(color, (x0, y0), (x1, y1), (255, 255, 255), 5)
        location = spatial_in.getSpatialLocations()[i]
        color = cv2.putText(color, f'{location.depthAverage:.01f}', (x0, y0), 0, 2, (255, 255, 255), 3)
    
    color_resized = cv2.resize(color, DISPLAY_SIZE)
    cv2.imshow('Color', color_resized)

    if show_depth:
        depth_in: dai.ImgFrame = msgs.get('depth', None)
        stereo_cfg_in: dai.StereoDepthConfig = sync.device.getOutputQueue('stereo_cfg').get()
        depth = utils.depth_to_cv_frame(depth_in, stereo_cfg_in)
        depth_resized = cv2.resize(depth, DISPLAY_SIZE)
        cv2.imshow('Depth', depth_resized)
    
    if cv2.waitKey(1) == ord('q'):
        return False
    
    return True


if __name__ == '__main__':
    main()

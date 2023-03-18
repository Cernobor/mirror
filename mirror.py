import pipeline
import depthai as dai
import host_sync
import cv2
import utils

DISPLAY_SIZE = (500, 500)

def main():
    show_depth = True
    print('Initializing device...')
    with dai.Device(pipeline.create_pipeline(show_depth)) as device:
        print('Device initialized.')
        run(device, show_depth)


def run(device: dai.Device, show_depth: bool):
    queues = ['color', 'faces']
    if show_depth:
        queues.append('depth')
    sync = host_sync.HostSync(device, *queues)
    while loop(sync, show_depth):
        pass


def loop(sync: host_sync.HostSync, show_depth: bool) -> bool:
    msgs, seq = sync.get()
    if msgs is None:
        return True
    
    print('Lag', sync.get_lag())

    color_in: dai.ImgFrame = msgs.get('color', None)
    faces_in: dai.ImgDetections = msgs.get('faces', None)

    color = color_in.getCvFrame()

    for det in faces_in.detections:
        rect = utils.process_detection(color_in, det)
        color = cv2.rectangle(color, [int(rect.topLeft().x), int(rect.topLeft().y)], [int(rect.bottomRight().x), int(rect.bottomRight().y)], (255, 255, 255), 5)
    
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

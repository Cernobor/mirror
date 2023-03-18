import pipeline
import depthai as dai
import host_sync
import cv2
import utils

DISPLAY_SIZE = (500, 500)

def main():
    print('Initializing device...')
    with dai.Device(pipeline.create_pipeline()) as device:
        print('Device initialized.')
        run(device)


def run(device: dai.Device):
    sync = host_sync.HostSync(device, 'color', 'depth')
    while loop(sync):
        pass


def loop(sync: host_sync.HostSync) -> bool:
    msgs, seq = sync.get()
    if msgs is None:
        return True
    
    print('Lag', sync.get_lag())

    color_in: dai.ImgFrame = msgs.get('color', None)
    depth_in: dai.ImgFrame = msgs.get('depth', None)
    stereo_cfg_in: dai.StereoDepthConfig = sync.device.getOutputQueue('stereo_cfg').get()

    color = color_in.getCvFrame()
    color_resized = cv2.resize(color, DISPLAY_SIZE)
    cv2.imshow('Color', color_resized)

    depth = utils.depth_to_cv_frame(depth_in, stereo_cfg_in)
    depth_resized = cv2.resize(depth, DISPLAY_SIZE)
    cv2.imshow('Depth', depth_resized)
    
    if cv2.waitKey(1) == ord('q'):
        return False
    
    return True


if __name__ == '__main__':
    main()

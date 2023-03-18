import depthai as dai

def create_pipeline():
    pipeline = dai.Pipeline()

    ## Nodes

    # Color
    color = pipeline.create(dai.node.ColorCamera)
    color.setPreviewSize(300, 300)
    color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    color.setVideoSize(1080,1080)
    color.setInterleaved(False)
    color.setFps(25)

    # Mono left
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoLeft.setFps(25)
    
    # Mono right
    monoRight = pipeline.create(dai.node.MonoCamera)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoRight.setFps(25)

    # Stereo
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.initialConfig.setConfidenceThreshold(245)
    stereo.initialConfig.setMedianFilter(dai.StereoDepthConfig.MedianFilter.KERNEL_5x5)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    ## IO

    # Color out
    colorXout = pipeline.create(dai.node.XLinkOut)
    colorXout.setStreamName('color')
    # Depth out
    depthXout = pipeline.create(dai.node.XLinkOut)
    depthXout.setStreamName('depth')

    ## Linking

    # Mono left/right -> stereo
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    # Color -> color out
    color.video.link(colorXout.input)
    # Stereo depth -> depth out
    stereo.depth.link(depthXout.input)

    return pipeline
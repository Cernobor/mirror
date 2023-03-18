import depthai as dai
import blobconverter

def create_pipeline(transport_depth=False):
    pipeline = dai.Pipeline()

    ## Nodes

    # Color
    color = pipeline.create(dai.node.ColorCamera)
    color.setPreviewSize(300, 300)
    color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    color.setVideoSize(1080,1080)
    color.setInterleaved(False)
    color.setFps(25)

    # Color flip
    colorFlip = pipeline.create(dai.node.ImageManip)
    colorFlip.setMaxOutputFrameSize(1749600)
    colorFlip.initialConfig.setHorizontalFlip(True)

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
    stereo.initialConfig.setMedianFilter(dai.StereoDepthConfig.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Depth crop
    depthCrop = pipeline.create(dai.node.ImageManip)
    depthCrop.setMaxOutputFrameSize(2332800)
    depthCrop.initialConfig.setCropRect(420/1920, 0, (1080+420)/1920, 1)

    # Face detection NN
    faceDetNN = pipeline.create(dai.node.MobileNetDetectionNetwork)
    faceDetNN.setConfidenceThreshold(0.5)
    faceDetNN.setBlobPath(blobconverter.from_zoo(
        name='face-detection-retail-0004',
        shaves=6,
    ))
    faceDetNN.input.setBlocking(False)
    faceDetNN.input.setQueueSize(1)

    ## IO

    # Color out
    colorXout = pipeline.create(dai.node.XLinkOut)
    colorXout.setStreamName('color')
    # Stereo config out
    stereoCfgXout = pipeline.create(dai.node.XLinkOut)
    stereoCfgXout.setStreamName('stereo_cfg')
    # Depth out
    if transport_depth:
        depthXout = pipeline.create(dai.node.XLinkOut)
        depthXout.setStreamName('depth')
    # Fece detection out
    facesXout = pipeline.create(dai.node.XLinkOut)
    facesXout.setStreamName('faces')

    ## Linking

    # Color preview -> face detection
    color.preview.link(faceDetNN.input)
    # Mono left/right -> stereo
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    # Color -> color flip
    color.video.link(colorFlip.inputImage)
    # Color flip -> color out
    colorFlip.out.link(colorXout.input)
    # Stereo config -> stereo config out
    stereo.outConfig.link(stereoCfgXout.input)
    # Stereo depth -> depth crop
    stereo.depth.link(depthCrop.inputImage)
    # Depth crop -> depth out
    if transport_depth:
        depthCrop.out.link(depthXout.input)
    # Face detection -> face detection out
    faceDetNN.out.link(facesXout.input)

    return pipeline
import depthai as dai
import blobconverter


def create_pipeline(transport_depth=False):
    pipeline = dai.Pipeline()

    # Nodes

    # Color
    color = pipeline.create(dai.node.ColorCamera)
    color.setPreviewSize(300, 300)
    color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    color.setVideoSize(1080, 1080)
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
    monoRight.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoRight.setFps(25)

    # Stereo
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.initialConfig.setConfidenceThreshold(245)
    stereo.initialConfig.setMedianFilter(
        dai.StereoDepthConfig.MedianFilter.KERNEL_5x5)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Depth crop
    depthCrop = pipeline.create(dai.node.ImageManip)
    depthCrop.setMaxOutputFrameSize(2332800)
    depthCrop.initialConfig.setCropRect(420/1920, 0, (1080+420)/1920, 1)
    depthCrop.inputImage.setBlocking(False)
    depthCrop.inputImage.setQueueSize(1)

    # Face detection NN
    faceDetNN = pipeline.create(dai.node.MobileNetDetectionNetwork)
    faceDetNN.setConfidenceThreshold(0.5)
    faceDetNN.setBlobPath(blobconverter.from_zoo(
        name='face-detection-retail-0004',
        shaves=6,
    ))
    faceDetNN.input.setBlocking(False)
    faceDetNN.input.setQueueSize(1)

    # Script collecting face detections and stereo depth, sending both to spatial locator
    detectionsDepthToSpatial = pipeline.create(dai.node.Script)
    detectionsDepthToSpatial.setProcessor(dai.ProcessorType.LEON_CSS)
    detectionsDepthToSpatial.setScript(load_script('device_scripts/detections_depth_to_spatial.py'))
    detectionsDepthToSpatial.inputs['depth'].setBlocking(False)
    detectionsDepthToSpatial.inputs['depth'].setQueueSize(1)

    # Spatial locator
    spatial = pipeline.create(dai.node.SpatialLocationCalculator)
    roi = dai.SpatialLocationCalculatorConfigData()
    roi.roi = dai.Rect(dai.Point2f(1080 // 2 - 5, 1080 // 2 - 5), dai.Size2f(10, 10))
    roi.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
    spatial.initialConfig.addROI(roi)
    spatial.inputDepth.setBlocking(False)
    spatial.inputDepth.setQueueSize(1)

    # IO

    # Color out
    colorXout = pipeline.create(dai.node.XLinkOut)
    colorXout.setStreamName('color')
    colorXout.input.setBlocking(False)
    colorXout.input.setQueueSize(1)
    if transport_depth:
        # Stereo config out
        stereoCfgXout = pipeline.create(dai.node.XLinkOut)
        stereoCfgXout.setStreamName('stereo_cfg')
        # Depth out
        depthXout = pipeline.create(dai.node.XLinkOut)
        depthXout.setStreamName('depth')
    # Face detection out
    facesXout = pipeline.create(dai.node.XLinkOut)
    facesXout.setStreamName('faces')
    facesXout.input.setBlocking(False)
    facesXout.input.setQueueSize(1)
    # Spatial locations out
    spatialXout = pipeline.create(dai.node.XLinkOut)
    spatialXout.setStreamName('spatial')
    spatialXout.input.setBlocking(False)
    spatialXout.input.setQueueSize(1)

    # Linking

    # Color preview -> face detection
    color.preview.link(faceDetNN.input)
    # Mono left/right -> stereo
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    # Color -> color flip
    color.video.link(colorFlip.inputImage)
    # Color flip -> color out
    colorFlip.out.link(colorXout.input)
    # Stereo depth -> depth crop
    stereo.depth.link(depthCrop.inputImage)
    # Depth crop -> detections-depth to spatial
    depthCrop.out.link(detectionsDepthToSpatial.inputs['depth'])
    depthCrop.out.link(spatial.inputDepth)
    if transport_depth:
        # Stereo config -> stereo config out
        stereo.outConfig.link(stereoCfgXout.input)
        # Depth crop -> depth out
        depthCrop.out.link(depthXout.input)
    # Face detection -> face detection out
    faceDetNN.out.link(facesXout.input)
    # Detections-depth to spatial cfg/depth -> spatial locator cfg/depth
    detectionsDepthToSpatial.outputs['depth_cfg'].link(spatial.inputConfig)
    detectionsDepthToSpatial.outputs['depth_img'].link(spatial.inputDepth)
    # Spatial locator -> spatial out
    spatial.out.link(spatialXout.input)

    return pipeline


def load_script(path: str) -> str:
    data = ''
    with open(path, 'r') as f:
        while True:
            l = f.readline()
            if l == '# --start--\n':
                data = f.read()
                break
            if l == '':
                break
    return data

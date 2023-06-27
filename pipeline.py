import depthai as dai
import blobconverter


def create_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()
    fps = 20

    # Nodes

    # Color
    color = pipeline.create(dai.node.ColorCamera)
    color.setPreviewSize(300, 300)
    color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    color.setVideoSize(1080, 1080)
    color.setInterleaved(False)
    color.setFps(fps)

    # Preview flip
    previewFlip = pipeline.create(dai.node.ImageManip)
    previewFlip.setMaxOutputFrameSize(270_000)
    previewFlip.initialConfig.setHorizontalFlip(True)
    previewFlip.inputImage.setBlocking(False)
    previewFlip.inputImage.setQueueSize(1)

    # Color flip
    colorFlip = pipeline.create(dai.node.ImageManip)
    colorFlip.setMaxOutputFrameSize(1_749_600)
    colorFlip.initialConfig.setHorizontalFlip(True)
    colorFlip.inputImage.setBlocking(False)
    colorFlip.inputImage.setQueueSize(1)

    # Mono left
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoLeft.setFps(fps)

    # Mono right
    monoRight = pipeline.create(dai.node.MonoCamera)
    monoRight.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoRight.setFps(fps)

    # Stereo
    stereo = pipeline.create(dai.node.StereoDepth)
    
    cfg = dai.RawStereoDepthConfig()
    cfg.costMatching.confidenceThreshold = 245
    cfg.postProcessing.median = dai.MedianFilter.KERNEL_7x7
    
    cfg.postProcessing.temporalFilter.enable = True
    cfg.postProcessing.temporalFilter.persistencyMode = dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.PERSISTENCY_INDEFINITELY
    cfg.postProcessing.temporalFilter.alpha = .4
    cfg.postProcessing.temporalFilter.delta = 0
    
    cfg.postProcessing.spatialFilter.enable = False
    cfg.postProcessing.spatialFilter.holeFillingRadius = 3
    cfg.postProcessing.spatialFilter.alpha = .5
    cfg.postProcessing.spatialFilter.delta = 0
    cfg.postProcessing.spatialFilter.numIterations = 1
    
    #cfg.postProcessing.thresholdFilter = dai.RawStereoDepthConfig.PostProcessing.ThresholdFilter()
    #cfg.postProcessing.thresholdFilter.minRange = 0
    #cfg.postProcessing.thresholdFilter.maxRange = 2000
    
    cfg.postProcessing.speckleFilter.enable = False
    cfg.postProcessing.speckleFilter.speckleRange = 300
    
    cfg.algorithmControl.depthUnit = dai.RawStereoDepthConfig.AlgorithmControl.DepthUnit.MILLIMETER
    cfg.algorithmControl.enableLeftRightCheck = True
    cfg.algorithmControl.enableSubpixel = True
    
    stereo.initialConfig.set(cfg)

    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Depth crop & flip
    depthCropFlip = pipeline.create(dai.node.ImageManip)
    depthCropFlip.setMaxOutputFrameSize(2332800)
    depthCropFlip.initialConfig.setCropRect(420/1920, 0, (1080+420)/1920, 1)
    depthCropFlip.initialConfig.setResize(1080 // 3, 1080 // 3)
    depthCropFlip.initialConfig.setHorizontalFlip(True)
    depthCropFlip.inputImage.setBlocking(False)
    depthCropFlip.inputImage.setQueueSize(1)

    # Face detection NN
    faceDetNN = pipeline.create(dai.node.MobileNetDetectionNetwork)
    faceDetNN.setConfidenceThreshold(0.5)
    faceDetNN.setBlobPath(blobconverter.from_zoo(
        name='face-detection-retail-0004',
        shaves=9,
    ))
    faceDetNN.setNumInferenceThreads(2)
    faceDetNN.setNumNCEPerInferenceThread(1)
    faceDetNN.input.setBlocking(False)
    faceDetNN.input.setQueueSize(1)

    # Script collecting face detections and stereo depth, sending both to spatial locator
    detectionsDepthToSpatial = pipeline.create(dai.node.Script)
    detectionsDepthToSpatial.setProcessor(dai.ProcessorType.LEON_CSS)
    detectionsDepthToSpatial.setScript(load_script('device_scripts/detections_depth_to_spatial.py'))
    detectionsDepthToSpatial.inputs['depth'].setBlocking(False)
    detectionsDepthToSpatial.inputs['depth'].setQueueSize(1)
    detectionsDepthToSpatial.inputs['faces'].setBlocking(False)
    detectionsDepthToSpatial.inputs['faces'].setQueueSize(1)

    # Spatial locator
    spatial = pipeline.create(dai.node.SpatialLocationCalculator)
    roi = dai.SpatialLocationCalculatorConfigData()
    roi.roi = dai.Rect(dai.Point2f(1080 // 2 - 5, 1080 // 2 - 5), dai.Size2f(10, 10))
    roi.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
    spatial.initialConfig.addROI(roi)
    spatial.inputDepth.setBlocking(False)
    spatial.inputDepth.setQueueSize(1)

    # Script collecting face detections, their distances, and video frame, picking out the nearest face, and sending it with the frame out
    faceSpatialSelectionColorSync = pipeline.create(dai.node.Script)
    faceSpatialSelectionColorSync.setProcessor(dai.ProcessorType.LEON_CSS)
    faceSpatialSelectionColorSync.setScript(load_script('device_scripts/nearest_face_with_frame.py'))
    faceSpatialSelectionColorSync.inputs['color'].setBlocking(False)
    faceSpatialSelectionColorSync.inputs['color'].setQueueSize(1)
    faceSpatialSelectionColorSync.inputs['faces'].setBlocking(False)
    faceSpatialSelectionColorSync.inputs['faces'].setQueueSize(1)
    faceSpatialSelectionColorSync.inputs['spatial'].setBlocking(False)
    faceSpatialSelectionColorSync.inputs['spatial'].setQueueSize(1)

    # IO

    # Color out
    colorXout = pipeline.create(dai.node.XLinkOut)
    colorXout.setStreamName('color')
    colorXout.input.setBlocking(False)
    colorXout.input.setQueueSize(1)
    # Stereo config out
    stereoCfgXout = pipeline.create(dai.node.XLinkOut)
    stereoCfgXout.setStreamName('stereo_cfg')
    stereoCfgXout.input.setBlocking(False)
    stereoCfgXout.input.setQueueSize(1)
    # Depth out
    depthXout = pipeline.create(dai.node.XLinkOut)
    depthXout.setStreamName('depth')
    depthXout.input.setBlocking(False)
    depthXout.input.setQueueSize(1)
    # Nearest face detection out
    nearestFaceXout = pipeline.create(dai.node.XLinkOut)
    nearestFaceXout.setStreamName('nearest_face')
    nearestFaceXout.input.setBlocking(False)
    nearestFaceXout.input.setQueueSize(1)

    # Linking

    # Color preview -> preview flip
    color.preview.link(previewFlip.inputImage)
    # Flipped preview -> face detection
    previewFlip.out.link(faceDetNN.input)
    # Mono left/right -> stereo
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    # Color -> color flip
    color.video.link(colorFlip.inputImage)
    # Color flip -> nearest face picker
    colorFlip.out.link(faceSpatialSelectionColorSync.inputs['color'])
    # Stereo depth -> depth crop
    stereo.depth.link(depthCropFlip.inputImage)
    # Depth crop -> detections-depth to spatial
    depthCropFlip.out.link(detectionsDepthToSpatial.inputs['depth'])
    # Stereo config -> stereo config out
    stereo.outConfig.link(stereoCfgXout.input) # type: ignore
    # Depth crop -> depth out
    depthCropFlip.out.link(depthXout.input) # type: ignore
    # Face detection -> detections-depth to spatial
    faceDetNN.out.link(detectionsDepthToSpatial.inputs['faces'])
    # Face detection -> nearest face picker
    faceDetNN.out.link(faceSpatialSelectionColorSync.inputs['faces'])
    # Detections-depth to spatial cfg/depth -> spatial locator cfg/depth
    detectionsDepthToSpatial.outputs['depth_cfg'].link(spatial.inputConfig)
    detectionsDepthToSpatial.outputs['depth_img'].link(spatial.inputDepth)
    # Spatial locator -> nearest face picker
    spatial.out.link(faceSpatialSelectionColorSync.inputs['spatial'])
    # Nearest face picker - face -> nearest face out
    faceSpatialSelectionColorSync.outputs['nearest_face'].link(nearestFaceXout.input)
    # Nearest face picker - color -> color out
    faceSpatialSelectionColorSync.outputs['color_pass'].link(colorXout.input)

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

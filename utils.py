from dataclasses import dataclass
import depthai as dai
import numpy as np
import cv2
import math

BASELINE = 75
FOV = 73

def depth_to_cv_frame(image: dai.ImgFrame, cfg: dai.StereoDepthConfig) -> np.ndarray:
    max_disparity = cfg.getMaxDisparity()
    subpixel_levels = math.pow(2, cfg.get().algorithmControl.subpixelFractionalBits)
    subpixel = cfg.get().algorithmControl.enableSubpixel
    if subpixel:
        disparity_integer_levels = max_disparity / subpixel_levels
    else:
        disparity_integer_levels = max_disparity
    
    frame = image.getFrame()
    focal = frame.shape[1] / (2 * math.tan(FOV / 2 / 180 * math.pi))
    disparity_scale_factor = BASELINE * focal

    with np.errstate(divide='ignore'):
        frame = disparity_scale_factor / frame

    frame = (frame * 255. / disparity_integer_levels).astype(np.uint8)
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
    return frame

def process_detection(img: dai.ImgFrame, xmin, ymin, xmax, ymax) -> dai.Rect:
    return dai.Rect(dai.Point2f(img.getWidth() - xmax, ymin),
                    dai.Point2f(img.getWidth() - xmin, ymax))


@dataclass
class Config:
    debug: bool
    screen_rotated: bool
    show_depth: bool

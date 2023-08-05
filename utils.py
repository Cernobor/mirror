from dataclasses import dataclass
import depthai as dai
import numpy as np
import cv2
import math
from typing import List, Optional, Tuple, Union

BASELINE = 75
FOV = 73

def depth_to_cv_frame(frame: np.ndarray, cfg: dai.StereoDepthConfig) -> np.ndarray:
    max_disparity = cfg.getMaxDisparity()
    subpixel_levels = math.pow(2, cfg.get().algorithmControl.subpixelFractionalBits)
    subpixel = cfg.get().algorithmControl.enableSubpixel
    if subpixel:
        disparity_integer_levels = max_disparity / subpixel_levels
    else:
        disparity_integer_levels = max_disparity
    
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

def conv_comb(a, b, fac_a):
    return fac_a * a + (1 - fac_a) * b


T = TypeVar('T', int, Tuple[int, ...])
def scale(value: T, numerator: int, denominator: int) -> T:
    if isinstance(value, tuple):
        return tuple((v * numerator) // denominator for v in value)
    else:
        return (value * numerator) // denominator


@dataclass
class Config:
    debug: bool = False
    screen_rotated: bool = False
    camera_flipped: bool = False
    depth: int = 0
    halo_common: str = ''
    halo_special: str = ''
    halo_common_blow_factor: float = ''
    halo_special_blow_factor: float = ''
    halo_position_mixing_coef: float = ''
    halo_decay_coef: float = 0
    background_stars_no: int = 0
    common_constellations: str = ''
    special_constellations: str = ''
    halo_fade_in_time: float = 1
    constellation_fade_in_time: float = 1
    halo_delay_time: float = 0
    constellation_delay_time: float = 0
    special_trigger_file: Optional[str] = None
    final_trigger_file: Optional[str] = None
    special_trigger_pin: Optional[int] = None
    final_trigger_pin: Optional[int] = None
    global_res_scale: Tuple[int, int] = (1, 1)
    video_res_scale: Tuple[int, int] = (1, 1)

    margin_bottom: float = 0.21875
    margin_top: float = 0.0
    margin_left: float = 0.0
    constellations_x_size: float = 0.38889

    def validate(self) -> List[str]:
        res = []
        if self.special_trigger_file is None and self.special_trigger_pin:
            res.append('exactly one of special_trigger_file and special_trigger_pin must be specified')
        if self.final_trigger_file is None and self.final_trigger_pin:
            res.append('exactly one of final_trigger_file and final_trigger_pin must be specified')
        if not (0 <= self.halo_position_mixing_coef <= 1):
            res.append('halo_position_mixing_coef must be in the range [0, 1]')
        if not (0 <= self.halo_decay_coef < 1):
            res.append('halo_decay_coef must be in the range [0, 1)')
        
        return res

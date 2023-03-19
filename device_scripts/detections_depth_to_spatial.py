from depthai import Rect, Point2f, SpatialLocationCalculatorConfig, SpatialLocationCalculatorConfigData, SpatialLocationCalculatorAlgorithm, DataInputQueue, DataOutputQueue, ImgFrame
from typing import Mapping, Union


class Node:
    def __init__(self) -> None:
        self.io: Mapping[str, Union[DataInputQueue, DataOutputQueue]] = dict()

    def warn(self, msg: str):
        pass


node = Node()
# --start--
import time  # noqa

while True:
    time.sleep(0.001)

    depth: ImgFrame = node.io['depth'].tryGet()
    if depth is None:
        continue
    seq = depth.getSequenceNum()
    #node.warn(f'Got depth[{seq}]')
    
    rois = []
    # Dummy ROI
    roi = SpatialLocationCalculatorConfigData()
    roi.roi = Rect(Point2f(depth.getWidth() // 2 - 5, depth.getHeight() // 2 - 5),
                   Point2f(depth.getWidth() // 2 + 5, depth.getHeight() // 2 + 5))
    roi.calculationAlgorithm = SpatialLocationCalculatorAlgorithm.AVERAGE
    rois.append(roi)
    cfg = SpatialLocationCalculatorConfig()
    cfg.setROIs(rois)
    node.io['depth_cfg'].send(cfg)
    node.io['depth_img'].send(depth)
    del cfg
    del depth

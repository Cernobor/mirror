from depthai import Rect, Point2f, SpatialLocationCalculatorConfig, SpatialLocationCalculatorConfigData, SpatialLocationCalculatorAlgorithm, DataInputQueue, DataOutputQueue, ImgFrame, ImgDetections
from typing import Mapping, Union


class Node:
    def __init__(self) -> None:
        self.io: Mapping[str, Union[DataInputQueue, DataOutputQueue]] = dict()

    def warn(self, msg: str):
        pass


node = Node()
# --start--
import time  # noqa

BUFFER_SIZE = 10
NEIGHBOURHOOD = 3

class RB:
    def __init__(self, size) -> None:
        self._buf = [None] * size
        self._size = size
        self._start = 0
        self._end = 0
        self._len = 0
    
    def push(self, item):
        if self._len + 1 >= self._size:
            raise ValueError('buffer full')
        self._buf[self._end] = item
        self._end = (self._end + 1) % self._size
        self._len += 1
    
    def pop(self):
        if self._len <= 0:
            return None
        res = self._buf[self._start]
        self._buf[self._start] = None
        self._start = (self._start + 1) % self._size
        self._len -= 1
        return res

    def peek(self):
        if self._len <= 0:
            return None
        return self._buf[self._start]

depth_buffer = RB(BUFFER_SIZE)
faces_buffer = RB(BUFFER_SIZE)

while True:
    time.sleep(0.001)

    change = False
    depth: ImgFrame = node.io['depth'].tryGet()
    if depth is not None:
        seq = depth.getSequenceNum()
        #node.warn(f'Got depth[{seq}]')
        if faces_buffer.peek() is None or seq >= faces_buffer.peek().getSequenceNum():
            depth_buffer.push(depth)
        #node.warn(f'db {[x.getSequenceNum() if x is not None else "-" for x in depth_buffer._buf]}')
        #node.warn(f'fb {[x.getSequenceNum() if x is not None else "-" for x in faces_buffer._buf]}')

    faces: ImgDetections = node.io['faces'].tryGet()
    if faces is not None:
        seq = faces.getSequenceNum()
        #node.warn(f'Got faces[{seq}]')
        if depth_buffer.peek() is None or seq >= depth_buffer.peek().getSequenceNum():
            faces_buffer.push(faces)
        #node.warn(f'db {[x.getSequenceNum() if x is not None else "-" for x in depth_buffer._buf]}')
        #node.warn(f'fb {[x.getSequenceNum() if x is not None else "-" for x in faces_buffer._buf]}')
        
    
    depth = depth_buffer.peek()
    faces = faces_buffer.peek()
    while depth is not None and faces is not None:
        if depth.getSequenceNum() < faces.getSequenceNum():
            depth_buffer.pop()
            depth = depth_buffer.peek()
        elif depth.getSequenceNum() > faces.getSequenceNum():
            faces_buffer.pop()
            faces = faces_buffer.peek()
        else:
            break
    if depth is None or faces is None:
        continue
    depth_buffer.pop()
    faces_buffer.pop()
    #node.warn(f'db {[x.getSequenceNum() if x is not None else "-" for x in depth_buffer._buf]}')
    #node.warn(f'fb {[x.getSequenceNum() if x is not None else "-" for x in faces_buffer._buf]}')
    #node.warn(f'Got depth[{depth.getSequenceNum()}] and faces[{faces.getSequenceNum()}]')
    
    rois = []
    for det in faces.detections:
        rect = Rect(Point2f(det.xmin, det.ymin), Point2f(det.xmax, det.ymax))
        rect = rect.denormalize(depth.getWidth(), depth.getHeight())
        center = Point2f(int((rect.topLeft().x + rect.bottomRight().x) / 2),
                         int((rect.topLeft().y + rect.bottomRight().y) / 2))
        
        roi = SpatialLocationCalculatorConfigData()
        roi.roi = Rect(Point2f(center.x - NEIGHBOURHOOD, center.y - NEIGHBOURHOOD),
                       Point2f(center.x + NEIGHBOURHOOD, center.y + NEIGHBOURHOOD))
        roi.calculationAlgorithm = SpatialLocationCalculatorAlgorithm.AVERAGE
        rois.append(roi)
    # Dummy ROI in case there were no detections
    if len(rois) == 0:
        roi = SpatialLocationCalculatorConfigData()
        roi.roi = Rect(Point2f(depth.getWidth() // 2 - NEIGHBOURHOOD, depth.getHeight() // 2 - NEIGHBOURHOOD),
                       Point2f(depth.getWidth() // 2 + NEIGHBOURHOOD, depth.getHeight() // 2 + NEIGHBOURHOOD))
        roi.calculationAlgorithm = SpatialLocationCalculatorAlgorithm.AVERAGE
        rois.append(roi)
    cfg = SpatialLocationCalculatorConfig()
    cfg.setROIs(rois)
    node.io['depth_cfg'].send(cfg)
    node.io['depth_img'].send(depth)
    del cfg
    del depth

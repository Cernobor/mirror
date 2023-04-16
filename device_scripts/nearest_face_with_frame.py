from depthai import Rect, Point2f, SpatialLocationCalculatorConfig, SpatialLocationCalculatorConfigData, SpatialLocationCalculatorAlgorithm, DataInputQueue, DataOutputQueue, ImgFrame, ImgDetections, SpatialLocationCalculatorData, NNData
from typing import Mapping, Union


class Node:
    def __init__(self) -> None:
        self.io: Mapping[str, Union[DataInputQueue, DataOutputQueue]] = dict()

    def warn(self, msg: str):
        pass


node = Node()
# --start--
import time  # noqa
import struct # noqa

BUFFER_SIZE = 10
NEIGHBOURHOOD = 3
INF = float('inf')

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

color_buffer = RB(BUFFER_SIZE)
faces_buffer = RB(BUFFER_SIZE)
spatial_buffer = RB(BUFFER_SIZE)

while True:
    time.sleep(0.001)

    change = False
    color: ImgFrame = node.io['color'].tryGet()
    if color is not None:
        seq = color.getSequenceNum()
        #node.warn(f'Got color[{seq}]')
        if (faces_buffer.peek() is None or seq >= faces_buffer.peek().getSequenceNum() or
            spatial_buffer.peek() is None or seq >= spatial_buffer.peek().getSequenceNum()):
            color_buffer.push(color)
        #node.warn(f'cb {[x.getSequenceNum() if x is not None else "-" for x in color_buffer._buf]}')
        #node.warn(f'fb {[x.getSequenceNum() if x is not None else "-" for x in faces_buffer._buf]}')
        #node.warn(f'sb {[x.getSequenceNum() if x is not None else "-" for x in spatial_buffer._buf]}')

    faces: ImgDetections = node.io['faces'].tryGet()
    if faces is not None:
        seq = faces.getSequenceNum()
        #node.warn(f'Got faces[{seq}]: {[(f.xmin, f.xmax, f.ymin, f.ymax) for f in faces.detections]}')
        if (color_buffer.peek() is None or seq >= color_buffer.peek().getSequenceNum() or
            spatial_buffer.peek() is None or seq >= spatial_buffer.peek().getSequenceNum()):
            faces_buffer.push(faces)
        #node.warn(f'cb {[x.getSequenceNum() if x is not None else "-" for x in color_buffer._buf]}')
        #node.warn(f'fb {[x.getSequenceNum() if x is not None else "-" for x in faces_buffer._buf]}')
        #node.warn(f'sb {[x.getSequenceNum() if x is not None else "-" for x in spatial_buffer._buf]}')
    
    spatial: SpatialLocationCalculatorData = node.io['spatial'].tryGet()
    if spatial is not None:
        seq = spatial.getSequenceNum()
        #node.warn(f'Got spatial[{seq}]: {spatial}')
        if (color_buffer.peek() is None or seq >= color_buffer.peek().getSequenceNum() or
            faces_buffer.peek() is None or seq >= faces_buffer.peek().getSequenceNum()):
            spatial_buffer.push(spatial)
        #node.warn(f'cb {[x.getSequenceNum() if x is not None else "-" for x in color_buffer._buf]}')
        #node.warn(f'fb {[x.getSequenceNum() if x is not None else "-" for x in faces_buffer._buf]}')
        #node.warn(f'sb {[x.getSequenceNum() if x is not None else "-" for x in spatial_buffer._buf]}')
        
    
    color = color_buffer.peek()
    faces = faces_buffer.peek()
    spatial = spatial_buffer.peek()
    while color is not None and faces is not None and spatial is not None:
        cs = color.getSequenceNum()
        fs = faces.getSequenceNum()
        ss = spatial.getSequenceNum()
        if cs < fs and cs < ss:
            color_buffer.pop()
            color = color_buffer.peek()
        elif fs < cs and fs < ss:
            faces_buffer.pop()
            faces = faces_buffer.peek()
        elif ss < cs and ss < fs:
            spatial_buffer.pop()
            spatial = spatial_buffer.peek()
        elif cs == fs and cs < ss:
            color_buffer.pop()
            color = color_buffer.peek()
            faces_buffer.pop()
            faces = faces_buffer.peek()
        elif fs == ss and fs < cs:
            faces_buffer.pop()
            faces = faces_buffer.peek()
            spatial_buffer.pop()
            spatial = spatial_buffer.peek()
        elif cs == ss and cs < fs:
            color_buffer.pop()
            color = color_buffer.peek()
            spatial_buffer.pop()
            spatial = spatial_buffer.peek()
        else:
            break
    if color is None or faces is None or spatial is None:
        continue
    color_buffer.pop()
    faces_buffer.pop()
    spatial_buffer.pop()
    #node.warn(f'cb {[x.getSequenceNum() if x is not None else "-" for x in color_buffer._buf]}')
    #node.warn(f'fb {[x.getSequenceNum() if x is not None else "-" for x in faces_buffer._buf]}')
    #node.warn(f'sb {[x.getSequenceNum() if x is not None else "-" for x in spatial_buffer._buf]}')
    #node.warn(f'Got color[{color.getSequenceNum()}]: {color}')
    #node.warn(f'Got faces[{faces.getSequenceNum()}]: {[(f.xmin, f.xmax, f.ymin, f.ymax) for f in faces.detections]}')
    #node.warn(f'Got spatial[{spatial.getSequenceNum()}]: {spatial}')
    
    locations = spatial.getSpatialLocations()
    #node.warn(f'Got spatial locations: {locations}')
    min_i = None
    min_z = INF
    for i, det in enumerate(faces.detections):
        l = locations[i]
        #node.warn(f'location[{i}]: {l.depthAverage}')
        if l.depthAverage < min_z:
            min_i = i
            min_z = l.depthAverage
    nearest_face = NNData(12)
    nearest_face.setSequenceNum(color.getSequenceNum())
    if min_i is not None:
        face = faces.detections[min_i]
        #node.warn(f'{int(face.xmin), int(face.ymin), int(face.xmax), int(face.ymax)}')
        #data = struct.pack('<4i', int(face.xmin), int(face.ymin), int(face.xmax), int(face.ymax))
        data = [face.xmin, face.ymin, face.xmax, face.ymax]
        #node.warn(f'{data}')
        #nearest_face.setLayer('bbox', [int(x) for x in data])
        nearest_face.setLayer('bbox', data)
    #node.warn(f'sending {nearest_face}')
    node.io['nearest_face'].send(nearest_face)
    #node.warn(f'sending {color}')
    node.io['color_pass'].send(color)

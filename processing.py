from dataclasses import dataclass
from typing import Optional, Tuple
import cv2


@dataclass
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    def top_left(self) -> Tuple[int, int]:
        return (self.x0, self.y0)
    
    def bottom_right(self) -> Tuple[int, int]:
        return (self.x1, self.y1)


class Processor:
    def __init__(self) -> None:
        self.bbox = BBox(0, 0, 0, 0)
        self.last_true_seq = -1
    
    def process(self, image: cv2.Mat, face_bbox: Optional[BBox], seq: int) -> cv2.Mat:
        thickness = 10
        if face_bbox is None:
            coef = (2 ** (0.3 * (self.last_true_seq - seq)))
            thickness = thickness * coef
            if int(thickness) <= 0:
                return image
        else:
            self.bbox = face_bbox
            self.last_true_seq = seq
        
        return cv2.rectangle(image, self.bbox.top_left(), self.bbox.bottom_right(), (255, 255, 255), int(thickness))

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
        self.offset = 840
    
    def process(self, face_image: cv2.Mat, dest_image: cv2.Mat, face_bbox: Optional[BBox], seq: int) -> cv2.Mat:
        """Receives a 1:1 (1080x1080) image containing the face, the bounding box of the face, and sequential number of the frame.
        Returns a 16:9 (1920x1080) image containing the processed face image and anything in the 840x1080 space above."""
        dest_image[:self.offset, :, :] = seq % 256
        dest_image[self.offset:, :, :] = face_image

        thickness = 10
        if face_bbox is None:
            coef = (2 ** (0.3 * (self.last_true_seq - seq)))
            thickness = thickness * coef
            if int(thickness) <= 0:
                return
        else:
            tl = face_bbox.top_left()
            br = face_bbox.bottom_right()
            self.bbox = BBox(tl[0], tl[1] + self.offset, br[0], br[1] + self.offset)
            self.last_true_seq = seq
        

        cv2.rectangle(dest_image, self.bbox.top_left(), self.bbox.bottom_right(), (255, 255, 255), int(thickness))

from dataclasses import dataclass
import time
from typing import Optional, Tuple
import cv2
import pygame
import numpy as np


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


class Renderer:
    def __init__(self, display_size: Tuple[int, int], image_size: Tuple[int, int], debug=True) -> None:
        self.bbox = BBox(0, 0, 0, 0)
        self.last_true_seq = -1
        self.debug = debug

        self.display_size = display_size
        self.image_size = image_size
        print('display_size', display_size)
        print('image_size', image_size)
        self.vertical_diff = self.display_size[0] - self.image_size[0]
        self.horizontal_diff = self.display_size[1] - self.image_size[1]
        print('{vertical,horizontal}_diff', self.vertical_diff, self.horizontal_diff)

        self._prepare()

    def _prepare(self):
        self.dest_image = np.zeros((self.display_size[0], self.display_size[1], 3), dtype=np.uint8)

        ds = self.display_size
        fs = pygame.FULLSCREEN
        if self.debug:
            ds = (self.display_size[1] // 4, self.display_size[0] // 4)
            fs = 0
        self.pg_screen = pygame.display.set_mode(size=ds, flags=fs)

        self.time = time.time()
    
    def render(self, face_image: cv2.Mat, face_bbox: Optional[BBox], seq: int):
        """Receives an image containing the face, the bounding box of the face, and sequential number of the frame.
        Renders the final image.
        
        Returns True when user wants to terminate."""
        #self.dest_image[:, :, :] = 128
        self.dest_image[:self.vertical_diff, :, :] = 0
        self.dest_image[self.vertical_diff:, :, :] = face_image
        now = time.time()

        thickness = 10
        if face_bbox is None:
            coef = (2 ** (0.3 * (self.last_true_seq - seq)))
            thickness = thickness * coef
        else:
            tl = face_bbox.top_left()
            br = face_bbox.bottom_right()
            self.bbox = BBox(tl[0], tl[1] + self.vertical_diff, br[0], br[1] + self.vertical_diff)
            self.last_true_seq = seq
        
        it = int(thickness)
        if it > 0:
            cv2.rectangle(self.dest_image, self.bbox.top_left(), self.bbox.bottom_right(), (255, 255, 255), it)
            pass

        cv2.flip(self.dest_image, 1, self.dest_image)
        r = self.dest_image
        if self.debug:
            r = cv2.resize(self.dest_image, (self.display_size[1] // 4, self.display_size[0] // 4))
            r = np.rot90(r)
        pg_frame = pygame.surfarray.make_surface(cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
        self.pg_screen.blit(pg_frame, (0, 0))

        pygame.display.update()

        self.time = now
        for event in pygame.event.get():
            if event.type == pygame.KEYUP and event.unicode == 'q':
                pygame.quit()
                return True
        return False

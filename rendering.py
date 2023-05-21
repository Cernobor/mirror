from dataclasses import dataclass
import random
import time
from typing import Optional, Tuple
import cv2
import pygame
import numpy as np
from particlepy import particlepy


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

    def x_bounds(self, flip: bool=False, offset: int=0) -> Tuple[int, int]:
        if flip:
            return -self.x1 + offset, -self.x0 + offset
        else:
            return self.x0 + offset, self.x1 + offset
    
    def y_bounds(self, flip: bool=False, offset: int=0) -> Tuple[int, int]:
        if flip:
            return -self.y1 + offset, -self.y0 + offset
        else:
            return self.y0 + offset, self.y1 + offset


class Renderer:
    def __init__(self, display_size: Tuple[int, int], image_size: Tuple[int, int], screen_rotated: bool, debug: bool=False) -> None:
        self.display_size = display_size
        self.image_size = image_size
        print('display_size', display_size)
        print('image_size', image_size)
        self.vertical_diff = self.display_size[0] - self.image_size[0]
        self.horizontal_diff = self.display_size[1] - self.image_size[1]
        print('{vertical,horizontal}_diff', self.vertical_diff, self.horizontal_diff)

        self.screen_rotated = screen_rotated
        self.debug = debug

        self._prepare()

    def _prepare(self):
        pygame.init()
        self.bbox = BBox(0, 0, 0, 0)
        self.first_true_seq = -1
        self.last_true_seq = -1
        
        ds = self.display_size
        fs = pygame.FULLSCREEN
        if self.debug:
            ds = (self.display_size[1] // 4, self.display_size[0] // 4)
            fs = 0
        elif not self.screen_rotated:
            ds = tuple(reversed(ds))
        self.pg_screen = pygame.display.set_mode(size=ds, flags=fs)
        self.psys = particlepy.particle.ParticleSystem()
        self.rng = np.random.default_rng()

        self.max_emit = 200
        if self.debug:
            self.max_emit = self.max_emit // 4
        self.max_thickness = 10
        if self.debug:
            self.max_thickness = self.max_thickness // 4
        self.cov_scale = 20
        if self.debug:
            self.cov_scale = self.cov_scale / 4
        
        self.horizontal_idx = 1
        self.vertical_idx = 0
        if self.debug or self.screen_rotated:
            self.horizontal_idx = 0
            self.vertical_idx = 1

        self.time = time.time()
    
    def render(self, frame: cv2.Mat, face_bbox: Optional[BBox], seq: int):
        """Receives an image containing the face, the bounding box of the face, and sequential number of the frame.
        Renders the final image.
        
        Returns True when user wants to terminate."""
        now = time.time()
        delta = now - self.time

        if self.debug:
            frame = cv2.resize(frame, (self.image_size[1] // 4, self.image_size[0] // 4))

        emit = self.max_emit
        thickness = self.max_thickness
        if face_bbox is None:
            coef = (2 ** (0.3 * (self.last_true_seq - seq)))
            emit *= coef
            thickness *= coef
            self.first_true_seq = -1
        else:
            if self.first_true_seq == -1:
                self.first_true_seq = seq
            
            coef = (2 ** (0.34 * min(seq - self.first_true_seq - 10, 0)))
            emit *= coef
            thickness *= coef
            tl = face_bbox.top_left()
            br = face_bbox.bottom_right()
            self.bbox = BBox(tl[0], tl[1], br[0], br[1])
            
            if self.debug:
                self.bbox = BBox(self.bbox.x0 // 4, self.bbox.y0 // 4, self.bbox.x1 // 4, self.bbox.y1 // 4)
            
            self.last_true_seq = seq
        
        it = int(thickness)
        if it > 0 and self.debug:
            cv2.rectangle(frame, self.bbox.top_left(), self.bbox.bottom_right(), (255, 255, 255), it)
            pass

        cv2.flip(frame, 1, frame)
        if self.debug or self.screen_rotated:
            frame = np.rot90(frame)
        
        pg_face = pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pg_psys = pygame.Surface(pg_face.get_size())
        #pg_psys.fill((255, 255, 255))

        self.psys.update(delta_time=delta)
        for _ in range(int(emit)):
            if self.debug or self.screen_rotated:
                bounds1 = self.bbox.x_bounds()
                bounds2 = self.bbox.y_bounds()
            else:
                bounds1 = self.bbox.y_bounds()
                bounds2 = self.bbox.x_bounds(flip=True, offset=self.image_size[0])
            center = np.array((np.mean(bounds1), np.mean(bounds2)))
            pos = self.rng.multivariate_normal(
                mean=center,
                cov=np.array([[(bounds1[1] - bounds1[0]) * self.cov_scale, 0],
                                [0, (bounds2[1] - bounds2[0]) * self.cov_scale]]))
            vel = (random.uniform(0, 200), (pos[self.horizontal_idx] - center[self.horizontal_idx]) * 2)
            if self.debug:
                vel = (vel[0] / 4, vel[1])
            if self.debug or self.screen_rotated:
                vel = tuple(reversed(vel))
            self.psys.emit(
                particle=particlepy.particle.Particle(
                    shape=particlepy.shape.Circle(
                        radius=10,
                        color=(255, 0, 0),
                        alpha=64
                    ),
                    position=(int(pos[0]), int(pos[1])),
                    velocity=vel,
                    delta_radius=.75,
                )
            )
        dv = -1500 * delta
        if self.debug:
            dv /= 4
        for p in self.psys.particles:
            p.velocity[self.vertical_idx] += dv
        self.psys.make_shape()

        self.psys.render(surface=pg_psys)

        pg_face.blit(pg_psys, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        if self.debug:
            self.pg_screen.blit(pg_face, (0, self.vertical_diff // 4))
        elif not self.screen_rotated:
            self.pg_screen.blit(pg_face, (self.vertical_diff, 0))
        else:
            self.pg_screen.blit(pg_face, (0, self.vertical_diff))

        pygame.display.update()

        self.time = now
        for event in pygame.event.get():
            if event.type == pygame.KEYUP and event.unicode == 'q':
                pygame.quit()
                return True
        return False

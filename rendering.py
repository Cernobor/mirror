from dataclasses import dataclass
import math
import random
import time
from typing import Any, Optional, Tuple, TypeVar, Union
import cv2
import pygame
import numpy as np
from particlepy import particlepy

T = TypeVar('T')


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
        self.debug_divisor = 2
        
        ds = self.display_size
        fs = pygame.FULLSCREEN
        if self.debug:
            ds = (self.display_size[1] // self.debug_divisor, self.display_size[0] // self.debug_divisor)
            fs = 0

            size = (self.display_size[1] // self.debug_divisor, self.vertical_diff // self.debug_divisor)
            dim = min(size)
            self.indicator_offset = ((max(size) - dim) // 2, min(size) - dim)
        elif not self.screen_rotated:
            ds = tuple(reversed(ds))

            size = (self.vertical_diff, self.display_size[1])
            dim = min(size)
            self.indicator_offset = (min(size) - dim, (max(size) - dim) // 2)
        else:
            size = (self.display_size[1], self.vertical_diff)
            dim = min(size)
            self.indicator_offset = ((max(size) - dim) // 2, min(size) - dim)
        self.pg_screen = pygame.display.set_mode(size=ds, flags=fs)
        self.pg_indicator = pygame.Surface((dim, dim))
        self.psys = particlepy.particle.ParticleSystem()
        self.rng = np.random.default_rng()

        self.max_emit = 200
        if self.debug:
            self.max_emit = self.max_emit // self.debug_divisor
        self.max_thickness = 10
        if self.debug:
            self.max_thickness = self.max_thickness // self.debug_divisor
        self.cov_scale = 20
        if self.debug:
            self.cov_scale = self.cov_scale / self.debug_divisor
        
        self.horizontal_idx = 1
        self.vertical_idx = 0
        if self.debug or self.screen_rotated:
            self.horizontal_idx = 0
            self.vertical_idx = 1
        
        self.stars = dict()
        for size in [s for s in range(2, 50)]:
            coord_vec = np.linspace(-1, 1, size)
            xx, yy = np.meshgrid(coord_vec, coord_vec)
            brightness = 1 - np.sqrt(xx**2 + yy**2)
            brightness[brightness < 0] = 0
            brightness[brightness > 0.5] = 1
            brightness[brightness < 1] /= brightness[brightness < 1].max()
            brightness = np.rint(brightness * 255).astype(int)
            color = np.stack((brightness,) * 3, axis=-1)
            self.stars[size] = {
                'data': color,
                'center_offset': (-size // 2, -size // 2)
            }
        self.background_stars = []
        for _ in range(250):
            stars = [self.stars[s] for s in range(4, 8)]
            mult_range = (0.4, 0.7)
            mult_speed = 0.15 * (mult_range[1] - mult_range[0]) * (2 * random.random() - 1)
            mult = (mult_range[1] - mult_range[0]) * random.random() + mult_range[0]
            star = {
                'coords': np.random.rand(2),
                'star': random.choice(stars),
                'mult': mult,
                'mult_range': mult_range,
                'mult_speed': mult_speed,
                'background': True
            }
            self.background_stars.append(star)
        t = np.linspace(0, 1, 30) ** 1
        r_in = 0.5 * 0.2
        r_out = 0.5 * 0.75
        rots = 5.0
        v = r_out - r_in
        x = (v * t + r_in) * np.cos(2 * np.pi * rots * t) + 0.5
        y = (v * t + r_in) * np.sin(2 * np.pi * rots * t) + 0.5
        foreground_star_coords = np.stack((x, y), axis=-1)
        print(foreground_star_coords)
        for c in foreground_star_coords:
            stars = [self.stars[s] for s in range(12, 25)]
            mult_range = (0.9, 1)
            mult_speed = 0.15 * (mult_range[1] - mult_range[0]) * (2 * random.random() - 1)
            mult = (mult_range[1] - mult_range[0]) * random.random() + mult_range[0]
            self.background_stars.append({
                'coords': c,
                'star': random.choice(stars),
                'mult': mult,
                'mult_range': mult_range,
                'mult_speed': mult_speed,
                'background': False
            })

        self.time = time.time()
    
    def update_particles(self, emit: int):
        now = time.time()
        delta = now - self.time
        self.time = now

        self.psys.update(delta_time=delta)
        if self.debug or self.screen_rotated:
            bounds1 = self.bbox.x_bounds()
            bounds2 = self.bbox.y_bounds()
        else:
            bounds1 = self.bbox.y_bounds()
            bounds2 = self.bbox.x_bounds(flip=True, offset=self.image_size[0])
        center = np.array((np.mean(bounds1), np.mean(bounds2)))
        for _ in range(int(emit)):
            pos = self.rng.multivariate_normal(
                mean=center,
                cov=np.array([[(bounds1[1] - bounds1[0]) * self.cov_scale, 0],
                                [0, (bounds2[1] - bounds2[0]) * self.cov_scale]]))
            vel = (random.uniform(0, 200), (pos[self.horizontal_idx] - center[self.horizontal_idx]) * 2)
            if self.debug:
                vel = (vel[0] / self.debug_divisor, vel[1])
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
                    data={'emit_center': center}
                )
            )
        dv = -1500 * delta
        if self.debug:
            dv /= 4
        for p in self.psys.particles:
            p.velocity[self.vertical_idx] += dv

    def render_face(self, frame: cv2.Mat, face_bbox: Optional[BBox], seq: int):
        is_face = face_bbox is not None
        if is_face:
            #print(seq - self.first_true_seq)
            coef = (2 ** (0.34 * min(seq - self.first_true_seq - 10, 0)))
            tl = face_bbox.top_left()
            br = face_bbox.bottom_right()
            self.bbox = BBox(tl[0], tl[1], br[0], br[1])
            if self.debug:
                self.bbox = BBox(self.bbox.x0 // self.debug_divisor, self.bbox.y0 // self.debug_divisor, self.bbox.x1 // self.debug_divisor, self.bbox.y1 // self.debug_divisor)
        else:
            coef = (2 ** (0.3 * (self.last_true_seq - seq)))
        
        emit = self.max_emit * coef
        thickness = self.max_thickness * coef
        
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

        self.update_particles(emit)
        
        self.psys.make_shape()
        self.psys.render(surface=pg_psys)

        pg_face.blit(pg_psys, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        if self.debug:
            self.pg_screen.blit(pg_face, (0, self.vertical_diff // self.debug_divisor))
        elif not self.screen_rotated:
            self.pg_screen.blit(pg_face, (self.vertical_diff, 0))
        else:
            self.pg_screen.blit(pg_face, (0, self.vertical_diff))
    
    def render_indicator(self, is_face: bool, seq: int):
        self.pg_indicator.fill((0, 0, 0))

        delay_in_time = 24
        fade_time = 48
        if is_face:
            t = seq - self.first_true_seq
            coef = max(min((t - delay_in_time) / fade_time, 1), 0)
        else:
            t = seq - self.last_true_seq
            coef = -min(t / fade_time - 1, 0)
        
        size = self.pg_indicator.get_size()
        for bs in self.background_stars:
            s = bs['star']
            d = s['data']
            coords = self.switch_coords(bs['coords'])
            coords = (
                int(coords[0] * size[0]) + s['center_offset'][0],
                int(coords[1] * size[1]) + s['center_offset'][1]
            )

            m = bs['mult']
            mr = bs['mult_range']
            ms = bs['mult_speed']
            bs['mult'] = min(max(m + ms, mr[0]), mr[1])
            bs['mult_speed'] += 0.15 * (mr[1] - mr[0]) * (2 * random.random() - 1)
            if not bs['background']:
                m *= coef
            if int(255 * m) < 1:
                continue

            surf = pygame.surfarray.make_surface(d * m)
            self.pg_indicator.blit(surf, coords, special_flags=pygame.BLEND_RGB_ADD)

        self.pg_screen.blit(self.pg_indicator, self.indicator_offset)
    
    def render(self, frame: cv2.Mat, face_bbox: Optional[BBox], seq: int):
        """Receives an image containing the face, the bounding box of the face, and sequential number of the frame.
        Renders the final image.
        
        Returns True when user wants to terminate."""
        if self.debug:
            frame = cv2.resize(frame, (self.image_size[1] // self.debug_divisor, self.image_size[0] // self.debug_divisor))
        
        if face_bbox is not None:
            self.last_true_seq = seq
            if self.first_true_seq == -1:
                self.first_true_seq = seq
        else:
            self.first_true_seq = -1
        
        self.render_face(frame, face_bbox, seq)
        self.render_indicator(face_bbox is not None, seq)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.KEYUP and event.unicode == 'q':
                pygame.quit()
                return True
        return False

    def switch_coords(self, c: Tuple[int, int]) -> Tuple[int, int]:
        if not self.debug:
            return c[1], c[0]
        return c
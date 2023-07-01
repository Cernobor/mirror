from dataclasses import dataclass
import json
import os
import random
import time
from typing import Optional, Tuple, TypeVar
import cv2
import pygame
import numpy as np
import utils

T = TypeVar('T')


@dataclass
class Bounds:
    low: int
    high: int

    def size(self) -> int:
        return abs(self.high - self.low)
    
    def center(self) -> int:
        return (self.low + self.high) // 2


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

    def x_bounds(self, flip: bool=False, offset: int=0) -> Bounds:
        if flip:
            return Bounds(-self.x1 + offset, -self.x0 + offset)
        else:
            return Bounds(self.x0 + offset, self.x1 + offset)
    
    def y_bounds(self, flip: bool=False, offset: int=0) -> Bounds:
        if flip:
            return Bounds(-self.y1 + offset, -self.y0 + offset)
        else:
            return Bounds(self.y0 + offset, self.y1 + offset)
    
    def size(self) -> Tuple[int, int]:
        return (self.x1 - self.x0, self.y1 - self.y0)
    
    def x_size(self) -> int:
        return self.x1 - self.x0
    
    def y_size(self) -> int:
        return self.y1 - self.y0
    
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x0) // 2, (self.y1 + self.y0) // 2)


class Renderer:
    def __init__(self,
                 display_size: Tuple[int, int],
                 image_size: Tuple[int, int],
                 screen_rotated: bool,
                 depth: int,
                 halo_common_dir: str,
                 halo_special_dir: str,
                 background_stars_no: int,
                 common_constellations_js: str,
                 special_constellations_js: str,
                 debug: bool=False) -> None:
        self.display_size = display_size
        self.image_size = image_size
        print('display_size', display_size)
        print('image_size', image_size)
        self.vertical_diff = self.display_size[0] - self.image_size[0]
        self.horizontal_diff = self.display_size[1] - self.image_size[1]
        print('{vertical,horizontal}_diff', self.vertical_diff, self.horizontal_diff)

        self.screen_rotated = screen_rotated
        self.depth = depth
        self.halo_common_dir = halo_common_dir
        self.halo_special_dir = halo_special_dir
        self.background_stars_no = background_stars_no
        self.common_constellations_js = common_constellations_js
        self.special_constellations_js = special_constellations_js
        self.debug = debug

        self._prepare()

    def _prepare(self):
        # general init
        pygame.init()
        self.is_face = False
        self.bbox = BBox(0, 0, 0, 0)
        self.final_trigger_time = None
        self.constellation = None
        self.debug_divisor = 2
        self.special = False
        self.final = False
        self.halo_factor = None
        self.halo_center = None
        
        # setup display sizes and orientations
        self.horizontal_idx = 1
        self.vertical_idx = 0
        if self.debug or self.screen_rotated:
            self.horizontal_idx = 0
            self.vertical_idx = 1

        ds = self.display_size
        fs = pygame.FULLSCREEN
        if self.debug:
            ds = (self.display_size[1] // self.debug_divisor, self.display_size[0] // self.debug_divisor)
            fs = 0

            size = (self.display_size[1] // self.debug_divisor, self.vertical_diff // self.debug_divisor)
            v = min(size) // 2
            h = max(size)
            indicator_size = (v, v)
            self.indicator_offset_l = (0, 0)
            self.indicator_offset_r = (h - v, 0)
        elif not self.screen_rotated:
            ds = tuple(reversed(ds))

            size = (self.vertical_diff, self.display_size[1])
            v = min(size) // 2
            h = max(size)
            indicator_size = (v, v)
            self.indicator_offset_l = (0, 0)
            self.indicator_offset_r = (0, h - v)
        else:
            size = (self.display_size[1], self.vertical_diff)
            v = min(size) // 2
            h = max(size)
            indicator_size = (v, v)
            self.indicator_offset_l = (0, 0)
            self.indicator_offset_r = (h - v, 0)
        self.pg_screen = pygame.display.set_mode(size=ds, flags=fs)
        self.pg_indicator = pygame.Surface(indicator_size)
        self.rng = np.random.default_rng()

        # setup star types
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
        
        # setup background stars
        self.background_stars = []
        for _ in range(self.background_stars_no):
            stars = [self.stars[s] for s in range(4, 8)]
            mult_range = (0.4, 0.7)
            mult_speed = 0.15 * (mult_range[1] - mult_range[0]) * (2 * random.random() - 1)
            mult = (mult_range[1] - mult_range[0]) * random.random() + mult_range[0]
            star = {
                'coords': np.random.rand(2),
                'star': random.choice(stars),
                'mult': mult,
                'mult_range': mult_range,
                'mult_speed': mult_speed
            }
            self.background_stars.append(star)
        
        # setup common constellations
        self.common_constellations = []
        with open(self.common_constellations_js) as f:
            common_constellations = json.load(f)
        for v in common_constellations.values():
            c = []
            for p in v:
                s = p['mag'] * 2 + 10
                star = {
                    'coords': (p['x'], p['y']),
                    'star': self.stars[s]
                }
                c.append(star)
            self.common_constellations.append(c)
        
        # setup special constellations
        self.special_constellations = []
        with open(self.special_constellations_js) as f:
            special_constellations = json.load(f)
        for v in special_constellations.values():
            c = []
            for p in v:
                s = p['mag'] * 2 + 10
                star = {
                    'coords': (p['x'], p['y']),
                    'star': self.stars[s]
                }
                c.append(star)
            self.special_constellations.append(c)
        
        # setup mirror halo
        self.halo_common_imgs = []
        if self.halo_common_dir:
            files = os.listdir(self.halo_common_dir)
            files.sort()
            self.halo_common_imgs = [os.path.join(self.halo_common_dir, f) for f in files]
        self.halo_special_imgs = []
        if self.halo_special_dir:
            files = os.listdir(self.halo_special_dir)
            files.sort()
            self.halo_special_imgs = [os.path.join(self.halo_special_dir, f) for f in files]

        # save start time
        self.time = time.time()
    
    def render_face(self, color: cv2.Mat, depth: cv2.Mat, seq: int):
        if self.depth > 0:
            color[np.logical_or(depth > self.depth, depth < 350), :] = 0
        
        if self.debug and self.is_face:
            cv2.rectangle(color, self.bbox.top_left(), self.bbox.bottom_right(), (255, 255, 255), 1)

        cv2.flip(color, 1, color)
        if self.debug or self.screen_rotated:
            color = np.rot90(color)
        
        pg_face = pygame.surfarray.make_surface(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        
        if self.special and self.halo_special_imgs:
            imgs = self.halo_special_imgs
        elif not self.special and self.halo_common_imgs:
            imgs = self.halo_common_imgs
        else:
            imgs = []
        coef = self.effect_coef(2)
        if coef > 0 and imgs:
            img = pygame.image.load(imgs[seq % len(imgs)]).convert_alpha()
            size = img.get_size()
            img_h = size[1]
            bbox_h = self.bbox.size()[self.vertical_idx]
            factor = bbox_h / img_h * 2.5
            if self.halo_factor is not None:
                self.halo_factor = utils.conv_comb(factor, self.halo_factor, 0.5)
            else:
                self.halo_factor = factor
            scaled = pygame.transform.smoothscale_by(img, self.halo_factor)
            
            if self.debug or self.screen_rotated:
                center = (self.bbox.x_bounds().center(), self.bbox.y_bounds().center())
            else:
                center = (self.bbox.x_bounds(flip=True, offset=self.image_size[0]).center(), self.bbox.y_bounds().center())
            if self.halo_center is not None:
                self.halo_center = (utils.conv_comb(center[0], self.halo_center[0], 0.75), utils.conv_comb(center[1], self.halo_center[1], 0.75))
            else:
                self.halo_center = center
            
            top_left = (self.halo_center[self.horizontal_idx] - scaled.get_width() // 2,
                        self.halo_center[self.vertical_idx] - scaled.get_height() // 2)
            scaled.set_alpha(255 * coef)
            pg_face.blit(scaled, top_left)

        if self.debug:
            self.pg_screen.blit(pg_face, (0, self.vertical_diff // 2 // self.debug_divisor))
        elif not self.screen_rotated:
            self.pg_screen.blit(pg_face, (self.vertical_diff // 2, 0))
        else:
            self.pg_screen.blit(pg_face, (0, self.vertical_diff // 2))
    
    def render_indicator(self, seq: int):
        self.pg_indicator.fill((0, 0, 0))

        coef = self.effect_coef(2)
        
        size = self.pg_indicator.get_size()
        for star in self.background_stars:
            s = star['star']
            d = s['data']
            coords = self.switch_coords(star['coords'])
            coords = (
                int(coords[0] * size[0]) + s['center_offset'][0],
                int(coords[1] * size[1]) + s['center_offset'][1]
            )

            m = star['mult']
            mr = star['mult_range']
            ms = star['mult_speed']
            star['mult'] = min(max(m + ms, mr[0]), mr[1])
            star['mult_speed'] += 0.15 * (mr[1] - mr[0]) * (2 * random.random() - 1)
            #if not bs['background']:
            #    m *= coef
            if int(255 * m) < 1:
                continue

            surf = pygame.surfarray.make_surface(d * m)
            self.pg_indicator.blit(surf, coords, special_flags=pygame.BLEND_RGB_ADD)
        
        if self.final and self.constellation is not None:
            for star in self.constellation:
                s = star['star']
                d = s['data']
                coords = self.switch_coords(star['coords'])
                coords = (
                    int(coords[0] * size[0]) + s['center_offset'][0],
                    int(coords[1] * size[1]) + s['center_offset'][1]
                )

                if int(255 * coef) < 1:
                    continue

                surf = pygame.surfarray.make_surface(d * coef)
                self.pg_indicator.blit(surf, coords, special_flags=pygame.BLEND_RGB_ADD)

        self.pg_screen.blit(self.pg_indicator, self.indicator_offset_l)
        self.pg_screen.blit(self.pg_indicator, self.indicator_offset_r)
    
    def render(self, color: cv2.Mat, depth: cv2.Mat, face_bbox: Optional[BBox], seq: int) -> bool:
        """Receives an image containing the face, the bounding box of the face, and sequential number of the frame.
        Renders the final image.
        
        Returns True when user wants to terminate."""
        if color.shape[:2] != depth.shape[:2]:
            #print(color.shape, depth.shape)
            depth = cv2.resize(depth, color.shape[:2])
        if self.debug:
            color = cv2.resize(color, (self.image_size[1] // self.debug_divisor, self.image_size[0] // self.debug_divisor))
            depth = cv2.resize(depth, (self.image_size[1] // self.debug_divisor, self.image_size[0] // self.debug_divisor))
        
        if face_bbox is not None:
            self.is_face = True
            if self.final and self.final_trigger_time is None:
                self.final_trigger_time = time.time()
                if self.special:
                    self.constellation = random.choice(self.special_constellations)
                else:
                    self.constellation = random.choice(self.common_constellations)
            
            tl = face_bbox.top_left()
            br = face_bbox.bottom_right()
            self.bbox = BBox(tl[0], tl[1], br[0], br[1])
            if self.debug:
                self.bbox = BBox(self.bbox.x0 // self.debug_divisor, self.bbox.y0 // self.debug_divisor, self.bbox.x1 // self.debug_divisor, self.bbox.y1 // self.debug_divisor)
        else:
            self.is_face = False
            self.final_trigger_time = None
        
        self.render_face(color, depth, seq)
        self.render_indicator(seq)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.KEYUP and event.unicode == 'q':
                pygame.quit()
                return True
        return False

    def effect_coef(self, fade_in_time: float) -> float:
        if self.final_trigger_time is not None:
            t = time.time() - self.final_trigger_time
            return max(min(t - fade_in_time, 1), 0)
        return 0
    
    def switch_coords(self, c: Tuple[int, int]) -> Tuple[int, int]:
        if not self.debug:
            return c[1], c[0]
        return c

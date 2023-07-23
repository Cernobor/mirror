from dataclasses import dataclass
import json
import math
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
                 global_upscale: Tuple[int, int],
                 screen_rotated: bool,
                 depth: int,
                 halo_common_dir: str,
                 halo_special_dir: str,
                 halo_common_blow_factor: float,
                 halo_special_blow_factor: float,
                 halo_position_mixing_coef: float,
                 background_stars_no: int,
                 #common_constellations_js: str,
                 #special_constellations_js: str,
                 common_constellations_dir: str,
                 special_constellations_dir: str,
                 halo_fade_in_time: float,
                 constellation_fade_in_time: float,
                 halo_delay_time: float,
                 constellation_delay_time: float,
                 debug: bool=False) -> None:
        self.display_size = display_size
        self.image_size = image_size
        self.global_upscale = global_upscale
        self.screen_rotated = screen_rotated
        self.depth = depth
        self.halo_common_dir = halo_common_dir
        self.halo_special_dir = halo_special_dir
        self.halo_common_blow_factor = halo_common_blow_factor
        self.halo_special_blow_factor = halo_special_blow_factor
        self.halo_position_mixing_coef = halo_position_mixing_coef
        self.background_stars_no = background_stars_no
        #self.common_constellations_js = common_constellations_js
        #self.special_constellations_js = special_constellations_js
        self.common_constellations_dir = common_constellations_dir
        self.special_constellations_dir = special_constellations_dir
        self.halo_fade_in_time = halo_fade_in_time
        self.constellation_fade_in_time = constellation_fade_in_time
        self.halo_delay_time = halo_delay_time
        self.constellation_delay_time = constellation_delay_time
        self.debug = debug

        self._prepare()

    def _prepare(self):
        # general init
        pygame.init()
        self.is_face_prev = False
        self.is_face = False
        self.bbox = BBox(0, 0, 0, 0)
        self.final_trigger_time = None
        self.constellation = None
        self.halo = None
        self.halo_blow_factor = 1
        self.debug_divisor = 2
        self.special = False
        self.final_trigger_prev = False
        self.final_trigger = False
        self.final = False
        self.halo_factor = None
        self.halo_center = None
        
        # setup display sizes and orientations
        self.horizontal_idx = 1
        self.vertical_idx = 0
        if self.debug or self.screen_rotated:
            self.horizontal_idx = 0
            self.vertical_idx = 1

        full_ds = self.display_size
        canvas_size = utils.scale(full_ds, *reversed(self.global_upscale))
        cs = canvas_size
        self.vertical_diff = canvas_size[0] - self.image_size[0]
        self.horizontal_diff = canvas_size[1] - self.image_size[1]
        fs = pygame.FULLSCREEN
        if self.debug:
            cs = (canvas_size[1] // self.debug_divisor, canvas_size[0] // self.debug_divisor)
            full_ds = (full_ds[1] // self.debug_divisor, full_ds[0] // self.debug_divisor)
            self.display_size = full_ds
            self.image_size = (self.image_size[1] // self.debug_divisor, self.image_size[1] // self.debug_divisor)
            fs = 0

            size = (canvas_size[1] // self.debug_divisor, self.vertical_diff // self.debug_divisor)
            v = min(size) // 2
            h = max(size)
            indicator_size = (v, v)
            self.indicator_offset_l = (0, 0)
            self.indicator_offset_r = (h - v, 0)
        elif not self.screen_rotated:
            size = (self.vertical_diff, canvas_size[1])
            v = min(size) // 2
            h = max(size)
            indicator_size = (v, v)
            self.indicator_offset_l = (0, 0)
            self.indicator_offset_r = (0, h - v)
        else:
            cs = (canvas_size[1], canvas_size[0])
            size = (canvas_size[1], self.vertical_diff)
            v = min(size) // 2
            h = max(size)
            indicator_size = (v, v)
            self.indicator_offset_l = (0, 0)
            self.indicator_offset_r = (h - v, 0)
        self.pg_display = pygame.display.set_mode(size=self.display_size, flags=fs)
        self.pg_canvas = pygame.Surface(size=cs, flags=fs)
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
        
        # setup constellations
        #self.common_constellations = []
        #with open(self.common_constellations_js) as f:
        #    common_constellations = json.load(f)
        #for v in common_constellations.values():
        #    c = []
        #    for p in v:
        #        s = p['mag'] * 2 + 10
        #        star = {
        #            'coords': (p['x'], p['y']),
        #            'star': self.stars[s]
        #        }
        #        c.append(star)
        #    self.common_constellations.append(c)
        self.common_constellations_imgs = []
        if self.common_constellations_dir:
            files = os.listdir(self.common_constellations_dir)
            files.sort()
            self.common_constellations_imgs = [os.path.join(self.common_constellations_dir, f) for f in files]
        #self.special_constellations = []
        #with open(self.special_constellations_js) as f:
        #    special_constellations = json.load(f)
        #for v in special_constellations.values():
        #    c = []
        #    for p in v:
        #        s = p['mag'] * 2 + 10
        #        star = {
        #            'coords': (p['x'], p['y']),
        #            'star': self.stars[s]
        #        }
        #        c.append(star)
        #    self.special_constellations.append(c)
        self.special_constellations_imgs = []
        if self.special_constellations_dir:
            files = os.listdir(self.special_constellations_dir)
            files.sort()
            self.special_constellations_imgs = [os.path.join(self.special_constellations_dir, f) for f in files]
        
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
        if self.debug and self.is_face:
            cv2.rectangle(color, self.bbox.top_left(), self.bbox.bottom_right(), (255, 255, 255), 1)
        
        bg_mask = None
        if self.depth > 0:
            bg_mask = np.logical_or(depth > self.depth, depth < 350)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        cv2.flip(color, 1, color)
        if self.debug or self.screen_rotated:
            color = np.rot90(color)
        if bg_mask is not None:
            if self.debug or self.screen_rotated:
                bg_mask = np.transpose(bg_mask)
            else:
                bg_mask = np.flip(bg_mask, 1)

        
        pg_face = pygame.surfarray.make_surface(color)
        
        if self.final:
            coef = self.effect_coef(self.halo_delay_time, self.halo_fade_in_time)
        else:
            coef = 0
        if coef > 0 and self.halo:
            img = pygame.image.load(self.halo[seq % len(self.halo)]).convert_alpha()
            size = img.get_size()
            img_h = size[1]
            bbox_h = self.bbox.size()[self.vertical_idx]
            factor = bbox_h / img_h * self.halo_blow_factor
            if self.halo_factor is not None:
                self.halo_factor = utils.conv_comb(factor, self.halo_factor, 0.5)
            else:
                self.halo_factor = factor
            scaled = pygame.transform.smoothscale_by(img, self.halo_factor)
            
            if self.debug or self.screen_rotated:
                center = (self.bbox.x_bounds().center(), self.bbox.y_bounds().center())
            else:
                center = (self.bbox.x_bounds(flip=True, offset=pg_face.get_size()[0]).center(), self.bbox.y_bounds().center())
            if self.halo_center is not None:
                self.halo_center = (utils.conv_comb(center[0], self.halo_center[0], self.halo_position_mixing_coef),
                                    utils.conv_comb(center[1], self.halo_center[1], self.halo_position_mixing_coef))
            else:
                self.halo_center = center
            
            top_left = (self.halo_center[self.horizontal_idx] - scaled.get_width() // 2,
                        self.halo_center[self.vertical_idx] - scaled.get_height() // 2)
            scaled.set_alpha(255 * coef)
            pg_face.blit(scaled, top_left)
        else:
            self.halo_factor = None
            self.halo_center = None
        
        if bg_mask is not None:
            pg_face_fg = pygame.Surface(color.shape[:2], pygame.SRCALPHA, 32)
            pygame.pixelcopy.array_to_surface(pg_face_fg, color)
            alpha = np.array(pg_face_fg.get_view('A'), copy=False)
            alpha[bg_mask] = 0
            del alpha
            pg_face.blit(pg_face_fg, (0, 0))

        if self.debug:
            if pg_face.get_size() == self.image_size:
                self.pg_canvas.blit(pg_face, (0, self.vertical_diff // 2 // self.debug_divisor))
            else:
                self.pg_canvas.blit(pygame.transform.scale(pg_face, self.image_size), (0, self.vertical_diff // 2 // self.debug_divisor))
        elif not self.screen_rotated:
            if pg_face.get_size() == self.image_size:
                self.pg_canvas.blit(pg_face, (self.vertical_diff // 2, 0))
            else:
                self.pg_canvas.blit(pygame.transform.scale(pg_face, self.image_size), (self.vertical_diff // 2, 0))
        else:
            if pg_face.get_size() == self.image_size:
                self.pg_canvas.blit(pg_face, (0, self.vertical_diff // 2))
            else:
                self.pg_canvas.blit(pygame.transform.scale(pg_face, self.image_size), (0, self.vertical_diff // 2))
    
    def render_indicator(self, seq: int):
        self.pg_indicator.fill((0, 0, 0))

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
        
        if self.final:
            coef = self.effect_coef(self.constellation_delay_time, self.constellation_fade_in_time)
        else:
            coef = 0
        
        if coef > 0 and self.constellation is not None:
            #for star in self.constellation:
            #    s = star['star']
            #    d = s['data']
            #    coords = self.switch_coords(star['coords'])
            #    coords = (
            #        int(coords[0] * size[0]) + s['center_offset'][0],
            #        int(coords[1] * size[1]) + s['center_offset'][1]
            #    )
            #
            #    if int(255 * coef) < 1:
            #        continue
            #
            #    surf = pygame.surfarray.make_surface(d * coef)
            #    self.pg_indicator.blit(surf, coords, special_flags=pygame.BLEND_RGB_ADD)
            self.constellation.set_alpha(255 * coef)
            self.pg_indicator.blit(self.constellation, (0, 0))

        self.pg_canvas.blit(self.pg_indicator, self.indicator_offset_l)
        self.pg_canvas.blit(self.pg_indicator, self.indicator_offset_r)
    
    def render(self, color: cv2.Mat, depth: cv2.Mat, face_bbox: Optional[BBox], seq: int) -> bool:
        """Receives an image containing the face, the bounding box of the face, and sequential number of the frame.
        Renders the final image.
        
        Returns True when user wants to terminate."""
        if color.shape[:2] != depth.shape[:2]:
            #print(color.shape, depth.shape)
            depth = cv2.resize(depth, color.shape[:2])
        if self.debug:
            if face_bbox is not None:
                face_bbox = BBox(face_bbox.x0 * self.image_size[1] // color.shape[0],
                                 face_bbox.y0 * self.image_size[0] // color.shape[1],
                                 face_bbox.x1 * self.image_size[1] // color.shape[0],
                                 face_bbox.y1 * self.image_size[0] // color.shape[1])
            color = cv2.resize(color, (self.image_size[1], self.image_size[0]))
            depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]))
        
        if face_bbox is not None:
            self.is_face = True
            tl = face_bbox.top_left()
            br = face_bbox.bottom_right()
            self.bbox = BBox(tl[0], tl[1], br[0], br[1])
        else:
            self.is_face = False
        
        
        face_change = 0
        if not self.is_face and self.is_face_prev:
            face_change = -1
            print(f'-face final={self.final} ftt={self.final_trigger_time}')
        elif self.is_face and not self.is_face_prev:
            face_change = 1
            print(f'+face final={self.final} ftt={self.final_trigger_time}')
        final_change = 0
        if not self.final_trigger and self.final_trigger_prev:
            final_change = -1
            print(f'-final final={self.final} ftt={self.final_trigger_time}')
        elif self.final_trigger and not self.final_trigger_prev:
            final_change = 1
            print(f'+final final={self.final} ftt={self.final_trigger_time}')
        
        if self.final:
            if face_change == -1:
                self.final = False
                self.final_trigger_time = None
            elif face_change == 1:
                self.final_trigger_time = time.time()
                self.choose_effect()
        else:
            if final_change == 1:
                self.final = True
                if self.is_face:
                    self.final_trigger_time = time.time()
                    self.choose_effect()
        
        
        self.final_trigger_prev = self.final_trigger
        self.is_face_prev = self.is_face
        
        self.render_face(color, depth, seq)
        self.render_indicator(seq)
        if self.global_upscale[0] == self.global_upscale[1]:
            self.pg_display.blit(self.pg_canvas, (0, 0))
        else:
            pygame.transform.scale(self.pg_canvas, self.pg_display.get_size(), self.pg_display)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.KEYUP and event.unicode == 'q':
                pygame.quit()
                return True
        return False

    def choose_effect(self):
        print('Choosing effect...')
        halo = []
        if self.special:
            #choice = random.choice(self.special_constellations)
            constellation = random.choice(self.special_constellations_imgs)
            if self.halo_special_imgs:
                halo = self.halo_special_imgs
            self.halo_blow_factor = self.halo_special_blow_factor
        else:
            #choice = random.choice(self.common_constellations)
            constellation = random.choice(self.common_constellations_imgs)
            if not self.special and self.halo_common_imgs:
                halo = self.halo_common_imgs
            self.halo_blow_factor = self.halo_common_blow_factor
        print(f'Chosen constellation: {constellation}  Chosen halo: {halo}')
        
        #self.constellation = choice
        img = pygame.image.load(constellation).convert_alpha()
        img_size = img.get_size()
        surf_size = self.pg_indicator.get_size()
        factor = surf_size[0] / img_size[0], surf_size[1] / img_size[1]
        print(f'Indicator surf size: {surf_size}  Constellation img size: {img_size}  Scaling factor: {factor}')
        self.constellation = pygame.transform.smoothscale_by(img, factor)

        self.halo = halo
    
    def effect_coef(self, delay_time: float, fade_in_time: float) -> float:
        if self.final_trigger_time is not None:
            t = time.time() - self.final_trigger_time - delay_time
            if t < 0:
                return 0
            #coef = 1 / (1 + math.exp2(-(t - fade_in_time / 2) * 2 * 9 / fade_in_time))
            #coef = t / fade_in_time
            coef = (1 - math.cos(math.pi * t / fade_in_time)) / 2 if t < fade_in_time else 1
            return max(min(coef, 1), 0)
        return 0
    
    def switch_coords(self, c: Tuple[int, int]) -> Tuple[int, int]:
        if not self.debug:
            return c[1], c[0]
        return c

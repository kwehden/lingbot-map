"""Render overlays: CameraOverlay.

Overlays modify visible points and/or add extra geometry (lines, points, meshes)
to each rendered frame. The dict-based geometry format is consumed by Open3DRenderer.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import cv2
import numpy as np

from .camera import Camera
from .scene import Scene


# ---------------------------------------------------------------------------
# Built-in color ramps
# ---------------------------------------------------------------------------

def _ramp_cyan_blue(age: int, tail_len: int, total: int) -> np.ndarray:
    if age < tail_len:
        t = age / max(tail_len - 1, 1)
        r, g, b = 0.0, 1.0 - 0.5 * t, 1.0
    else:
        t = (age - tail_len) / max(total - tail_len - 1, 1)
        r, g, b = 0.0, 0.5 - 0.35 * t, 1.0 - 0.5 * t
    return np.array([r, g, b], dtype=np.float64).clip(0, 1)


def _ramp_white(age: int, tail_len: int, total: int) -> np.ndarray:
    t = age / max(total - 1, 1)
    v = 1.0 - 0.7 * t
    return np.array([v, v, v], dtype=np.float64)


def _ramp_rainbow(age: int, tail_len: int, total: int) -> np.ndarray:
    hue = (age / max(total - 1, 1)) * 0.8
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
    return np.array([r, g, b], dtype=np.float64)


def _make_solid_ramp(base_r, base_g, base_b):
    """Create a ramp that fades from a base color to darker."""
    def _ramp(age: int, tail_len: int, total: int) -> np.ndarray:
        t = age / max(total - 1, 1)
        fade = 1.0 - 0.7 * t
        return np.array([base_r * fade, base_g * fade, base_b * fade],
                        dtype=np.float64).clip(0, 1)
    return _ramp


_COLOR_RAMPS = {
    'cyan_blue': _ramp_cyan_blue,
    'white': _ramp_white,
    'rainbow': _ramp_rainbow,
    'red': _make_solid_ramp(1.0, 0.2, 0.1),
    'green': _make_solid_ramp(0.2, 1.0, 0.3),
    'yellow': _make_solid_ramp(1.0, 0.9, 0.1),
    'magenta': _make_solid_ramp(1.0, 0.2, 0.8),
}


# ---------------------------------------------------------------------------
# Color parsing
# ---------------------------------------------------------------------------

_COLOR_ALIASES = {
    'b': '#000000', 'black': '#000000', 'k': '#000000',
    'w': '#FFFFFF', 'white': '#FFFFFF',
    'r': '#FF0000', 'red': '#FF0000',
    'g': '#00FF00', 'green': '#00FF00',
    'gray': '#808080', 'grey': '#808080',
    'darkgray': '#0D0D0D',
    'cyan': '#00FFFF',
    'blue': '#0000FF',
    'yellow': '#FFFF00',
    'magenta': '#FF00FF',
    'orange': '#FFA500',
}


def parse_color(s: str) -> List[float]:
    """Parse color string → [r, g, b] float 0-1.

    Accepts: '#RGB', '#RRGGBB', english name, single letter.
    Returns None for empty string.
    """
    if not s:
        return None
    s = s.strip().lower()
    if s in _COLOR_ALIASES:
        s = _COLOR_ALIASES[s]
    if s.startswith('#'):
        h = s[1:]
        if len(h) == 3:
            h = h[0]*2 + h[1]*2 + h[2]*2
        if len(h) == 6:
            return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    raise ValueError(f"Cannot parse color: {s!r}. "
                     f"Use '#RRGGBB', '#RGB', or a name like 'black', 'white', 'r', 'g'.")


# ---------------------------------------------------------------------------
# Fixture category colors (ported from analytics/vocabulary.py)
#
# The analyze side lives in a separate repo whose Python module cannot be
# imported here, so the SHA-1-seeded HSL kernel is re-derived directly rather
# than duplicating a hand-picked palette. Same (category name, group) input
# yields the same bytes as vocabulary.py's class_color() — keep in lockstep.
# ---------------------------------------------------------------------------

_GROUP_HUE = {
    "none": 0.0,
    "lighting": 0.13,
    "mep": 0.52,
    "architectural": 0.62,
    "furniture": 0.08,
    "waste": 0.30,
    "safety": 0.0,
    "mining": 0.90,
    "farm": 0.25,
    "port": 0.58,
    "desert": 0.07,
    "vlm": 0.75,
    "other": 0.75,
}

_HUE_JITTER = 0.08
_SAT_RANGE = (0.65, 1.00)
_LIGHT_RANGE = (0.42, 0.70)
_UNLABELED_RGB = (128, 128, 128)


def _hash_unit(name: str, salt: str) -> float:
    """Deterministic value in [0, 1) from (salt, name) via SHA-1."""
    digest = hashlib.sha1((salt + name).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def category_color(name: str, group: str) -> tuple:
    """Deterministic RGB (0-1 floats) for a fixture category, matching
    analytics/vocabulary.py's class_color() up to the 0-255 rounding it does
    at the end (this returns unrounded 0-1 floats for the renderer's LineSet
    color arrays). name 'unlabeled' is always neutral gray."""
    if name == "unlabeled":
        r, g, b = _UNLABELED_RGB
        return (r / 255.0, g / 255.0, b / 255.0)
    base_hue = _GROUP_HUE.get(group, _GROUP_HUE["other"])
    hue = (base_hue + (_hash_unit(name, "hue") - 0.5) * _HUE_JITTER) % 1.0
    sat = _SAT_RANGE[0] + _hash_unit(name, "sat") * (_SAT_RANGE[1] - _SAT_RANGE[0])
    light = _LIGHT_RANGE[0] + _hash_unit(name, "light") * (_LIGHT_RANGE[1] - _LIGHT_RANGE[0])
    return colorsys.hls_to_rgb(hue, light, sat)


# ---------------------------------------------------------------------------
# Style dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrailStyle:
    """Trajectory trail line style."""
    enabled: bool = True
    tail_len: int = 50
    line_width: float = 2.0
    color_ramp: str = 'cyan_blue'

    def get_color_fn(self) -> Callable:
        return _COLOR_RAMPS.get(self.color_ramp, _ramp_cyan_blue)


@dataclass
class HeadStyle:
    """Camera head visualization style."""
    mode: str = 'points'        # 'points' | 'frustum' | 'frustum_textured'
    num_frames: int = 8
    point_size: float = 8.0
    frustum_scale: float = 0.05
    frustum_line_width: float = 1.5
    frustum_color: Optional[List[float]] = None
    texture_alpha: float = 0.8


# ---------------------------------------------------------------------------
# Overlay base
# ---------------------------------------------------------------------------

class Overlay:
    """Base class for render overlays."""

    def apply(self, scene: Scene, camera: Camera, frame_idx: int,
              vis_xyz: np.ndarray, vis_rgb: np.ndarray,
              vis_frames: Optional[np.ndarray]) -> tuple:
        """Returns (vis_xyz, vis_rgb, extra_geoms)."""
        return vis_xyz, vis_rgb, []


# ---------------------------------------------------------------------------
# CameraOverlay
# ---------------------------------------------------------------------------

class CameraOverlay(Overlay):
    """Configurable camera visualization: trail line + head display."""

    def __init__(self, c2w_poses: np.ndarray,
                 intrinsics: Optional[np.ndarray] = None,
                 images: Optional[np.ndarray] = None,
                 trail: Optional[TrailStyle] = None,
                 head: Optional[HeadStyle] = None):
        self.c2w_poses = c2w_poses
        self.intrinsics = intrinsics
        self.images = images
        self.trail = trail if trail is not None else TrailStyle()
        self.head = head if head is not ... else HeadStyle()
        if head is None:
            self.head = None

    def apply(self, scene, camera, frame_idx, vis_xyz, vis_rgb, vis_frames):
        cam_positions = self.c2w_poses[:frame_idx + 1, :3, 3]
        K = len(cam_positions)
        geoms = []

        if self.trail and self.trail.enabled and K >= 2:
            geoms.extend(self._build_trail(cam_positions))

        if self.head and K > 0:
            if self.head.mode == 'points':
                geoms.extend(self._build_head_points(cam_positions))
            elif self.head.mode in ('frustum', 'frustum_textured'):
                geoms.extend(self._build_head_frustums(frame_idx))

        return vis_xyz, vis_rgb, geoms

    def _build_trail(self, cam_positions: np.ndarray) -> List[dict]:
        pts = cam_positions.astype(np.float64)
        K = len(pts)
        segs = np.array([[i, i + 1] for i in range(K - 1)], dtype=np.int32)
        total = K - 1
        color_fn = self.trail.get_color_fn()
        colors = np.array([color_fn(total - 1 - si, self.trail.tail_len, total)
                           for si in range(total)])
        return [{
            'type': 'lines', 'name': 'traj_lines',
            'points': pts, 'segments': segs, 'colors': colors,
            'line_width': self.trail.line_width,
        }]

    def _build_head_points(self, cam_positions: np.ndarray) -> List[dict]:
        K = len(cam_positions)
        head_pts = cam_positions[max(0, K - self.head.num_frames):].astype(np.float64)
        nh = len(head_pts)
        if nh == 0:
            return []
        head_colors = np.zeros((nh, 3), dtype=np.float64)
        for j in range(nh):
            t = (nh - 1 - j) / max(nh - 1, 1)
            head_colors[j] = [1.0 - 0.2 * t, 1.0, 1.0]
        return [{
            'type': 'points', 'name': 'traj_head',
            'points': head_pts, 'colors': head_colors,
            'point_size': self.head.point_size,
        }]

    def _build_head_frustums(self, frame_idx: int) -> List[dict]:
        start = max(0, frame_idx + 1 - self.head.num_frames)
        end = frame_idx + 1
        geoms = []

        for fi in range(start, end):
            c2w = self.c2w_poses[fi]
            scale = self.head.frustum_scale
            if self.intrinsics is not None:
                K_mat = self.intrinsics[fi]
                fx, fy = K_mat[0, 0], K_mat[1, 1]
                cx, cy = K_mat[0, 2], K_mat[1, 2]
                w, h = cx * 2, cy * 2
                corners_cam = np.array([
                    [(-cx) / fx, (-cy) / fy, 1.0],
                    [(w - cx) / fx, (-cy) / fy, 1.0],
                    [(w - cx) / fx, (h - cy) / fy, 1.0],
                    [(-cx) / fx, (h - cy) / fy, 1.0],
                ], dtype=np.float64) * scale
            else:
                corners_cam = np.array([
                    [-1, -0.75, 1],
                    [ 1, -0.75, 1],
                    [ 1,  0.75, 1],
                    [-1,  0.75, 1],
                ], dtype=np.float64) * scale

            R = c2w[:3, :3].astype(np.float64)
            t = c2w[:3, 3].astype(np.float64)
            origin = t
            corners_world = (corners_cam @ R.T) + t

            pts = np.vstack([origin[None, :], corners_world])
            lines = np.array([
                [0, 1], [0, 2], [0, 3], [0, 4],
                [1, 2], [2, 3], [3, 4], [4, 1],
            ], dtype=np.int32)

            age = frame_idx - fi
            brightness = 1.0 - 0.6 * (age / max(self.head.num_frames - 1, 1))
            if self.head.frustum_color:
                c = np.array(self.head.frustum_color, dtype=np.float64) * brightness
            else:
                # Follow trail ramp newest color (age=0)
                color_fn = self.trail.get_color_fn()
                base = color_fn(0, self.trail.tail_len, max(frame_idx, 1))
                c = base * brightness
            line_colors = np.tile(c, (len(lines), 1))

            geoms.append({
                'type': 'lines',
                'name': f'frustum_{fi}',
                'points': pts,
                'segments': lines,
                'colors': line_colors,
                'line_width': self.head.frustum_line_width,
            })

            if self.head.mode == 'frustum_textured' and self.images is not None:
                geoms.append({
                    'type': 'textured_quad',
                    'name': f'frustum_tex_{fi}',
                    'corners': corners_world,
                    'image': self.images[fi],
                    'alpha': self.head.texture_alpha * brightness,
                })

        return geoms

    @classmethod
    def from_preset(cls, preset: str, c2w_poses: np.ndarray,
                    intrinsics: Optional[np.ndarray] = None,
                    images: Optional[np.ndarray] = None,
                    trail: Optional[TrailStyle] = None,
                    head: Optional[HeadStyle] = None) -> 'CameraOverlay':
        """Create from a named preset, with optional style overrides."""
        base_trail, base_head = preset_styles(preset)
        final_trail = trail if trail is not None else base_trail
        # head=None means "no head" only if preset also has no head;
        # if caller passes head explicitly, use it
        if head is not None:
            final_head = head
        else:
            final_head = base_head
        return cls(c2w_poses, intrinsics=intrinsics, images=images,
                   trail=final_trail, head=final_head)


# ---------------------------------------------------------------------------
# BoxOverlay
# ---------------------------------------------------------------------------

# Unit-cube corners (min=0, max=1 per axis) and the 12 edges connecting them.
# Same 8-point/edge-list topology as CameraOverlay._build_head_frustums(),
# generalized from a 5-point pyramid to an 8-point axis-aligned box.
_BOX_UNIT_CORNERS = np.array([
    [0, 0, 0],   # 0
    [1, 0, 0],   # 1
    [1, 1, 0],   # 2
    [0, 1, 0],   # 3
    [0, 0, 1],   # 4
    [1, 0, 1],   # 5
    [1, 1, 1],   # 6
    [0, 1, 1],   # 7
], dtype=np.float64)

_BOX_EDGES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],   # bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],   # top face
    [0, 4], [1, 5], [2, 6], [3, 7],   # vertical edges
], dtype=np.int32)


class FrameIndexMapper:
    """Converts a render-job frame index into the analyze job's frame-index
    space via timestamp equivalence, so visibility windows computed by the
    analyze job apply correctly even when the two jobs sampled the source
    video at different rates (REQ-015).

        timestamp_seconds     = render_frame_idx * render_frame_interval / render_src_fps
        analyze_frame_idx_equiv = round(timestamp_seconds * analyze_src_fps / analyze_frame_interval)

    When the two jobs' sampling parameters match (today's common case) this
    reduces algebraically to ``analyze_frame_idx_equiv == render_frame_idx``,
    i.e. an exact no-op — but it stops silently trusting index equality the
    moment either side's ``src_fps``/``frame_interval`` diverges.
    """

    def __init__(self, render_src_fps: float, render_frame_interval: float,
                 analyze_src_fps: float, analyze_frame_interval: float):
        self.render_src_fps = float(render_src_fps)
        self.render_frame_interval = float(render_frame_interval)
        self.analyze_src_fps = float(analyze_src_fps)
        self.analyze_frame_interval = float(analyze_frame_interval)

    def map(self, render_frame_idx: int) -> int:
        """Return the analyze-frame-space index equivalent to this render
        frame. Always an int: the formula is well-defined for any
        ``render_frame_idx >= 0`` and any analyze index that lands outside a
        fixture's ``[first_frame, last_frame]`` window is simply filtered out
        by ``BoxOverlay.visible_fixtures`` — so there is no separate
        out-of-range sentinel to return here."""
        timestamp_seconds = (render_frame_idx * self.render_frame_interval
                             / self.render_src_fps)
        return round(timestamp_seconds * self.analyze_src_fps
                     / self.analyze_frame_interval)


class BoxOverlay(Overlay):
    """Draws fixture 3D bounding boxes as batched LineSet edges, filtered by
    each fixture's frame-visibility window (remapped into this render job's
    own frame-index space via the frame_mapper)."""

    def __init__(self, fixtures: List[dict], frame_mapper,
                 line_width: float = 2.0, category_colors: Optional[dict] = None,
                 box_depth_test: bool = False):
        """
        Args:
            fixtures: parsed DC-2 JSON ``fixtures`` list — each dict has
                bbox_min (3), bbox_max (3), category, group, first_frame,
                last_frame, frames_seen (optional).
            frame_mapper: maps this render job's frame_idx -> the analyze
                job's frame-index space via ``.map(frame_idx) -> int | None``
                (None when the render frame has no analyze-space equivalent).
                ``None`` here means identity mapping (render idx == analyze
                idx), the v1 same-sampling case.
            line_width: LineSet edge width.
            category_colors: optional {category: (r, g, b) 0-1} override;
                defaults to analytics/vocabulary.json's color logic
                (re-derived at construction time — see ``category_color``).
            box_depth_test: when True, ``apply()`` emits NO 3D LineSet — box
                edges are instead drawn by the CPU-side 2D depth-tested
                compositor (``composite_box_edges``) at the label-stamping
                call site, because the occlusion spike (REQ-017/TASK-031)
                found unlitLine LineSets do not depth-test here. Emitting the
                LineSet too would double-draw the edges (always-on-top 3D
                lines plus the depth-tested 2D lines).
        """
        self.frame_mapper = frame_mapper
        self.line_width = line_width
        self.box_depth_test = box_depth_test
        self.fixtures = self._validate_fixtures(fixtures)
        # OBS-2 per-run tally: fixtures that survived validation (rendered
        # whenever their window is active) vs. dropped as malformed.
        self.n_rendered = len(self.fixtures)
        self.n_skipped = len(fixtures) - len(self.fixtures)

        if category_colors is None:
            category_colors = {}
            for f in self.fixtures:
                cat = f['category']
                if cat not in category_colors:
                    category_colors[cat] = category_color(cat, f.get('group', 'other'))
        self.category_colors = category_colors

    @staticmethod
    def _validate_fixtures(fixtures: List[dict]) -> List[dict]:
        """Drop fixtures missing required geometry/label/window fields, warning
        on each skip (feeds TASK-017's rendered/skipped tally). Fixtures with
        only ``frames_seen`` get first_frame/last_frame derived from it."""
        cleaned = []
        for f in fixtures:
            fid = f.get('id', '?')
            missing = next((field_name for field_name in ('bbox_min', 'bbox_max', 'category')
                            if f.get(field_name) is None), None)
            if missing is not None:
                print(f"[warn] Skipping fixture id={fid}: missing required field {missing}")
                continue

            first = f.get('first_frame')
            last = f.get('last_frame')
            if first is None or last is None:
                seen = f.get('frames_seen')
                if not seen:
                    print(f"[warn] Skipping fixture id={fid}: missing required field "
                          f"first_frame/last_frame or frames_seen")
                    continue
                first, last = min(seen), max(seen)

            if first > last:
                print(f"[warn] Skipping fixture id={fid}: malformed window "
                      f"first_frame={first} > last_frame={last}")
                continue

            entry = dict(f)
            entry['first_frame'] = first
            entry['last_frame'] = last
            cleaned.append(entry)
        return cleaned

    def visible_fixtures(self, frame_idx: int) -> List[dict]:
        """Fixtures whose [first_frame, last_frame] window covers this render
        frame (after remapping into analyze-frame space). Also consumed by the
        label-stamping step so box and label visibility can never disagree."""
        if self.frame_mapper is None:
            analyze_idx = frame_idx
        else:
            analyze_idx = self.frame_mapper.map(frame_idx)
            if analyze_idx is None:
                return []
        return [f for f in self.fixtures
                if f['first_frame'] <= analyze_idx <= f['last_frame']]

    def apply(self, scene, camera, frame_idx, vis_xyz, vis_rgb, vis_frames):
        # Under the 2D depth-test fallback the edges are composited on the CPU
        # after render (see composite_box_edges); emit no 3D LineSet here so
        # the two paths never double-draw the same edges.
        if self.box_depth_test:
            return vis_xyz, vis_rgb, []
        visible = self.visible_fixtures(frame_idx)
        if not visible:
            return vis_xyz, vis_rgb, []
        return vis_xyz, vis_rgb, [self._build_boxes(visible)]

    def _build_boxes(self, visible: List[dict]) -> dict:
        n = len(visible)
        points = np.empty((8 * n, 3), dtype=np.float64)
        segments = np.empty((12 * n, 2), dtype=np.int32)
        colors = np.empty((12 * n, 3), dtype=np.float64)

        for k, f in enumerate(visible):
            bmin = np.asarray(f['bbox_min'], dtype=np.float64)
            bmax = np.asarray(f['bbox_max'], dtype=np.float64)
            points[8 * k:8 * k + 8] = bmin + _BOX_UNIT_CORNERS * (bmax - bmin)
            segments[12 * k:12 * k + 12] = _BOX_EDGES + 8 * k
            cat = f['category']
            color = self.category_colors.get(cat)
            if color is None:
                color = category_color(cat, f.get('group', 'other'))
            colors[12 * k:12 * k + 12] = color

        return {
            'type': 'lines',
            'name': 'fixture_boxes',
            'points': points,
            'segments': segments,
            'colors': colors,
            'line_width': self.line_width,
        }


def build_overlays(cfg, scene):
    """Build overlay instances + serializable specs from config.

    Reads ``cfg.overlay`` and the rendered scene, returning ``(overlays, specs)``
    suitable for ``OfflinePipeline``.  When ``cfg.overlay.camera_vis`` is empty,
    returns ``([], [])``.
    """
    ov = cfg.overlay
    overlays, specs = [], []

    if ov.camera_vis:
        base_trail, base_head = preset_styles(ov.camera_vis)

        base_trail.enabled = ov.trail_enabled
        base_trail.tail_len = ov.trail_tail_len
        base_trail.line_width = ov.trail_line_width
        base_trail.color_ramp = ov.trail_color_ramp

        if base_head is not None:
            base_head.num_frames = ov.head_num_frames
            base_head.point_size = ov.head_point_size
            base_head.frustum_scale = ov.head_frustum_scale
            base_head.frustum_line_width = ov.head_frustum_line_width
            base_head.frustum_color = parse_color(ov.head_frustum_color)
            base_head.texture_alpha = ov.head_texture_alpha

        thumb = scene.thumbnail_images() if ov.camera_vis == 'textured' else None
        overlays.append(CameraOverlay(
            scene.c2w_poses, intrinsics=scene.intrinsics,
            images=thumb, trail=base_trail, head=base_head))
        specs.append({'type': 'camera', 'preset': ov.camera_vis,
                      'overlay_config': {
                          'trail_color_ramp': ov.trail_color_ramp,
                          'trail_tail_len': ov.trail_tail_len,
                          'trail_line_width': ov.trail_line_width,
                          'trail_enabled': ov.trail_enabled,
                          'head_num_frames': ov.head_num_frames,
                          'head_point_size': ov.head_point_size,
                          'head_frustum_scale': ov.head_frustum_scale,
                          'head_frustum_line_width': ov.head_frustum_line_width,
                          'head_frustum_color': ov.head_frustum_color,
                          'head_texture_alpha': ov.head_texture_alpha,
                      }})

    boxes_json = getattr(ov, 'boxes_json', None)
    show_labels = getattr(ov, 'show_labels', False)
    if boxes_json:
        try:
            with open(boxes_json) as f:
                doc = json.load(f)
        except Exception as e:
            print(f"[warn] --boxes_json {boxes_json} could not be read ({e}); "
                  f"rendering without box overlay")
        else:
            fixtures = doc.get('fixtures', [])
            frame_mapper = _build_frame_mapper(cfg, doc.get('analyze_frame_sampling'))
            # REQ-018 fallback: the occlusion spike (TASK-031) FAILED for this
            # deployment, so box edges are always drawn via the CPU-side 2D
            # depth-test path instead of a 3D LineSet. Turning it on here (a)
            # tells the renderer to capture the point-cloud-only depth buffer
            # and (b) tells BoxOverlay.apply() to emit no LineSet. cfg is the
            # same object the pipeline receives, and box_depth_test round-trips
            # to parallel workers via RenderConfig.to_dict()/from_dict().
            cfg.render.box_depth_test = True
            box_overlay = BoxOverlay(fixtures, frame_mapper,
                                     line_width=ov.box_line_width,
                                     box_depth_test=True)
            overlays.append(box_overlay)
            specs.append({'type': 'box',
                          'fixtures': box_overlay.fixtures,
                          'line_width': ov.box_line_width,
                          'category_colors': box_overlay.category_colors,
                          'box_depth_test': True,
                          'frame_mapper': None if frame_mapper is None else {
                              'render_src_fps': frame_mapper.render_src_fps,
                              'render_frame_interval': frame_mapper.render_frame_interval,
                              'analyze_src_fps': frame_mapper.analyze_src_fps,
                              'analyze_frame_interval': frame_mapper.analyze_frame_interval,
                          }})
    elif show_labels:
        print("[warn] --show_labels has no effect without --boxes_json; "
              "no box overlay to label")

    return overlays, specs


def _build_frame_mapper(cfg, analyze_sampling):
    """Build a FrameIndexMapper from the render job's observed sampling
    (``cfg.observed_src_fps``/``cfg.observed_frame_interval``) and the DC-2
    artifact's ``analyze_frame_sampling`` block. Returns ``None`` (identity
    mapping) when any of the four values is missing — the safe v1 fallback
    when either side didn't record real sampling metadata (e.g. image-folder
    input, or a legacy analyze run with sentinel values)."""
    if not analyze_sampling:
        return None
    render_src_fps = getattr(cfg, 'observed_src_fps', None)
    render_interval = getattr(cfg, 'observed_frame_interval', None)
    analyze_src_fps = analyze_sampling.get('src_fps')
    analyze_interval = analyze_sampling.get('frame_interval')
    values = (render_src_fps, render_interval, analyze_src_fps, analyze_interval)
    if any(v is None for v in values) or render_src_fps == 0 or analyze_interval == 0:
        return None
    return FrameIndexMapper(render_src_fps, render_interval,
                            analyze_src_fps, analyze_interval)


def preset_styles(preset: str):
    """Return (TrailStyle, HeadStyle or None) for a named preset."""
    if preset == 'default':
        return TrailStyle(), HeadStyle(mode='points', num_frames=8)
    elif preset == 'frustum':
        return TrailStyle(), HeadStyle(mode='frustum', num_frames=3,
                                       frustum_scale=0.05)
    elif preset == 'textured':
        return TrailStyle(), HeadStyle(mode='frustum_textured', num_frames=3,
                                       frustum_scale=0.05)
    elif preset == 'trail':
        return TrailStyle(), None
    else:
        raise ValueError(f"Unknown camera_vis preset: {preset!r}. "
                         f"Choose from: default, frustum, textured, trail")


# ---------------------------------------------------------------------------
# 2D projection (post-processing, not a 3D overlay)
# ---------------------------------------------------------------------------

def _project_core(point_world: np.ndarray, w2c: np.ndarray,
                  fx: float, fy: float, cx: float, cy: float):
    """Project one world point through a precomputed ``w2c`` + pinhole
    intrinsics. Returns ``(px, py, z_cam)`` where ``z_cam`` is the camera-space
    (view) depth before the perspective divide, or ``None`` if the point is at/
    behind the camera plane (``z_cam <= 1e-6``). Does NOT apply frame-bounds
    clipping — callers decide how to treat off-frame pixels. Shared by
    ``project_point_to_screen()`` (adds bounds check) and the box-edge depth
    compositor (needs ``z_cam`` for the depth test)."""
    p_cam = w2c @ np.append(point_world, 1.0)
    z = p_cam[2]
    if z <= 1e-6:
        return None
    return float(p_cam[0] * fx / z + cx), float(p_cam[1] * fy / z + cy), float(z)


def project_point_to_screen(point_world: np.ndarray, camera: Camera,
                            render_w: int, render_h: int):
    """Pinhole-project one 3D world point to 2D screen pixels, using the same
    camera model the renderer's own unprojection/culling uses
    (Camera.w2c() + Camera.fov_intrinsics()).

    Returns (px, py) if the point is in front of the camera (z > 0 in camera
    space) and within [0, render_w) x [0, render_h) — else None (off-frame or
    behind camera; never clamped to an edge).
    """
    w2c = camera.w2c()
    fx, fy, cx, cy = camera.fov_intrinsics(render_w, render_h)
    r = _project_core(point_world, w2c, fx, fy, cx, cy)
    if r is None:
        return None
    px, py, _ = r
    if not (0 <= px < render_w and 0 <= py < render_h):
        return None
    return px, py


def stamp_fixture_labels(image: np.ndarray, camera: Camera,
                         visible_fixtures: List[dict],
                         font_scale: float = 0.5) -> np.ndarray:
    """For each visible fixture, project its centroid (mean of bbox_min/
    bbox_max) to screen space; if the projection succeeds (on-frame, in front
    of camera), draw its ``category`` text via cv2.putText at that position
    (same outline+fill style as stamp_frame_tag). If projection fails, skip
    that fixture's label entirely for this frame (REQ-016b) — never clamp to
    an edge position. Modifies ``image`` in place, returns it."""
    render_h, render_w = image.shape[:2]
    for fx in visible_fixtures:
        centroid = (np.asarray(fx['bbox_min'], dtype=np.float64)
                    + np.asarray(fx['bbox_max'], dtype=np.float64)) / 2.0
        pos = project_point_to_screen(centroid, camera, render_w, render_h)
        if pos is None:
            continue
        px, py = int(pos[0]), int(pos[1])
        text = fx['category']
        cv2.putText(image, text, (px, py), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 3, cv2.LINE_AA)       # outline
        cv2.putText(image, text, (px, py), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)  # fill
    return image


# Depth tolerance (metres, camera-space) for the box-edge depth test. An edge
# sample draws when its view-space depth is nearer than the point cloud at that
# pixel, plus this slack — so a box surface that sits flush against the cloud
# geometry it bounds still draws instead of z-fighting into invisibility.
# Sized to the near-plane scale already used by RenderConfig.near (0.1 m).
_BOX_DEPTH_EPSILON = 0.05


def composite_box_edges(image: np.ndarray, camera: Camera,
                        visible_fixtures: List[dict], cloud_depth: np.ndarray,
                        category_colors: Optional[dict] = None,
                        line_width: float = 2.0,
                        epsilon: float = _BOX_DEPTH_EPSILON) -> np.ndarray:
    """CPU-side 2D depth-tested box-edge compositor — the REQ-018 fallback for
    when Open3D's ``unlitLine`` LineSet does not depth-test against the point
    cloud (occlusion spike / TASK-031 FAIL).

    For each visible fixture's 12 box edges, samples points along the 3D edge,
    projects each to screen space, and draws short ``cv2.line`` segments only
    between consecutive samples that are (a) in front of the camera, (b)
    on-frame, and (c) nearer than the point cloud at that pixel
    (``edge_depth < cloud_depth + epsilon``). Where the cloud has no valid
    depth (background), the edge always draws — nothing occludes it there.

    ``cloud_depth`` is the point-cloud-only view-space depth buffer captured by
    the renderer (``render_to_depth_image(z_in_view_space=True)``), the same
    per-pixel camera-space Z that ``_project_core`` returns for each sample, so
    the comparison is apples-to-apples. Rather than interpolating the two
    endpoints' depths, each sample's exact projected Z is used (equivalent in
    the limit, and correct across perspective foreshortening). Off-frame /
    behind-camera samples are skipped, never clamped (REQ-016b principle).
    Modifies ``image`` in place, returns it."""
    if cloud_depth is None or not visible_fixtures:
        return image
    render_h, render_w = image.shape[:2]
    w2c = camera.w2c()
    fx, fy, cx, cy = camera.fov_intrinsics(render_w, render_h)
    thickness = max(1, int(round(line_width)))
    cloud_valid = np.isfinite(cloud_depth) & (cloud_depth > 0)

    def _sample_visible(pt_world):
        """Project one world point; return ((ix, iy), passes_depth) or None if
        behind camera or off-frame."""
        r = _project_core(pt_world, w2c, fx, fy, cx, cy)
        if r is None:
            return None
        px, py, z = r
        ix, iy = int(px), int(py)
        if not (0 <= ix < render_w and 0 <= iy < render_h):
            return None
        # Background (no cloud here) never occludes; else nearer-than-cloud wins.
        passes = (not cloud_valid[iy, ix]) or (z < cloud_depth[iy, ix] + epsilon)
        return (ix, iy), passes

    for f in visible_fixtures:
        bmin = np.asarray(f['bbox_min'], dtype=np.float64)
        bmax = np.asarray(f['bbox_max'], dtype=np.float64)
        corners = bmin + _BOX_UNIT_CORNERS * (bmax - bmin)
        cat = f['category']
        color = (category_colors or {}).get(cat)
        if color is None:
            color = category_color(cat, f.get('group', 'other'))
        # image is RGB at this stage (BGR conversion happens at imwrite), so
        # pass color in R,G,B order — matching the 3D LineSet's RGB colors.
        rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

        for a, b in _BOX_EDGES:
            p0, p1 = corners[a], corners[b]
            # Sample density from the edge's on-screen length (~4 px spacing),
            # capped so a long near-camera edge can't explode the sample count.
            e0 = _project_core(p0, w2c, fx, fy, cx, cy)
            e1 = _project_core(p1, w2c, fx, fy, cx, cy)
            if e0 is not None and e1 is not None:
                pix_len = np.hypot(e0[0] - e1[0], e0[1] - e1[1])
                n = int(min(64, max(2, pix_len / 4.0 + 2)))
            else:
                n = 64  # an endpoint crosses the camera plane; sample densely
            prev = None
            for s in range(n + 1):
                t = s / n
                cur = _sample_visible(p0 + (p1 - p0) * t)
                if prev is not None and cur is not None and prev[1] and cur[1]:
                    cv2.line(image, prev[0], cur[0], rgb, thickness, cv2.LINE_AA)
                prev = cur
    return image


# ---------------------------------------------------------------------------
# 2D frame tag (post-processing, not a 3D overlay)
# ---------------------------------------------------------------------------

def stamp_frame_tag(image: np.ndarray, frame_idx: int, total_frames: int,
                    position: str = 'top_left') -> np.ndarray:
    """Draw frame counter text on rendered image. Modifies in-place."""
    text = f"{frame_idx + 1} / {total_frames} Frames"
    h, w = image.shape[:2]
    scale = max(0.4, w / 1920 * 0.8)
    thickness = max(1, int(w / 1920 * 2))
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    margin = int(w * 0.02)

    positions = {
        'top_left':     (margin, margin + th),
        'top_right':    (w - tw - margin, margin + th),
        'bottom_left':  (margin, h - margin),
        'bottom_right': (w - tw - margin, h - margin),
    }
    org = positions.get(position, positions['top_left'])

    # Black outline for readability, then white text
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image

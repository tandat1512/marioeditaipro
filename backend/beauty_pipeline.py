from __future__ import annotations

import base64
import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    from .models import BeautyConfig, FaceMeta, MouthValues, PointModel
except ImportError:
    # Fallback for running directly from backend directory
    from models import BeautyConfig, FaceMeta, MouthValues, PointModel

mp_face_mesh = mp.solutions.face_mesh
_FACE_LOCK = threading.Lock()

MAX_WARP_SHIFT_PX = 20.0
MAX_SMILE_SHIFT_PX = 18.0
LIP_ROI_PADDING = 12


def _indices_from_connections(connections: Sequence[Tuple[int, int]]) -> List[int]:
    unique = set()
    for a, b in connections:
        unique.add(a)
        unique.add(b)
    return sorted(unique)


FACE_OVAL_INDICES = _indices_from_connections(mp_face_mesh.FACEMESH_FACE_OVAL)
LEFT_BROW_INDICES = _indices_from_connections(mp_face_mesh.FACEMESH_LEFT_EYEBROW)
RIGHT_BROW_INDICES = _indices_from_connections(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)
FOREHEAD_BASE_INDICES = sorted(set(LEFT_BROW_INDICES + RIGHT_BROW_INDICES))


@dataclass
class FaceAnalysis:
    bbox: Tuple[int, int, int, int]
    landmarks: np.ndarray  # shape: (468, 2)
    mask: np.ndarray  # uint8 mask 0-255


@dataclass
class IrisRegion:
    mask: np.ndarray
    center: Tuple[float, float]
    radius: float
    pupil_mask: np.ndarray


def _iris_ring_mask(region: IrisRegion) -> np.ndarray:
    if region.radius <= 0 or not np.any(region.mask):
        return np.zeros_like(region.mask)
    ring = cv2.subtract(region.mask, region.pupil_mask)
    sigma = max(1, int(region.radius * 0.35))
    return cv2.GaussianBlur(ring, (0, 0), sigmaX=sigma)


def _iris_donut_mask(
    region: IrisRegion,
    buffer_px: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    if region.radius <= 0:
        return (
            np.zeros_like(region.mask),
            np.zeros_like(region.mask),
        )
    h, w = region.mask.shape
    donut = np.zeros((h, w), dtype=np.uint8)
    pupil_guard = np.zeros((h, w), dtype=np.uint8)
    center_int = (int(round(region.center[0])), int(round(region.center[1])))
    iris_r = max(1, int(round(region.radius)))
    cv2.circle(donut, center_int, iris_r, 255, -1, lineType=cv2.LINE_AA)
    pupil_radius = max(1, int(round(region.radius * 0.38)))
    guard_radius = pupil_radius + max(1, buffer_px)
    cv2.circle(pupil_guard, center_int, guard_radius, 255, -1, lineType=cv2.LINE_AA)
    donut = cv2.subtract(donut, pupil_guard)
    donut = cv2.GaussianBlur(donut, (0, 0), sigmaX=max(1, int(region.radius * 0.3)))
    pupil_guard = cv2.GaussianBlur(
        pupil_guard, (0, 0), sigmaX=max(1, int(guard_radius * 0.5))
    )
    return donut, pupil_guard


class FaceAnalyzer:
    """MediaPipe FaceMesh wrapper with thread safety."""

    def __init__(self) -> None:
        self._mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, image: np.ndarray) -> Optional[FaceAnalysis]:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with _FACE_LOCK:
            results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0].landmark
        pts = np.array(
            [[lm.x * w, lm.y * h] for lm in face_landmarks],
            dtype=np.float32,
        )

        hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
        mask = _build_full_face_mask(pts, (h, w))
        if not np.any(mask):
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=w * 0.01)

        x, y, bw, bh = cv2.boundingRect(hull)
        bbox = (int(x), int(y), int(bw), int(bh))
        return FaceAnalysis(bbox=bbox, landmarks=pts, mask=mask)

    def serialize(self, analysis: Optional[FaceAnalysis]) -> Optional[FaceMeta]:
        if not analysis:
            return None
        bbox = [int(v) for v in analysis.bbox]
        landmarks = [
            {"x": float(pt[0]), "y": float(pt[1])} for pt in analysis.landmarks
        ]
        return FaceMeta(bbox=bbox, confidence=1.0, landmarks=landmarks)


def _blend(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    alpha = np.clip(alpha, 0.0, 1.0)
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    return cv2.convertScaleAbs(
        base.astype(np.float32) * (1 - mask_3d * alpha)
        + overlay.astype(np.float32) * (mask_3d * alpha)
    )


def _ease_strength(value: float, power: float = 1.15) -> float:
    if value == 0:
        return 0.0
    strength = min(1.0, abs(value) / 100.0) ** power
    return strength if value > 0 else -strength


def _alpha_from_strength(base: float, span: float, strength: float) -> float:
    return np.clip(base + span * min(1.0, abs(strength)), 0.0, 1.0)


FACE_SCULPT_ATTENUATION = {
    "slim": 0.45,
    "vline": 0.5,
    "chin": 0.5,
    "forehead": 0.5,
    "jaw": 0.48,
    "nose_slim": 0.45,
    "nose_bridge": 0.48,
}
DEFAULT_SCULPT_ATTENUATION = 0.5


def _feature_strength(value: float, feature: str, power: float) -> float:
    if value == 0:
        return 0.0
    scale = FACE_SCULPT_ATTENUATION.get(feature, DEFAULT_SCULPT_ATTENUATION)
    return _ease_strength(value, power) * scale


def _mask_from_indices(landmarks: np.ndarray, indices: Sequence[int], shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array([landmarks[i] for i in indices if 0 <= i < len(landmarks)], dtype=np.float32)
    if len(pts) == 0:
        return mask
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2)
    return mask


def _convex_mask_from_indices(
    landmarks: np.ndarray,
    indices: Sequence[int],
    shape: Tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array([landmarks[i] for i in indices if 0 <= i < len(landmarks)], dtype=np.float32)
    if len(pts) < 3:
        return mask
    hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def _normalize_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    if not np.any(mask):
        return mask
    blurred = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1, sigma))
    return cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)


def _limit_displacement(src_pts: np.ndarray, dst_pts: np.ndarray, max_shift: float) -> np.ndarray:
    if max_shift <= 0:
        return dst_pts
    delta = dst_pts - src_pts
    delta = np.clip(delta, -max_shift, max_shift)
    return src_pts + delta


def _mask_bounds(
    mask: np.ndarray,
    padding: int,
    shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    if mask is None or not np.any(mask):
        return None
    h, w = shape
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    pad = max(0, padding)
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y1 = min(int(ys.max()) + pad + 1, h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _extrapolate_forehead_points(
    landmarks: np.ndarray,
    brow_indices: Sequence[int],
    oval_points: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    if not brow_indices or oval_points.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    h, w = shape
    brow_pts = np.array([landmarks[i] for i in brow_indices], dtype=np.float32)
    if brow_pts.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    face_center = np.mean(oval_points, axis=0)
    chin_y = float(np.max(oval_points[:, 1]))
    brow_y = float(np.mean(brow_pts[:, 1]))
    vertical_span = max(chin_y - brow_y, 1.0)
    extend_base = vertical_span * 0.45
    face_width = max(np.ptp(oval_points[:, 0]), 1.0)

    forehead_pts: List[np.ndarray] = []
    for pt in brow_pts:
        lateral_weight = 0.85 + 0.4 * abs(pt[0] - face_center[0]) / face_width
        new_y = max(0.0, pt[1] - extend_base * lateral_weight)
        horizontal_shift = (pt[0] - face_center[0]) * 0.04
        new_x = np.clip(pt[0] + horizontal_shift, 0.0, w - 1.0)
        forehead_pts.append(np.array([new_x, new_y], dtype=np.float32))

    if forehead_pts:
        top_y = min(pt[1] for pt in forehead_pts)
        apex = np.array(
            [face_center[0], max(0.0, top_y - extend_base * 0.2)],
            dtype=np.float32,
        )
        forehead_pts.append(apex)

    return np.array(forehead_pts, dtype=np.float32)


def _build_full_face_mask(landmarks: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    face_mask = np.zeros((h, w), dtype=np.uint8)

    base_points = np.array([landmarks[i] for i in FACE_OVAL_INDICES], dtype=np.float32)
    if base_points.size == 0:
        return face_mask

    forehead_pts = _extrapolate_forehead_points(
        landmarks, FOREHEAD_BASE_INDICES, base_points, shape
    )
    combined = base_points
    if forehead_pts.size > 0:
        combined = np.concatenate([combined, forehead_pts], axis=0)

    hull = cv2.convexHull(combined.astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(face_mask, hull, 255)

    sigma = max(1, int(w * 0.01))
    face_mask = cv2.GaussianBlur(face_mask, (0, 0), sigmaX=sigma)
    return face_mask


def _build_glow_mask(analysis: Optional[FaceAnalysis], shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    glow = np.zeros((h, w), dtype=np.uint8)
    if not analysis:
        glow.fill(200)
        return cv2.GaussianBlur(glow, (0, 0), sigmaX=9)

    x, y, bw, bh = analysis.bbox
    center = (int(x + bw * 0.5), int(y + bh * 0.55))
    cv2.ellipse(
        glow,
        center,
        (max(1, int(bw * 0.32)), max(1, int(bh * 0.4))),
        0,
        0,
        360,
        210,
        -1,
    )
    cheek_radius = max(4, int(bw * 0.18))
    left_cheek_center = (int(center[0] - bw * 0.22), int(center[1] - bh * 0.05))
    right_cheek_center = (int(center[0] + bw * 0.22), int(center[1] - bh * 0.05))
    cv2.circle(glow, left_cheek_center, cheek_radius, 255, -1)
    cv2.circle(glow, right_cheek_center, cheek_radius, 255, -1)
    bridge_center = (int(center[0]), int(y + bh * 0.25))
    cv2.ellipse(
        glow,
        bridge_center,
        (max(1, int(bw * 0.12)), max(1, int(bh * 0.25))),
        0,
        0,
        360,
        230,
        -1,
    )
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=13)
    return glow


def _restore_texture(
    original: np.ndarray, current: np.ndarray, mask: np.ndarray, texture_value: float
) -> np.ndarray:
    delta = texture_value - 50
    if delta == 0:
        return current

    mask_soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=3)
    orig_f = original.astype(np.float32)
    curr_f = current.astype(np.float32)
    detail = orig_f - cv2.GaussianBlur(orig_f, (0, 0), sigmaX=1.4)

    if delta > 0:
        strength = delta / 50.0
        boosted = np.clip(curr_f + detail * (0.45 * strength), 0, 255).astype(np.uint8)
        return _blend(
            current,
            boosted,
            mask_soft,
            alpha=min(0.6, 0.3 + strength * 0.35),
        )

    strength = -delta / 50.0
    softened = cv2.GaussianBlur(current, (0, 0), sigmaX=1 + strength * 2.5)
    return _blend(current, softened, mask_soft, alpha=0.25 + strength * 0.2)


def _vector_warp(image: np.ndarray, mask: np.ndarray, vectors: List[Tuple[Tuple[float, float], float, Tuple[float, float]]]) -> np.ndarray:
    if not vectors:
        return image

    h, w = image.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    total_dx = np.zeros((h, w), dtype=np.float32)
    total_dy = np.zeros((h, w), dtype=np.float32)

    mask_norm = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=5)

    for (cx, cy), radius, (sx, sy) in vectors:
        if radius <= 1:
            continue
        dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        weight = np.exp(-dist_sq / (2 * radius * radius))
        total_dx += sx * weight
        total_dy += sy * weight

    total_dx *= mask_norm
    total_dy *= mask_norm

    if MAX_SMILE_SHIFT_PX > 0:
        total_dx = np.clip(total_dx, -MAX_SMILE_SHIFT_PX, MAX_SMILE_SHIFT_PX)
        total_dy = np.clip(total_dy, -MAX_SMILE_SHIFT_PX, MAX_SMILE_SHIFT_PX)

    map_x = np.clip(grid_x - total_dx, 0, w - 1)
    map_y = np.clip(grid_y - total_dy, 0, h - 1)

    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _blend_mode_overlay(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    base_f = base.astype(np.float32) / 255.0
    overlay_f = overlay.astype(np.float32) / 255.0
    blended = np.where(
        base_f <= 0.5,
        2.0 * base_f * overlay_f,
        1.0 - 2.0 * (1.0 - base_f) * (1.0 - overlay_f),
    )
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


def _blend_mode_soft_light(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    base_f = base.astype(np.float32) / 255.0
    overlay_f = overlay.astype(np.float32) / 255.0
    blended = (1.0 - 2.0 * overlay_f) * (base_f**2) + 2.0 * overlay_f * base_f
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


def _thin_plate_warp(
    image: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    if src_pts.shape[0] < 3 or dst_pts.shape[0] < 3:
        return image
    src = src_pts.reshape(-1, 1, 2).astype(np.float32)
    dst = dst_pts.reshape(-1, 1, 2).astype(np.float32)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(dst, src, matches)
    warped = tps.warpImage(image)
    return _blend(image, warped, mask, alpha=1.0)


def _build_iris_regions(
    landmarks: np.ndarray,
    iris_pairs: Sequence[Sequence[int]],
    shape: Tuple[int, int],
) -> List[IrisRegion]:
    regions: List[IrisRegion] = []
    h, w = shape
    for indices in iris_pairs:
        pts = np.array([landmarks[i] for i in indices], dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        if pts.size == 0:
            regions.append(
                IrisRegion(
                    mask=mask,
                    center=(0.0, 0.0),
                    radius=0.0,
                    pupil_mask=np.zeros_like(mask),
                )
            )
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(pts.astype(np.float32))
        center = (float(cx), float(cy))
        r_int = max(1, int(radius))
        cv2.circle(mask, (int(cx), int(cy)), r_int, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1, int(radius * 0.6)))
        pupil_mask = np.zeros((h, w), dtype=np.uint8)
        pupil_radius = max(1, int(radius * 0.38))
        cv2.circle(pupil_mask, (int(cx), int(cy)), pupil_radius, 255, -1)
        pupil_mask = cv2.GaussianBlur(
            pupil_mask, (0, 0), sigmaX=max(1, int(pupil_radius * 0.8))
        )
        regions.append(
            IrisRegion(
                mask=mask,
                center=center,
                radius=float(radius),
                pupil_mask=pupil_mask,
            )
        )
    return regions


def _build_under_eye_mask(
    landmarks: np.ndarray,
    lower_indices: Sequence[int],
    shape: Tuple[int, int],
    offset: float,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array([landmarks[i] for i in lower_indices], dtype=np.float32)
    if pts.size == 0:
        return mask
    shift = np.array([0.0, offset], dtype=np.float32)
    extended = np.concatenate([pts, pts + shift], axis=0)
    hull = cv2.convexHull(extended.astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask, hull, 255)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=5)


def _frequency_separation_lighten(
    image: np.ndarray, mask: np.ndarray, strength: float
) -> np.ndarray:
    if strength <= 0:
        return image
    sigma = 4.0 + 4.0 * strength
    low = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    high = image.astype(np.float32) - low.astype(np.float32)
    lift = np.clip(low.astype(np.float32) + 35.0 * strength, 0, 255).astype(np.uint8)
    recombined = np.clip(lift.astype(np.float32) + high, 0, 255).astype(np.uint8)
    return _blend(image, recombined, mask, alpha=0.65 + 0.15 * strength)


def _cubic_bezier(points: Sequence[np.ndarray], steps: int = 60) -> np.ndarray:
    if len(points) < 4:
        return np.array(points, dtype=np.float32)
    p0, p1, p2, p3 = [np.array(p, dtype=np.float32) for p in points[:4]]
    ts = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    one_minus = 1.0 - ts
    curve = (
        (one_minus**3)[:, None] * p0
        + 3.0 * (one_minus**2 * ts)[:, None] * p1
        + 3.0 * (one_minus * (ts**2))[:, None] * p2
        + (ts**3)[:, None] * p3
    )
    return curve.astype(np.float32)


def _build_eyeliner_mask(
    landmarks: np.ndarray,
    upper_indices: Sequence[int],
    shape: Tuple[int, int],
    thickness: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = [landmarks[i] for i in upper_indices]
    if len(pts) < 4:
        return mask
    curve = _cubic_bezier(pts, steps=80)
    curve_int = curve.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [curve_int], False, 255, thickness=thickness)
    cv2.polylines(
        mask, [curve_int + np.array([0, thickness // 2]).reshape(1, 1, 2)], False, 255, thickness=max(1, thickness // 2)
    )
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=thickness * 0.8)


def _stroke_lash_curve(
    overlay: np.ndarray,
    mask: np.ndarray,
    root: np.ndarray,
    direction: np.ndarray,
    length: float,
    base_thickness: int,
    base_color: Tuple[int, int, int],
) -> None:
    if length <= 0:
        return
    dir_norm = direction / (np.linalg.norm(direction) + 1e-6)
    tip = root + dir_norm * length
    ortho = np.array([-dir_norm[1], dir_norm[0]], dtype=np.float32)
    ctrl1 = root + dir_norm * (length * 0.35) + ortho * (length * 0.18)
    ctrl2 = root + dir_norm * (length * 0.8) + ortho * (length * 0.06)
    curve = _cubic_bezier([root, ctrl1, ctrl2, tip], steps=28)
    base_color_arr = np.array(base_color, dtype=np.float32)
    for i in range(len(curve) - 1):
        t = i / max(1, len(curve) - 2)
        thickness = max(1, int(round(base_thickness * (1.0 - 0.7 * t))))
        start = tuple(np.round(curve[i]).astype(int))
        end = tuple(np.round(curve[i + 1]).astype(int))
        shade = np.clip(base_color_arr * (1.0 - 0.4 * t), 0, 255).astype(int)
        color = (int(shade[0]), int(shade[1]), int(shade[2]))
        cv2.line(
            overlay,
            start,
            end,
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            mask,
            start,
            end,
            255,
            thickness=max(1, thickness - 1),
            lineType=cv2.LINE_AA,
        )


def _render_lash_layer(
    shape: Tuple[int, int],
    landmarks: np.ndarray,
    upper_indices: Sequence[int],
    face_width: float,
    sample_step: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = [landmarks[i] for i in upper_indices if 0 <= i < len(landmarks)]
    if len(pts) < 3 or face_width <= 0:
        return overlay, mask
    curve = _cubic_bezier(pts, steps=160)
    base_length = max(face_width * 0.05, 10.0)
    base_thickness = max(1, int(face_width * 0.004))
    base_color = (12, 10, 8)
    stride = max(2, len(curve) // (len(pts) * sample_step))
    eye_center_x = float(np.mean(curve[:, 0]))
    for idx in range(stride, len(curve) - stride, stride):
        if (idx // stride) % sample_step != 0:
            continue
        root = curve[idx]
        prev_pt = curve[max(0, idx - stride)]
        next_pt = curve[min(len(curve) - 1, idx + stride)]
        tangent = next_pt - prev_pt
        if np.linalg.norm(tangent) < 1e-3:
            continue
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        if normal[1] > 0:
            normal[1] *= -1
        outward = 1.0 if root[0] > eye_center_x else -1.0
        normal[0] += outward * 0.35
        direction = normal / (np.linalg.norm(normal) + 1e-6)
        lash_length = base_length * (0.8 + 0.6 * abs(0.5 - idx / len(curve)))
        _stroke_lash_curve(
            overlay,
            mask,
            root,
            direction,
            lash_length,
            base_thickness,
            base_color,
        )
    if np.any(mask):
        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=1.1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1, int(base_thickness)))
    return overlay, mask


LIPS_OUTER_INDICES: List[int] = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
]
LIPS_INNER_INDICES: List[int] = [
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
]
CUPIDS_BOW = {"left": 185, "right": 40, "center": 13, "bottom": 14}
MOUTH_CORNERS = {"left": 61, "right": 291}
LIPS_UPPER_OUTER_INDICES: List[int] = [
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    291,
]
LIP_TPS_INDICES: List[int] = sorted(
    set(LIPS_OUTER_INDICES + LIPS_INNER_INDICES + list(CUPIDS_BOW.values()))
)

LIPSTICK_COLORS = {
    "nude_pink": (192, 178, 215),
    "earthy_pink": (164, 102, 184),
    "cherry_red": (70, 20, 200),
    "wine_red": (40, 0, 120),
    "coral": (90, 140, 255),
}


def _soft_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    if not np.any(mask):
        return mask
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1, sigma))


def _build_lip_masks(
    landmarks: np.ndarray,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    outer = np.zeros((h, w), dtype=np.uint8)
    outer_pts: List[np.ndarray] = []
    upper_offset = 2.0
    upper_set = set(LIPS_UPPER_OUTER_INDICES)
    for idx in LIPS_OUTER_INDICES:
        if idx >= len(landmarks):
            continue
        pt = np.array(landmarks[idx], dtype=np.float32)
        if idx in upper_set:
            pt = pt.copy()
            pt[1] += upper_offset
        outer_pts.append(pt)
    if len(outer_pts) >= 3:
        hull = cv2.convexHull(np.array(outer_pts, dtype=np.float32)).astype(np.int32)
        cv2.fillConvexPoly(outer, hull, 255)
    inner = _convex_mask_from_indices(landmarks, LIPS_INNER_INDICES, shape)
    if not np.any(outer):
        zeros = np.zeros((h, w), dtype=np.uint8)
        return zeros, zeros, zeros

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    outer = cv2.morphologyEx(outer, cv2.MORPH_CLOSE, kernel, iterations=1)
    inner = cv2.morphologyEx(inner, cv2.MORPH_CLOSE, kernel, iterations=1)
    inner = cv2.erode(inner, kernel, iterations=1)

    upper_mask = np.zeros((h, w), dtype=np.uint8)
    upper_pts = [
        np.array(landmarks[i], dtype=np.float32) + np.array([0.0, upper_offset])
        for i in LIPS_UPPER_OUTER_INDICES
        if 0 <= i < len(landmarks)
    ]
    if len(upper_pts) >= 3:
        poly = np.array(upper_pts, dtype=np.float32).astype(np.int32)
        cv2.fillPoly(upper_mask, [poly], 255)
        upper_mask = cv2.morphologyEx(upper_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        tight_upper = cv2.erode(
            upper_mask, np.ones((3, 3), dtype=np.uint8), iterations=1
        )
        preserved_lower = cv2.subtract(outer, upper_mask)
        outer = cv2.max(preserved_lower, tight_upper)

    refined_outer = _normalize_mask(outer, sigma=4.0)
    refined_inner = _normalize_mask(inner, sigma=2.2)
    lip_band = cv2.subtract(refined_outer, refined_inner)
    lip_band = _normalize_mask(lip_band, sigma=3.5)
    return refined_outer, refined_inner, lip_band


def _mouth_open_ratio(outer_mask: np.ndarray, inner_mask: np.ndarray) -> float:
    outer_area = max(1, int(np.count_nonzero(outer_mask)))
    inner_area = np.count_nonzero(inner_mask)
    return inner_area / outer_area


def _estimate_teeth_mask(image: np.ndarray, inner_mask: np.ndarray, lip_mask: np.ndarray) -> np.ndarray:
    if not np.any(inner_mask):
        return np.zeros_like(inner_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core_mask = cv2.erode(inner_mask, kernel, iterations=1)
    if lip_mask is not None and np.any(lip_mask):
        lip_guard = cv2.dilate(lip_mask, kernel, iterations=1)
        core_mask = cv2.subtract(core_mask, lip_guard)
    if not np.any(core_mask):
        return np.zeros_like(inner_mask)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)

    mask_bool = core_mask > 0
    neutral_ab = (a < 155) & (b < 155)
    bright = v > 120
    low_sat = s < 110
    gum_tongue = (a > 165) | (b > 170) | ((h > 5) & (h < 35))

    candidate = np.zeros_like(inner_mask)
    selection = bright & low_sat & neutral_ab & (~gum_tongue)
    candidate[mask_bool] = (selection[mask_bool] * 255).astype(np.uint8)
    candidate = cv2.medianBlur(candidate, 5)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=1)
    return _soft_mask(candidate, 1.8)


def _apply_lipstick(
    image: np.ndarray,
    lip_mask: np.ndarray,
    inner_mask: np.ndarray,
    shade: str,
) -> np.ndarray:
    if shade not in LIPSTICK_COLORS or not np.any(lip_mask):
        return image

    mask = lip_mask.copy()
    if np.any(inner_mask):
        inner_guard = cv2.erode(inner_mask, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.subtract(mask, inner_guard)
    if not np.any(mask):
        return image
    mask = _soft_mask(mask, 3.0)

    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    tint_color = np.full_like(image, LIPSTICK_COLORS[shade], dtype=np.uint8)
    mixed = cv2.addWeighted(image, 0.35, tint_color, 0.65, 0)
    soft_light = _blend_mode_soft_light(image, mixed)
    overlay = _blend_mode_overlay(image, tint_color)
    keep_texture = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    keep_texture = cv2.GaussianBlur(keep_texture, (0, 0), sigmaX=1.1)[..., None]
    gloss = np.clip(
        soft_light.astype(np.float32) * 0.65
        + overlay.astype(np.float32) * 0.35,
        0,
        255,
    )
    gloss *= (0.85 + 0.15 * keep_texture)
    gloss = np.clip(gloss, 0, 255).astype(np.uint8)
    return _blend(image, gloss, mask, alpha=0.9)


def _apply_lip_gloss(
    image: np.ndarray,
    lip_mask: np.ndarray,
    landmarks: np.ndarray,
    strength: float = 0.4,
) -> np.ndarray:
    if strength <= 0 or not np.any(lip_mask):
        return image
    bottom_idx = CUPIDS_BOW["bottom"]
    if bottom_idx >= len(landmarks):
        return image
    bottom_point = landmarks[bottom_idx]
    lip_pts = landmarks[LIPS_OUTER_INDICES]
    width = max(np.ptp(lip_pts[:, 0]), 1.0)
    gloss_mask = np.zeros_like(lip_mask)
    center = (int(bottom_point[0]), int(bottom_point[1] - width * 0.04))
    axes = (max(1, int(width * 0.22)), max(1, int(width * 0.09)))
    cv2.ellipse(gloss_mask, center, axes, 0, 0, 360, 255, -1)
    gloss_mask = _soft_mask(gloss_mask, axes[0] * 0.45)
    gloss_mask = cv2.min(gloss_mask, lip_mask)

    overlay = np.clip(image.astype(np.float32) + 90.0 * strength, 0, 255).astype(np.uint8)
    return _blend(image, overlay, gloss_mask, alpha=0.35 + 0.25 * strength)


def _apply_smile_warp(
    image: np.ndarray,
    landmarks: np.ndarray,
    lip_mask: np.ndarray,
    smile_strength: float,
    face_w: float,
    face_h: float,
) -> np.ndarray:
    if smile_strength == 0 or not np.any(lip_mask):
        return image
    max_shift = min(MAX_SMILE_SHIFT_PX, face_w * 0.04)
    up_shift = np.clip(face_h * 0.022 * smile_strength, -max_shift, max_shift)
    out_shift = np.clip(face_w * 0.015 * smile_strength, -max_shift, max_shift)
    lip_soft = _soft_mask(lip_mask, 5.0)
    left_corner = tuple(landmarks[MOUTH_CORNERS["left"]])
    right_corner = tuple(landmarks[MOUTH_CORNERS["right"]])
    vectors = [
        (left_corner, face_w * 0.12, (-out_shift, -up_shift)),
        (right_corner, face_w * 0.12, (out_shift, -up_shift)),
    ]
    return _vector_warp(image, lip_soft, vectors)


def _apply_mouth_pipeline(
    image: np.ndarray,
    landmarks: np.ndarray,
    mouth_vals: MouthValues,
    lipstick: str,
    face_w: float,
    face_h: float,
) -> np.ndarray:
    if landmarks is None:
        return image
    h, w = image.shape[:2]
    outer_mask, inner_mask, lip_band_mask = _build_lip_masks(landmarks, (h, w))
    if not np.any(lip_band_mask):
        return image

    result = image.copy()

    if lipstick != "none":
        result = _apply_lipstick(result, lip_band_mask, inner_mask, lipstick)
        result = _apply_lip_gloss(result, lip_band_mask, landmarks, strength=0.35)

    smile_strength = _ease_strength(mouth_vals.smile, 1.1)
    if smile_strength != 0:
        result = _apply_smile_warp(
            result, landmarks, outer_mask, smile_strength, face_w, face_h
        )

    return result


def _add_catch_light_precise(
    image: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    angle_deg: float,
    strength: float,
) -> np.ndarray:
    if radius <= 0 or strength <= 0:
        return image
    overlay = image.copy()
    angle_rad = math.radians(angle_deg)
    offset_radius = radius * (0.45 + 0.15 * strength)
    position = (
        int(round(center[0] + offset_radius * math.sin(angle_rad))),
        int(round(center[1] - offset_radius * math.cos(angle_rad))),
    )
    size = max(1, int(radius * (0.15 + 0.1 * strength)))
    cv2.circle(overlay, position, size, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    blur = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(1, int(size * 0.9)))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, position, max(1, int(size * 2.5)), 255, -1)
    return _blend(image, blur, mask, alpha=min(0.7, 0.35 + strength * 0.35))


def _apply_iris_ring_highlight(
    image: np.ndarray,
    region: IrisRegion,
    strength: float,
    catch_angle_deg: float,
) -> np.ndarray:
    ring_mask = _iris_ring_mask(region)
    if strength <= 0 or not np.any(ring_mask):
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    mask_bool = ring_mask > 0
    if not np.any(mask_bool):
        return image
    s_scale = 1.0 + 0.25 * strength
    v_scale = 1.0 + 0.45 * strength
    s[mask_bool] = np.clip(s[mask_bool] * s_scale, 0, 255)
    v[mask_bool] = np.clip(
        v[mask_bool] * v_scale + 35.0 * strength,
        0,
        255,
    )
    enhanced = cv2.cvtColor(
        cv2.merge(
            [
                h.astype(np.uint8),
                s.astype(np.uint8),
                v.astype(np.uint8),
            ]
        ),
        cv2.COLOR_HSV2BGR,
    )
    blend_alpha = _alpha_from_strength(0.3, 0.4, strength)
    result = _blend(image, enhanced, ring_mask, alpha=blend_alpha)
    return _add_catch_light_precise(result, region.center, region.radius, catch_angle_deg, strength)


def _apply_iris_donut_brightening(
    image: np.ndarray,
    region: IrisRegion,
    strength: float,
) -> np.ndarray:
    if strength <= 0:
        return image
    donut_mask, pupil_guard = _iris_donut_mask(region)
    if not np.any(donut_mask):
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=1.5 + 2.0 * strength,
        tileGridSize=(4, 4),
    )
    l_boost = clahe.apply(l)
    mask_norm = donut_mask.astype(np.float32) / 255.0
    mix = 0.45 + 0.4 * strength
    l = np.clip(
        l.astype(np.float32) * (1.0 - mask_norm * mix) + l_boost.astype(np.float32) * (mask_norm * mix),
        0,
        255,
    ).astype(np.uint8)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    result = _blend(
        image,
        enhanced,
        donut_mask,
        alpha=_alpha_from_strength(0.4, 0.35, strength),
    )
    if np.any(pupil_guard):
        result[pupil_guard > 0] = 0
    return result


def _local_eye_enlarge(
    image: np.ndarray,
    region: IrisRegion,
    eye_mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    if region.radius <= 0 or strength == 0 or not np.any(eye_mask):
        return image

    cx, cy = region.center
    h, w = image.shape[:2]
    influence = region.radius * (1.2 + 0.25 * abs(strength))
    x0 = max(int(cx - influence * 1.2), 0)
    x1 = min(int(cx + influence * 1.2) + 1, w)
    y0 = max(int(cy - influence * 1.2), 0)
    y1 = min(int(cy + influence * 1.2) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return image

    grid_x, grid_y = np.meshgrid(
        np.arange(x0, x1, dtype=np.float32),
        np.arange(y0, y1, dtype=np.float32),
    )
    dx = grid_x - cx
    dy = grid_y - cy
    dist = np.sqrt(dx**2 + dy**2)
    norm = dist / max(influence, 1.0)
    falloff = np.clip(1.0 - norm**2, 0.0, 1.0)
    magnitude = 0.35 * abs(strength)
    if strength > 0:
        scale = 1.0 + magnitude * falloff
    else:
        scale = np.clip(1.0 - magnitude * falloff, 0.3, 1.0)

    map_x = (cx + dx / scale).astype(np.float32)
    map_y = (cy + dy / scale).astype(np.float32)
    warped_roi = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    result = image.copy()
    roi_mask = eye_mask[y0:y1, x0:x1]
    roi_mask = cv2.GaussianBlur(roi_mask, (0, 0), sigmaX=3)
    alpha = _alpha_from_strength(0.4, 0.35, abs(strength))
    result[y0:y1, x0:x1] = _blend(
        image[y0:y1, x0:x1],
        warped_roi,
        roi_mask,
        alpha=alpha,
    )
    return result


LENS_PRESETS = {
    "natural_brown": {"hue": 15, "sat_boost": 0.3, "value_boost": 0.05, "mix": 0.65, "alpha": 0.55},
    "cool_brown": {"hue": 10, "sat_boost": 0.2, "value_boost": 0.08, "mix": 0.6, "alpha": 0.55},
    "gray": {"hue": 0, "sat_boost": -0.35, "value_boost": 0.12, "mix": 0.5, "alpha": 0.5},
    "smoky_blue": {"hue": 105, "sat_boost": 0.4, "value_boost": 0.05, "mix": 0.7, "alpha": 0.6},
}


def _tint_iris_hsv(
    image: np.ndarray,
    iris_mask: np.ndarray,
    lens: str,
) -> np.ndarray:
    preset = LENS_PRESETS.get(lens)
    if not preset or not np.any(iris_mask):
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    mask_bool = iris_mask > 0
    hue_mix = preset["mix"]
    target_hue = preset["hue"]
    h[mask_bool] = (1.0 - hue_mix) * h[mask_bool] + hue_mix * target_hue
    s_scale = 1.0 + preset["sat_boost"]
    v_scale = 1.0 + preset["value_boost"]
    s[mask_bool] = np.clip(s[mask_bool] * s_scale, 0, 255)
    v[mask_bool] = np.clip(v[mask_bool] * v_scale, 0, 255)
    tinted = cv2.cvtColor(
        cv2.merge(
            [
                h.astype(np.uint8),
                s.astype(np.uint8),
                v.astype(np.uint8),
            ]
        ),
        cv2.COLOR_HSV2BGR,
    )
    overlay = _blend_mode_overlay(image, tinted)
    return _blend(image, overlay, iris_mask, alpha=preset["alpha"])


def _add_catch_light(
    image: np.ndarray, center: Tuple[float, float], radius: float, intensity: float
) -> np.ndarray:
    if radius <= 0 or intensity <= 0:
        return image
    overlay = image.copy()
    size = max(1, int(radius * 0.35))
    offset = (int(center[0] - radius * 0.3), int(center[1] - radius * 0.4))
    cv2.circle(overlay, offset, size, (255, 255, 255), -1)
    blur = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(1, int(size * 0.8)))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, offset, size * 2, 255, -1)
    return _blend(image, blur, mask, alpha=min(0.6, 0.3 + intensity * 0.3))


def _apply_skin_pipeline(image: np.ndarray, analysis: Optional[FaceAnalysis], config: BeautyConfig) -> np.ndarray:
    base_image = image.copy()
    result = image.copy()
    h, w = result.shape[:2]
    mask = analysis.mask if analysis else np.full((h, w), 255, dtype=np.uint8)
    mask_soft = cv2.GaussianBlur(mask, (0, 0), 5)

    skin = config.skinValues

    if skin.smooth > 0:
        smooth_strength = skin.smooth / 100
        sigma_color = 50 + smooth_strength * 80
        sigma_space = 30 + smooth_strength * 60
        smooth_img = cv2.bilateralFilter(result, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        result = _blend(result, smooth_img, mask_soft, alpha=0.35 + 0.4 * smooth_strength)

    if config.skinMode == "strong":
        # preserve pores by mixing unsharp mask
        blur = cv2.GaussianBlur(result, (0, 0), 3)
        high = cv2.addWeighted(result, 1.5, blur, -0.5, 0)
        result = _blend(result, high, mask_soft, alpha=0.25)

    if skin.whiten > 0 or skin.even > 0:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Whitening giữ nguyên behaviour cũ
        if skin.whiten > 0:
            l = cv2.normalize(
                l,
                None,
                alpha=0,
                beta=255 + skin.whiten * 0.8,
                norm_type=cv2.NORM_MINMAX,
            )

        # Đều màu da: tăng độ nhạy theo slider thay vì gần như 1 mức
        if skin.even > 0:
            # Strength 0–1 với easing nhẹ để đầu slider vẫn dịu, cuối mạnh hơn
            even_strength = _ease_strength(skin.even, power=1.1)

            # CLAHE mạnh hơn khi strength cao (đều sáng vùng da)
            clip_limit = 1.5 + 2.0 * even_strength  # 1.5 → 3.5
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(8, 8),
            )
            l = clahe.apply(l)

            # Chuẩn hoá sắc độ a/b quanh mean để giảm loang màu không đều
            mean_a = np.mean(a[mask > 0])
            mean_b = np.mean(b[mask > 0])

            a_f = a.astype(np.float32)
            b_f = b.astype(np.float32)

            a = cv2.addWeighted(
                a_f,
                1 - even_strength,
                float(mean_a) * np.ones_like(a_f),
                even_strength,
                0,
            )
            b = cv2.addWeighted(
                b_f,
                1 - even_strength,
                float(mean_b) * np.ones_like(b_f),
                even_strength,
                0,
            )

            # convert lại uint8
            a = np.clip(a, 0, 255).astype(np.uint8)
            b = np.clip(b, 0, 255).astype(np.uint8)

        lab = cv2.merge([l, a, b])
        toned = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Alpha blend phụ thuộc strength để slider thay đổi rõ rệt hơn
        if skin.even > 0:
            strength_for_alpha = _ease_strength(skin.even, power=1.1)
            alpha = _alpha_from_strength(0.25, 0.35, strength_for_alpha)
        else:
            alpha = 0.4

        result = _blend(result, toned, mask_soft, alpha=alpha)

    if skin.korean != 0:
        glow_mask = _build_glow_mask(analysis, (h, w))
        glow_mask = cv2.normalize(glow_mask, None, 0, 255, cv2.NORM_MINMAX)
        glow_strength = abs(_ease_strength(skin.korean, 1.05))
        softened = cv2.bilateralFilter(result, d=7, sigmaColor=85, sigmaSpace=70)
        sheen = cv2.detailEnhance(softened, sigma_s=12, sigma_r=0.25)
        sheen = cv2.addWeighted(softened, 0.65, sheen, 0.35, 8 * glow_strength)
        sheen = cv2.addWeighted(
            sheen,
            1.0 + 0.08 * glow_strength,
            result,
            -0.08 * glow_strength,
            0,
        )
        adaptive_mask = cv2.GaussianBlur(
            glow_mask, (0, 0), sigmaX=5 + int(6 * glow_strength)
        )
        alpha = _alpha_from_strength(0.25, 0.45, glow_strength)
        result = _blend(result, sheen, adaptive_mask, alpha=alpha)

    if skin.texture != 50:
        result = _restore_texture(base_image, result, mask, skin.texture)

    # Acne auto removal
    if config.acneMode.auto:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), 2)
        high_pass = cv2.subtract(gray, blur)
        _, spots = cv2.threshold(high_pass, 10, 255, cv2.THRESH_BINARY)
        spots = cv2.bitwise_and(spots, mask)
        spots = cv2.dilate(spots, np.ones((3, 3), np.uint8), iterations=1)
        cleaned = cv2.inpaint(result, spots, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        result = _blend(result, cleaned, spots, alpha=0.8)

    # Manual acne healing
    if config.acneMode.manualPoints:
        heal_mask = np.zeros_like(mask)
        for point in config.acneMode.manualPoints:
            cx = int(point.x / 100 * w)
            cy = int(point.y / 100 * h)
            cv2.circle(heal_mask, (cx, cy), max(4, int(min(w, h) * 0.01)), 255, -1)
        healed = cv2.inpaint(result, heal_mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)
        result = _blend(result, healed, heal_mask, alpha=0.9)

    return result


def _apply_face_sculpt(image: np.ndarray, analysis: Optional[FaceAnalysis], config: BeautyConfig) -> np.ndarray:
    if not analysis:
        return image
    vectors: List[Tuple[Tuple[float, float], float, Tuple[float, float]]] = []
    landmarks = analysis.landmarks
    bbox = analysis.bbox
    face_w = bbox[2]
    face_h = bbox[3]

    def point(idx: int) -> Tuple[float, float]:
        x, y = landmarks[idx]
        return float(x), float(y)

    def add_vector(
        anchor: Tuple[float, float],
        radius_factor: float,
        shift: Tuple[float, float],
        axis: str = "w",
    ) -> None:
        scale = face_w if axis == "w" else face_h
        vectors.append((anchor, max(2.0, scale * radius_factor), shift))

    left_cheek = point(205)
    right_cheek = point(425)
    left_jaw = point(172)
    right_jaw = point(397)
    chin = point(152)
    mouth = point(13)
    nose = point(1)
    forehead = point(10)
    left_temple = point(54)
    right_temple = point(284)
    nose_tip = point(4)
    columella = point(2)
    left_alar = point(94)
    right_alar = point(331)
    bridge_top = point(6)
    bridge_mid = point(168)
    cheekbone_left = point(50)
    cheekbone_right = point(280)

    fv = config.faceValues

    slim_strength = _feature_strength(fv.slim, "slim", 1.1)
    if slim_strength != 0:
        mag = face_w * 0.04 * abs(slim_strength)
        vertical = -face_h * 0.008 * slim_strength
        shift = mag if slim_strength > 0 else -mag
        add_vector(left_cheek, 0.34, (shift, vertical))
        add_vector(right_cheek, 0.34, (-shift, vertical))
        add_vector(cheekbone_left, 0.22, (shift * 0.55, vertical * 0.5))
        add_vector(cheekbone_right, 0.22, (-shift * 0.55, vertical * 0.5))

    vline_strength = _feature_strength(fv.vline, "vline", 1.2)
    if vline_strength != 0:
        mag = face_w * 0.04 * abs(vline_strength)
        jaw_shift = mag if vline_strength > 0 else -mag
        lift = face_h * 0.012 * abs(vline_strength)
        add_vector(left_jaw, 0.28, (jaw_shift, -lift))
        add_vector(right_jaw, 0.28, (-jaw_shift, -lift))
        add_vector(left_cheek, 0.24, (jaw_shift * 0.25, -lift * 0.6))
        add_vector(right_cheek, 0.24, (-jaw_shift * 0.25, -lift * 0.6))

    chin_strength = _feature_strength(fv.chinShrink, "chin", 1.1)
    if chin_strength != 0:
        mag = face_h * 0.055 * abs(chin_strength)
        direction = -mag if chin_strength > 0 else mag
        add_vector(chin, 0.3, (0, direction), axis="h")
        add_vector(mouth, 0.18, (0, direction * 0.45), axis="h")
        add_vector(
            (chin[0], chin[1] - face_h * 0.08),
            0.22,
            (0, direction * 0.2),
            axis="h",
        )

    forehead_strength = _feature_strength(fv.forehead, "forehead", 1.05)
    if forehead_strength != 0:
        mag = face_h * 0.038 * forehead_strength
        temple_mag = face_w * 0.018 * forehead_strength
        add_vector(forehead, 0.4, (0, mag), axis="h")
        add_vector(left_temple, 0.22, (-temple_mag, mag * 0.12))
        add_vector(right_temple, 0.22, (temple_mag, mag * 0.12))

    jaw_strength = _feature_strength(fv.jaw, "jaw", 1.1)
    if jaw_strength != 0:
        mag = face_w * 0.04 * abs(jaw_strength)
        direction = mag if jaw_strength > 0 else -mag
        add_vector(left_jaw, 0.26, (direction, -mag * 0.2))
        add_vector(right_jaw, 0.26, (-direction, -mag * 0.2))
        contour_mag = face_w * 0.01 * abs(jaw_strength)
        add_vector(left_cheek, 0.2, (direction * 0.28, -contour_mag * 0.7))
        add_vector(right_cheek, 0.2, (-direction * 0.28, -contour_mag * 0.7))

    nose_slim_strength = _feature_strength(fv.noseSlim, "nose_slim", 1.15)
    if nose_slim_strength != 0:
        width_mag = face_w * 0.024 * abs(nose_slim_strength)
        direction = width_mag if nose_slim_strength > 0 else -width_mag
        add_vector(left_alar, 0.16, (direction, 0))
        add_vector(right_alar, 0.16, (-direction, 0))
        add_vector((nose[0] - face_w * 0.06, nose[1]), 0.14, (direction * 0.65, 0))
        add_vector((nose[0] + face_w * 0.06, nose[1]), 0.14, (-direction * 0.65, 0))
        tip_lift = face_h * 0.01 * nose_slim_strength
        add_vector(nose_tip, 0.12, (0, -tip_lift), axis="h")
        add_vector(columella, 0.1, (0, -tip_lift * 0.6), axis="h")

    nose_bridge_strength = _feature_strength(fv.noseBridge, "nose_bridge", 1.2)
    if nose_bridge_strength != 0:
        mag = face_h * 0.032 * nose_bridge_strength
        add_vector(bridge_top, 0.18, (0, -mag), axis="h")
        add_vector(bridge_mid, 0.14, (0, -mag * 0.65), axis="h")
        add_vector(nose_tip, 0.1, (0, -mag * 0.35), axis="h")

    sculpt_mix = max(abs(slim_strength), abs(vline_strength), abs(jaw_strength))
    if sculpt_mix > 0:
        lift = face_h * 0.016 * sculpt_mix
        add_vector(
            (left_cheek[0], left_cheek[1] - face_h * 0.1),
            0.22,
            (face_w * 0.006 * sculpt_mix, -lift),
        )
        add_vector(
            (right_cheek[0], right_cheek[1] - face_h * 0.1),
            0.22,
            (-face_w * 0.006 * sculpt_mix, -lift),
        )

    if not vectors:
        return image
    return _vector_warp(image, analysis.mask, vectors)


def _apply_eye_and_mouth(image: np.ndarray, analysis: Optional[FaceAnalysis], config: BeautyConfig) -> np.ndarray:
    if not analysis:
        return image

    result = image.copy()
    h, w = result.shape[:2]
    landmarks = analysis.landmarks
    face_w = analysis.bbox[2]
    face_h = analysis.bbox[3]

    left_eye_idx = [33, 160, 158, 133, 153, 144, 163, 7]
    right_eye_idx = [263, 387, 385, 362, 380, 373, 390, 249]
    left_upper_idx = [159, 158, 157, 173]
    right_upper_idx = [386, 385, 384, 373]
    left_lower_idx = [145, 153, 154, 155, 133]
    right_lower_idx = [374, 380, 381, 382, 263]
    left_iris_idx = [468, 469, 470, 471, 472]
    right_iris_idx = [473, 474, 475, 476, 477]
    left_eye_mask = _mask_from_indices(landmarks, left_eye_idx, (h, w))
    right_eye_mask = _mask_from_indices(landmarks, right_eye_idx, (h, w))
    eye_mask = cv2.max(left_eye_mask, right_eye_mask)
    iris_regions = _build_iris_regions(
        landmarks, [left_iris_idx, right_iris_idx], (h, w)
    )
    iris_mask = np.zeros((h, w), dtype=np.uint8)
    for region in iris_regions:
        iris_mask = cv2.max(iris_mask, region.mask)

    eye_vals = config.eyeValues
    mouth_vals = config.mouthValues

    if eye_vals.depth != 0:
        clahe = cv2.createCLAHE(clipLimit=2.0)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        depth_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        result = _blend(result, depth_img, eye_mask, alpha=eye_vals.depth / 120)

    enlarge_strength = _ease_strength(eye_vals.enlarge, 1.1)
    if enlarge_strength != 0 and iris_regions:
        if len(iris_regions) > 0:
            result = _local_eye_enlarge(
                result,
                iris_regions[0],
                left_eye_mask,
                enlarge_strength,
            )
        if len(iris_regions) > 1:
            result = _local_eye_enlarge(
                result,
                iris_regions[1],
                right_eye_mask,
                enlarge_strength,
            )

    if eye_vals.darkCircle > 0:
        dark_strength = min(1.0, eye_vals.darkCircle / 100.0)
        offset = face_h * (0.04 + 0.02 * dark_strength)
        left_dark_mask = _build_under_eye_mask(
            landmarks, left_lower_idx, (h, w), offset=offset
        )
        right_dark_mask = _build_under_eye_mask(
            landmarks, right_lower_idx, (h, w), offset=offset
        )
        result = _frequency_separation_lighten(result, left_dark_mask, dark_strength)
        result = _frequency_separation_lighten(result, right_dark_mask, dark_strength)

    if eye_vals.brightness > 0 and iris_regions:
        bright_strength = min(1.0, eye_vals.brightness / 100.0)
        for idx, region in enumerate(iris_regions):
            if not np.any(region.mask):
                continue
            result = _apply_iris_donut_brightening(
                result,
                region,
                bright_strength,
            )
            catch_angle = -55.0 if idx == 0 else 55.0
            result = _add_catch_light_precise(
                result,
                region.center,
                region.radius,
                catch_angle,
                bright_strength * 0.8,
            )

    if eye_vals.eyelid != 0:
        lid_strength = _ease_strength(eye_vals.eyelid, 1.25)
        if lid_strength != 0:
            vectors: List[Tuple[Tuple[float, float], float, Tuple[float, float]]] = []
            shift = face_h * 0.012 * lid_strength
            for idx in left_upper_idx + right_upper_idx:
                anchor = tuple(landmarks[idx])
                vectors.append((anchor, face_w * 0.04, (0, -shift)))
            result = _vector_warp(result, eye_mask, vectors)

    lens = config.eyeMakeup.lens
    if lens != "none":
        result = _tint_iris_hsv(result, iris_mask, lens)

    if config.eyeMakeup.eyeliner:
        thickness = max(1, int(face_w * 0.008))
        left_liner = _build_eyeliner_mask(
            landmarks, left_upper_idx, (h, w), thickness=thickness
        )
        right_liner = _build_eyeliner_mask(
            landmarks, right_upper_idx, (h, w), thickness=thickness
        )
        liner_mask = cv2.max(left_liner, right_liner)
        lash_overlay_total = np.zeros_like(result)
        lash_mask_total = np.zeros((h, w), dtype=np.uint8)
        for idx_group in (left_upper_idx, right_upper_idx):
            overlay_piece, mask_piece = _render_lash_layer(
                (h, w),
                landmarks,
                idx_group,
                face_w,
            )
            lash_overlay_total = cv2.add(lash_overlay_total, overlay_piece)
            lash_mask_total = cv2.max(lash_mask_total, mask_piece)

        liner_color = np.clip(result.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
        liner_color = cv2.add(liner_color, np.array([10, 10, 25], dtype=np.uint8))
        result = _blend(result, liner_color, liner_mask, alpha=0.85)
        if np.any(lash_mask_total):
            lash_base = np.clip(result.astype(np.float32) * 0.2, 0, 255).astype(np.uint8)
            lash_layer = cv2.add(lash_base, lash_overlay_total)
            result = _blend(result, lash_layer, lash_mask_total, alpha=0.92)

    result = _apply_mouth_pipeline(
        result,
        landmarks,
        mouth_vals,
        config.lipstick,
        face_w,
        face_h,
    )

    return result


class BeautyPipeline:
    def __init__(self) -> None:
        self.analyzer = FaceAnalyzer()

    def apply(self, image: np.ndarray, config: BeautyConfig) -> Tuple[np.ndarray, Optional[FaceMeta]]:
        analysis = self.analyzer.analyze(image)
        result = image.copy()

        result = _apply_skin_pipeline(result, analysis, config)
        result = _apply_face_sculpt(result, analysis, config)
        result = _apply_eye_and_mouth(result, analysis, config)

        return result, self.analyzer.serialize(analysis)

    def analyze(self, image: np.ndarray) -> Optional[FaceMeta]:
        return self.analyzer.serialize(self.analyzer.analyze(image))

    @staticmethod
    def encode_image(image: np.ndarray) -> str:
        success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            raise ValueError("Failed to encode image")
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
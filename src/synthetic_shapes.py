"""
Synthetic binary images for three shape classes: disk, axis-aligned square, equilateral triangle.

Used to train/test without external datasets; you can replace with your own images using the same pipeline.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import cv2
import numpy as np

ShapeName = str


def _rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def _salt_pepper(
    img: np.ndarray, amount: float, rng: random.Random
) -> np.ndarray:
    out = img.copy()
    h, w = out.shape
    n = int(h * w * amount)
    for _ in range(n):
        y, x = rng.randrange(h), rng.randrange(w)
        out[y, x] = 255 if rng.random() < 0.5 else 0
    return out


def make_disk(
    size: int, radius: int, center_jitter: int, rng: random.Random
) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    cx = size // 2 + rng.randint(-center_jitter, center_jitter + 1)
    cy = size // 2 + rng.randint(-center_jitter, center_jitter + 1)
    cv2.circle(img, (cx, cy), radius, 255, -1, lineType=cv2.LINE_AA)
    return (img > 127).astype(np.uint8) * 255


def make_square(
    size: int, side: int, center_jitter: int, rng: random.Random
) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    c = size // 2 + rng.randint(-center_jitter, center_jitter + 1)
    c2 = size // 2 + rng.randint(-center_jitter, center_jitter + 1)
    half = side // 2
    x0, y0 = c - half, c2 - half
    x1, y1 = c + half, c2 + half
    img[y0:y1, x0:x1] = 255
    return (img > 127).astype(np.uint8) * 255


def _equilateral_vertices(
    cx: float, cy: float, r: float, rotation_deg: float
) -> np.ndarray:
    """Three points, rows (x,y)."""
    from math import cos, radians, sin

    pts: List[Tuple[float, float]] = []
    base = rotation_deg
    for k in range(3):
        ang = radians(base + k * 120.0)
        pts.append((cx + r * cos(ang), cy + r * sin(ang)))
    return np.array(pts, dtype=np.float32)


def make_triangle(
    size: int, r: int, center_jitter: int, rng: random.Random
) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    cx = size // 2 + rng.randint(-center_jitter, center_jitter + 1)
    cy = size // 2 + rng.randint(-center_jitter, center_jitter + 1)
    rot = rng.random() * 360.0
    pts = _equilateral_vertices(
        float(cx), float(cy), float(r), rot
    ).reshape((-1, 1, 2))
    cv2.fillConvexPoly(
        img, np.round(pts).astype(np.int32), 255, lineType=cv2.LINE_AA
    )
    return (img > 127).astype(np.uint8) * 255


def generate_dataset(
    n_per_class: int = 40,
    size: int = 256,
    seed: int = 42,
    noise: float = 0.01,
) -> Tuple[List[np.ndarray], List[ShapeName], List[Tuple[str, int]]]:
    """
    Returns (images, labels, meta) where meta is list of (shape_name, sample_index).
    """
    rng = _rng(seed)
    images: List[np.ndarray] = []
    labels: List[ShapeName] = []
    meta: List[Tuple[str, int]] = []
    for cls in ("disk", "square", "triangle"):
        for k in range(n_per_class):
            r = rng.randint(28, min(90, size // 2 - 6))
            jitter = min(12, size // 8)
            if cls == "disk":
                im = make_disk(size, r, jitter, rng)
            elif cls == "square":
                side = 2 * r
                im = make_square(size, side, jitter, rng)
            else:
                im = make_triangle(size, r, jitter, rng)
            if noise > 0:
                im = _salt_pepper(im, noise, rng)
            images.append(im)
            labels.append(cls)
            meta.append((cls, k))
    return images, labels, meta

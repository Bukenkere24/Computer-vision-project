"""
Freeman chain code (8-direction) from a closed boundary.

BCS613B Module-5: boundary following and chain codes.
Consecutive OpenCV boundary pixels are decomposed into 8-connected unit
steps, then each step maps to a direction code 0..7.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2
import numpy as np

# (dx, dy) -> 0..7: E, NE, N, NW, W, SW, S, SE
_STEP_TO_CODE: dict[Tuple[int, int], int] = {
    (1, 0): 0,
    (1, -1): 1,
    (0, -1): 2,
    (-1, -1): 3,
    (-1, 0): 4,
    (-1, 1): 5,
    (0, 1): 6,
    (1, 1): 7,
}


def _decompose_step(dx: int, dy: int) -> List[Tuple[int, int]]:
    """Break one grid step (dx,dy) into a list of 8-neighbor unit moves."""
    out: List[Tuple[int, int]] = []
    while dx != 0 or dy != 0:
        if dx != 0 and dy != 0:
            sx = 1 if dx > 0 else -1
            sy = 1 if dy > 0 else -1
        elif dx != 0:
            sx, sy = (1 if dx > 0 else -1), 0
        else:
            sx, sy = 0, (1 if dy > 0 else -1)
        out.append((sx, sy))
        dx -= sx
        dy -= sy
    return out


def contour_to_chain_code(contour: np.ndarray) -> np.ndarray:
    """
    Build Freeman chain code from an OpenCV contour (Nx1x2 or Nx2).
    """
    if contour is None or len(contour) < 3:
        return np.empty(0, dtype=np.int32)
    pts = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
    if np.array_equal(pts[0], pts[-1]):
        pts = pts[:-1]
    n = len(pts)
    codes: List[int] = []
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        dx, dy = int(b[0] - a[0]), int(b[1] - a[1])
        for sx, sy in _decompose_step(dx, dy):
            key = (int(sx), int(sy))
            if key not in _STEP_TO_CODE:
                raise ValueError(f"Invalid unit step {key}")
            codes.append(_STEP_TO_CODE[key])
    return np.asarray(codes, dtype=np.int32)


def chain_code_histogram(chain: np.ndarray) -> np.ndarray:
    """
    8-bin normalized histogram of chain codes (feature vector for classification).
    """
    h = np.zeros(8, dtype=np.float64)
    if len(chain) == 0:
        h[0] = 1.0
        return h
    for d in chain:
        di = int(d)
        if 0 <= di < 8:
            h[di] += 1.0
    s = h.sum()
    h /= s if s > 0 else 1.0
    return h


def largest_contour_from_mask(mask: np.ndarray) -> np.ndarray | None:
    """Binary uint8 mask {0,255} -> largest external contour or None."""
    m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

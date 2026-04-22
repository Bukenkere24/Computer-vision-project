"""
HSV color thresholding and morphological post-processing (Module-4, Module-3/5).

Handy for a live demo: isolate a solid-colored paper shape from a web camera or image.
"""

from __future__ import annotations

import cv2
import numpy as np

from .morphology_ops import clean_shape_mask


def segment_hsv_range(
    bgr: np.ndarray,
    lower: tuple,
    upper: tuple,
) -> np.ndarray:
    """
    lower, upper: (H,S,V) in OpenCV ranges — H 0-179, S and V 0-255.
    Returns binary uint8 0/255 foreground mask.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(lower, dtype=np.uint8)
    hi = np.array(upper, dtype=np.uint8)
    m = cv2.inRange(hsv, lo, hi)
    return clean_shape_mask(m)

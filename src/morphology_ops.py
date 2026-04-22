"""
Morphological cleaning of binary masks (Module-5: erosion, dilation, opening, closing).
"""

from __future__ import annotations

import cv2
import numpy as np


def clean_shape_mask(
    binary_255: np.ndarray,
    open_ksize: int = 3,
    close_ksize: int = 5,
) -> np.ndarray:
    """
    Remove speckles (opening) then fill small gaps (closing).
    Input: uint8 image with foreground 255, background 0.
    """
    m = (binary_255 > 127).astype(np.uint8) * 255
    if open_ksize >= 3:
        k1 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k1)
    if close_ksize >= 3:
        k2 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2)
    return m

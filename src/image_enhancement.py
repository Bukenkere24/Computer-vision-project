"""
Image enhancement and denoising (Syllabus ABL: point ops and spatial filters).

For presentation: connect Module-1 (linear filtering) and Activity-Based Learning
on point processing and median/Gaussian filtering.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_gray_u8(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr.astype(np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def histogram_equalize(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray.astype(np.uint8))


def median_denoise(gray: np.ndarray, k: int = 5) -> np.ndarray:
    k2 = k if k % 2 == 1 else k + 1
    return cv2.medianBlur(gray, k2)


def gaussian_smooth(gray: np.ndarray, k: int = 5) -> np.ndarray:
    k2 = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(gray, (k2, k2), 0)

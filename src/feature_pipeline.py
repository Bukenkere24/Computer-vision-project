"""
Binary image -> morphological mask -> largest contour -> chain code histogram.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .chain_code import (
    chain_code_histogram,
    contour_to_chain_code,
    largest_contour_from_mask,
)
from .morphology_ops import clean_shape_mask


@dataclass
class FeatureResult:
    feature: np.ndarray
    mask: np.ndarray
    chain_len: int
    ok: bool
    message: str = ""
    circularity: float = 0.0


def binary_image_to_feature(
    binary_255: np.ndarray,
) -> FeatureResult:
    """
    End-to-end feature for one binary object image (foreground 255).
    """
    m = clean_shape_mask(binary_255)
    c = largest_contour_from_mask(m)
    if c is None or len(c) < 3:
        z = np.zeros(9, dtype=np.float64)
        z[0] = 1.0
        return FeatureResult(
            feature=z, mask=m, chain_len=0, ok=False, message="no contour"
        )
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    circ = (4.0 * np.pi * max(area, 1.0)) / (perim * perim + 1e-6)
    circ = float(np.clip(circ, 0.0, 1.0))

    chain = contour_to_chain_code(c)
    h = chain_code_histogram(chain)
    # 8-bin chain code histogram + boundary circularity (Module-5: shape from contour)
    feat = np.hstack([h, np.array([circ], dtype=np.float64)])
    return FeatureResult(
        feature=feat,
        mask=m,
        chain_len=int(len(chain)),
        ok=True,
        message="ok",
        circularity=circ,
    )


def features_from_images(
    images: list,
) -> Tuple[np.ndarray, list[str]]:
    feats = []
    msgs = []
    for im in images:
        r = binary_image_to_feature(im)
        feats.append(r.feature)
        msgs.append(r.message)
    return np.vstack(feats), msgs


def feature_details_from_images(images: list) -> list[FeatureResult]:
    return [binary_image_to_feature(im) for im in images]

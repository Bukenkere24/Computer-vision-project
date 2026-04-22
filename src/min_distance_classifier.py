"""
Minimum distance to class prototypes (Module-5, textbook: prototype matching).
Each class is represented by the mean feature vector of its training samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class MinimumDistanceClassifier:
    class_names: List[str]
    prototypes: np.ndarray  # shape (C, D)
    _name_to_index: dict

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: List[str],
    ) -> "MinimumDistanceClassifier":
        """
        features: (N, D), labels: length N
        """
        names = sorted(set(labels))
        name_to_i = {n: i for i, n in enumerate(names)}
        dim = features.shape[1]
        proto = np.zeros((len(names), dim), dtype=np.float64)
        counts = np.zeros(len(names), dtype=np.int32)
        for i, y in enumerate(labels):
            j = name_to_i[y]
            proto[j] += features[i]
            counts[j] += 1
        for j in range(len(names)):
            if counts[j] > 0:
                proto[j] /= counts[j]
        return cls(class_names=names, prototypes=proto, _name_to_index=name_to_i)

    def predict(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return predicted label index, class name per row, and distance to chosen prototype.
        features: (N, D)
        """
        n = features.shape[0]
        # distances (N, C)
        d2 = (
            (features[:, None, :] - self.prototypes[None, :, :]) ** 2
        ).sum(axis=2)
        idx = d2.argmin(axis=1)
        dist = np.sqrt(d2[np.arange(n), idx])
        pred_names = np.array([self.class_names[i] for i in idx], dtype=object)
        return idx, pred_names, dist

    def score(self, features: np.ndarray, labels: List[str]) -> float:
        _, pred, _ = self.predict(features)
        y = np.array(labels, dtype=object)
        return float((pred == y).mean())

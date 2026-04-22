"""
Optional demo: histogram equalization and median filter on a synthetic noisy image
(Syllabus: point operations + spatial filtering for enhancement/restoration).

Run: py demo_enhancement.py
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import random

from src.image_enhancement import (
    gaussian_smooth,
    histogram_equalize,
    median_denoise,
    to_gray_u8,
)
from src.synthetic_shapes import make_disk


def main() -> None:
    rng = random.Random(0)
    size = 200
    base = make_disk(size, 50, 6, rng)
    gray = base
    # salt-and-pepper on foreground region
    sp = gray.copy()
    h, w = sp.shape
    n = h * w // 80
    for _ in range(n):
        y, x = rng.randrange(h), rng.randrange(w)
        if sp[y, x] > 0:
            sp[y, x] = 0 if rng.random() < 0.5 else 255

    g = to_gray_u8(sp)
    eq = histogram_equalize(g)
    med = median_denoise(g, 5)
    gsmooth = gaussian_smooth(g, 5)

    out = os.path.join(_ROOT, "output", "enhancement_demo.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(g, cmap="gray")
    ax[0].set_title("noisy input")
    ax[1].imshow(eq, cmap="gray")
    ax[1].set_title("histogram equalize")
    ax[2].imshow(med, cmap="gray")
    ax[2].set_title("median 5x5")
    ax[3].imshow(gsmooth, cmap="gray")
    ax[3].set_title("Gaussian smooth")
    for a in ax:
        a.axis("off")
    fig.suptitle("BCS613B: enhancement and denoising (ABL)")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    print("Wrote", out)


if __name__ == "__main__":
    main()

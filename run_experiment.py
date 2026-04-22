"""
BCS613B project: shape recognition with morphology, chain codes, and minimum distance classification.

Run: py run_experiment.py
"""

from __future__ import annotations

import argparse
import os
import sys

# Project root on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from src.feature_pipeline import features_from_images
from src.min_distance_classifier import MinimumDistanceClassifier
from src.synthetic_shapes import generate_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=30, help="samples per class (train)")
    p.add_argument("--test", type=int, default=10, help="samples per class (test)")
    p.add_argument("--size", type=int, default=256, help="image side length")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--noise", type=float, default=0.012, help="salt & pepper rate")
    p.add_argument(
        "--out", type=str, default="", help="optional path to save a summary figure"
    )
    args = p.parse_args()

    n_tr, n_te = args.train, args.test
    total = n_tr + n_te
    images, labels, _ = generate_dataset(
        n_per_class=total,
        size=args.size,
        seed=args.seed,
        noise=args.noise,
    )

    def take_class(name: str) -> slice:
        # order from generate_dataset: all disk, then square, then triangle
        i = ("disk", "square", "triangle").index(name)
        lo = i * total
        return slice(lo, lo + total)

    tr_img, tr_y, te_img, te_y = [], [], [], []
    for name in ("disk", "square", "triangle"):
        sl = take_class(name)
        part = images[sl]
        part_y = labels[sl]
        tr_img.extend(part[:n_tr])
        tr_y.extend(part_y[:n_tr])
        te_img.extend(part[n_tr:])
        te_y.extend(part_y[n_tr:])

    Xtr, mtr = features_from_images(tr_img)
    Xte, mte = features_from_images(te_img)
    for i, s in enumerate(mtr + mte):
        if "no contour" in s and s:
            print("warning: failed feature on index", i, s, file=sys.stderr)

    clf = MinimumDistanceClassifier.fit(Xtr, tr_y)
    acc = clf.score(Xte, te_y)
    print("BCS613B: Morphology + chain code + minimum distance")
    print(f"  Train / test per class: {n_tr} / {n_te}  (noise={args.noise})")
    print(f"  Test accuracy: {100.0 * acc:.1f}%")

    _, pred, dists = clf.predict(Xte)
    for i in range(len(te_y)):
        mark = "OK" if pred[i] == te_y[i] else "X "
        print(f"  [{i:02d}] true={te_y[i]:7s} pred={str(pred[i]):7s} {mark}  d={dists[i]:.4f}")

    if args.out:
        _save_figure(
            tr_img, te_img, te_y, pred, acc, args.out
        )
        print("Wrote", args.out)


def _save_figure(
    tr_img, te_img, te_y, pred, acc, path: str
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for i, ax in enumerate(axes[0]):
        if i < min(3, len(tr_img)):
            ax.imshow(tr_img[i], cmap="gray")
            ax.set_title("train sample")
        ax.axis("off")
    for j, ax in enumerate(axes[1]):
        idx = j
        if idx < len(te_img):
            ax.imshow(te_img[idx], cmap="gray")
            ok = pred[idx] == te_y[idx]
            c = "green" if ok else "red"
            ax.set_title(f"test: {te_y[idx]} / pred {pred[idx]}", color=c)
        ax.axis("off")
    fig.suptitle(
        f"BCS613B: chain-code histogram + min-distance — test acc {100*acc:.0f}%"
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

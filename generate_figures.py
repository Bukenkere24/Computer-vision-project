"""
Build presentation graphs for the BCS613B project (saved under ./figures/).

Run: py generate_figures.py
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from src.feature_pipeline import feature_details_from_images, features_from_images
from src.min_distance_classifier import MinimumDistanceClassifier
from src.synthetic_shapes import generate_dataset


def _split_data(n_tr: int, n_te: int, size: int, seed: int, noise: float):
    total = n_tr + n_te
    images, labels, _ = generate_dataset(
        n_per_class=total,
        size=size,
        seed=seed,
        noise=noise,
    )
    tr_img, tr_y, te_img, te_y = [], [], [], []
    for name in ("disk", "square", "triangle"):
        i = ("disk", "square", "triangle").index(name)
        lo = i * total
        part, part_y = images[lo : lo + total], labels[lo : lo + total]
        tr_img.extend(part[:n_tr])
        tr_y.extend(part_y[:n_tr])
        te_img.extend(part[n_tr:])
        te_y.extend(part_y[n_tr:])
    return tr_img, tr_y, te_img, te_y


def _fig_pipeline(path: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        (0.15, "Input\n(binary image)"),
        (1.9, "Morphology\nopen + close"),
        (3.65, "Find\noutline"),
        (5.4, "Chain code\n(8 directions)"),
        (7.15, "Features\nhistogram + roundness"),
        (8.9, "Pick closest\nclass average"),
    ]
    w, h = 1.35, 0.55
    y0 = 0.22
    for k, (x, label) in enumerate(steps):
        box = FancyBboxPatch(
            (x, y0),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="#e8f4f8" if k % 2 == 0 else "#fff4e6",
            edgecolor="#333",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y0 + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="medium",
        )
        if k < len(steps) - 1:
            arr = FancyArrowPatch(
                (x + w + 0.02, y0 + h / 2),
                (steps[k + 1][0] - 0.02, y0 + h / 2),
                arrowstyle="Simple,tail_width=0.5,head_width=8,head_length=8",
                color="#444",
                mutation_scale=1,
            )
            ax.add_patch(arr)

    ax.set_title(
        "Plain-English pipeline: from a shape picture to a predicted label",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    fig.text(
        0.5,
        0.06,
        "We measure the outline, turn it into numbers, then match those numbers to stored averages for disk / square / triangle.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#333",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _fig_prototypes(
    clf: MinimumDistanceClassifier, path: str
) -> None:
    import matplotlib.pyplot as plt

    names = clf.class_names
    P = clf.prototypes
    h8 = P[:, :8]
    circ = P[:, 8]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(8)
    w = 0.25
    for i, name in enumerate(names):
        ax0.bar(
            x + (i - 1) * w,
            h8[i],
            w,
            label=name,
        )
    ax0.set_xlabel("Chain direction (0 = east, … 7 = south-east on the grid)")
    ax0.set_ylabel("Average share of outline")
    ax0.set_title("Learned “fingerprints”: chain-code part of each class prototype")
    ax0.set_xticks(x + 0.0, [str(i) for i in range(8)])
    ax0.legend(title="Class average")
    ax0.grid(axis="y", alpha=0.3)

    ax1.barh(names, circ, color=["#4a90d9", "#e6a23c", "#67c27a"])
    ax1.set_xlabel("Roundness (4πA / P²)")
    ax1.set_xlim(0, 1.05)
    ax1.set_title("Extra number: how “circle-like” the outline is")
    for j, v in enumerate(circ):
        ax1.text(v + 0.02, j, f"{v:.2f}", va="center", fontsize=10)
    fig.suptitle("What the program stores for each shape (training averages)", y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _fig_confusion(te_y, pred, names, path: str) -> None:
    import matplotlib.pyplot as plt

    n = len(names)
    C = np.zeros((n, n), dtype=int)
    idx = {a: j for j, a in enumerate(names)}
    for t, p in zip(te_y, pred):
        C[idx[str(t)]][idx[str(p)]] += 1
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=max(C.max(), 1))
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                int(C[i, j]),
                ha="center",
                va="center",
                color="white" if C[i, j] > C.max() / 2 else "black",
                fontsize=14,
                fontweight="bold",
            )
    ax.set_xticks(np.arange(n), names)
    ax.set_yticks(np.arange(n), names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _fig_circularity(tr_y, details, path: str) -> None:
    import matplotlib.pyplot as plt

    order = ("disk", "square", "triangle")
    data = {k: [] for k in order}
    for y, d in zip(tr_y, details):
        if d.ok:
            data[y].append(d.circularity)
    fig, ax = plt.subplots(figsize=(7, 4))
    pos = [np.array(data[k], dtype=float) for k in order]
    parts = ax.violinplot(
        [p for p in pos if len(p) > 0],
        positions=range(len(order)),
        showmeans=True,
        showmedians=True,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Roundness (4πA / P²)")
    ax.set_xlabel("True shape (training set)")
    ax.set_title("Why roundness helps: disks are rounder than squares and triangles")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _fig_samples(tr_img, te_img, te_y, pred, acc, path: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.5))
    for i, ax in enumerate(axes[0]):
        if i < min(3, len(tr_img)):
            ax.imshow(tr_img[i], cmap="gray")
            ax.set_title("Training example", fontsize=10)
        ax.axis("off")
    for j, ax in enumerate(axes[1]):
        if j < len(te_img):
            ax.imshow(te_img[j], cmap="gray")
            ok = pred[j] == te_y[j]
            c = "#1a7f37" if ok else "#c41e1e"
            ax.set_title(f"True: {te_y[j]}  |  Guess: {pred[j]}", color=c, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"Sample images (test accuracy {100 * acc:.0f}%)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    n_tr, n_te, size, seed, noise = 30, 10, 256, 7, 0.012
    tr_img, tr_y, te_img, te_y = _split_data(n_tr, n_te, size, seed, noise)
    Xtr, _ = features_from_images(tr_img)
    Xte, _ = features_from_images(te_img)
    clf = MinimumDistanceClassifier.fit(Xtr, tr_y)
    acc = clf.score(Xte, te_y)
    _, pred, _ = clf.predict(Xte)
    details = feature_details_from_images(tr_img)

    out = os.path.join(_ROOT, "figures")
    os.makedirs(out, exist_ok=True)

    _fig_pipeline(os.path.join(out, "01_pipeline_overview.png"))
    _fig_prototypes(clf, os.path.join(out, "02_prototype_fingerprints.png"))
    _fig_confusion(
        te_y, pred, clf.class_names, os.path.join(out, "03_confusion_matrix.png")
    )
    _fig_circularity(tr_y, details, os.path.join(out, "04_roundness_by_class.png"))
    _fig_samples(tr_img, te_img, te_y, pred, acc, os.path.join(out, "05_sample_results.png"))

    print("Wrote figures to:", out)
    for name in sorted(os.listdir(out)):
        if name.endswith(".png"):
            print("  ", name)


if __name__ == "__main__":
    main()

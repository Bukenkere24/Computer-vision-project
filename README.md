# BCS613B Computer Vision — Course Project

**Topic (from VTU BCS613B syllabus):** Module 5 — *morphological image processing*, *boundary following and chain codes*, and *pattern classification by prototype matching (minimum distance classifier)*, applied to **simple shape recognition**. Supporting demos cover **image enhancement** (ABL: point operations and spatial filters) and **HSV color segmentation** (Module 4), as listed in the practical component of the syllabus.

## What this project does

1. **Binary object mask** — synthetic solid shapes (disk, square, triangle) with light salt-and-pepper noise; you can swap in your own binary images.
2. **Morphology** — `opening` then `closing` to clean the mask (erosion / dilation as in the syllabus).
3. **Boundary** — largest external contour; **Freeman 8-direction chain code**; **8-bin normalized histogram** = main feature vector.
4. **Extra boundary descriptor** — **circularity** \(4\pi A / P^2\) from the same contour to separate similar chain-code patterns (e.g. noisy disk vs. triangle) while staying in the “boundary / feature” theme of Module 5.
5. **Minimum distance classifier** — one **prototype** per class = **mean feature vector** over training samples; test sample is assigned to the class with **smallest Euclidean distance** to its prototype.

## How to run

**Everything at once (install, experiment, all graphs, enhancement demo, tests):**

```text
py run_all.py
```

If dependencies are already installed:

```text
py run_all.py --skip-install
```

### Run parts separately

```text
py -m pip install -r requirements.txt
py run_experiment.py
```

**Graphs for slides (saved under `figures/`, see `figures/ABOUT.txt`):**

```text
py generate_figures.py
py demo_enhancement.py
```

Optional: save a one-off summary of train/test samples:

```text
py run_experiment.py --out output/summary.png
```

Optional (ABL — enhancement & filtering; also updates `figures/06_enhancement_filters.png`):

```text
py demo_enhancement.py
```

## Suggested 5-minute class presentation

1. **Syllabus link** — Module 5: morphology, chain codes, minimum distance; ABL: enhancement and segmentation.
2. **Pipeline diagram** — binary image → open/close → contour → chain code → histogram (+ circularity) → nearest prototype.
3. **Live or recorded run** of `run_experiment.py` and show `output/summary.png`.
4. **One slide** on **minimum distance**: prototypes = class means; decision = argmin distance.
5. **Limitation** — chain-code histogram is not rotation-invariant; circularity helps for these three classes but real photos may need HSV segmentation (`src/color_segmentation.py`) and tuning.

## Repository layout

| Path | Role |
|------|------|
| `src/chain_code.py` | Freeman chain code + histogram |
| `src/morphology_ops.py` | Opening / closing |
| `src/min_distance_classifier.py` | Train prototypes, predict |
| `src/synthetic_shapes.py` | Training data (disk / square / triangle) |
| `src/feature_pipeline.py` | Full feature extraction |
| `src/image_enhancement.py` | Histogram equalization, median, Gaussian |
| `src/color_segmentation.py` | HSV range + morphology |
| `run_experiment.py` | Main experiment + optional figure |
| `demo_enhancement.py` | Enhancement demo image |

## References (as in syllabus)

- R. Szeliski, *Computer Vision: Algorithms and Applications* (2nd ed.).
- Gonzalez & Woods, *Digital Image Processing* (4th ed.) — morphological and shape analysis chapters.

## Course

**BCS613B — Computer Vision** (VTU, CSE program).

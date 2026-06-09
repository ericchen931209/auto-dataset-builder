---
license: mit
task_categories:
  - object-detection
tags:
  - neural-dqs
  - dataset-quality
  - computer-vision
  - yolo
  - clip
  - map-prediction
pretty_name: Neural DQS Benchmark — COCO128 Degradation Variants
size_categories:
  - n<1K
---

# Neural DQS Benchmark

**96-variant COCO128 degradation benchmark for training and evaluating Neural Dataset Quality Score (Neural DQS) models.**

This dataset contains feature vectors and ground-truth mAP@0.5 values for 96 systematically degraded versions of COCO128, used to validate the hypothesis:

> **DQS(D) ↑ ⟹ mAP(YOLO trained on D) ↑**
>
> Result: CV Pearson **r = 0.929** (n=96, p<0.001)

---

## Dataset Description

Each row represents one dataset variant. Features are extracted from the image/annotation set; `map50` is the ground truth obtained by training YOLOv11n for 15 epochs.

### Features

| Column | Description | Range |
|--------|-------------|-------|
| `annotation_quality` (AQ) | `0.6 × completeness + 0.4 × bbox geometry` | [0, 1] |
| `sharpness` (IQ) | `√(blur_score × noise_cleanliness)` | [0, 1] |
| `clip_diversity` (CD) | Mean pairwise cosine distance in CLIP ViT-B/32 space | [0, 1] |
| `lighting_diversity` (LD) | Normalized brightness entropy (3 buckets) | [0, 1] |
| `pose_diversity` (PD) | Normalized aspect-ratio entropy | [0, 1] |
| `class_balance` (CB) | `1 − Gini coefficient` | [0, 1] |
| `map50` | mAP@0.5 — YOLOv11n trained 15 epochs (**target variable**) | [0, 1] |
| `map50_95` | mAP@0.5:0.95 | [0, 1] |

### Degradation Categories (10 types)

| Category | Variants | Description |
|----------|----------|-------------|
| Baseline | 1 | Original COCO128 |
| Blur | 20 | Gaussian blur, kernel 3–61 |
| Noise | 8 | Gaussian noise σ=2–100 |
| Brightness | 15 | Factor 0.05–2.0 |
| Label missing | 13 | 10%–90% of label files blanked |
| Label noise | 3 | Bbox cx/cy shifted ±3%–20% |
| Combined | 9 | blur+dark, noise+dark, noise+blur (3 severities each) |
| Other | 27 | Dense sweeps across blur/brightness ranges |

---

## Key Result

Trained with `Ridge(α=1.0) + PolynomialFeatures(degree=2)`:

| Metric | Value |
|--------|-------|
| CV Pearson r (k=5) | **0.929** |
| CV R² | 0.854 |
| Train Pearson r | 0.970 |

**Top predictors:**
- CD (CLIP Diversity): r = 0.892
- IQ (Image Quality): r = 0.661

---

## Usage

```python
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, KFold
import numpy as np

df = pd.read_csv("hf://datasets/EricChenWei/neural-dqs-benchmark/dqs_training_data_v5.csv")

FEATURES = ["annotation_quality", "sharpness", "clip_diversity",
            "lighting_diversity", "pose_diversity", "class_balance"]

X = df[FEATURES].values
y = df["map50"].values

model = Pipeline([
    ("scaler", StandardScaler()),
    ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
    ("ridge",  Ridge(alpha=1.0)),
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_cv = cross_val_predict(model, X, y, cv=cv)
r = np.corrcoef(y, y_cv)[0, 1]
print(f"CV Pearson r = {r:.4f}")  # → ~0.929
```

---

## Related

- **GitHub**: [ericchen931209/auto-dataset-builder](https://github.com/ericchen931209/auto-dataset-builder)
- **Model**: [EricChenWei/neural-dqs](https://huggingface.co/EricChenWei/neural-dqs)

## Citation

```bibtex
@software{chen2026adb,
  author  = {Chen, Yu-Wei},
  title   = {Auto Dataset Builder: An LLM-Assisted Framework for
             Automatic Dataset Construction with Neural Dataset Quality Scoring},
  year    = {2026},
  url     = {https://github.com/ericchen931209/auto-dataset-builder},
  license = {MIT}
}
```

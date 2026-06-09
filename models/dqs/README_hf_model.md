---
license: mit
tags:
  - neural-dqs
  - dataset-quality
  - computer-vision
  - regression
  - sklearn
  - clip
library_name: sklearn
---

# Neural DQS — Dataset Quality Score Predictor

**Predicts post-training mAP@0.5 from 6 dataset-level features, before training any model.**

CV Pearson **r = 0.929** (n=96, p<0.001) on the [Neural DQS Benchmark](https://huggingface.co/datasets/EricChenWei/neural-dqs-benchmark).

---

## Model Description

A `Ridge(α=1.0)` regression with `StandardScaler + PolynomialFeatures(degree=2)` operating on a 6-dimensional feature vector extracted from a computer vision dataset.

### Feature Vector: f(D) ∈ ℝ⁶

| Feature | Symbol | Description |
|---------|--------|-------------|
| Annotation Quality | AQ | `0.6 × completeness + 0.4 × bbox geometry` |
| Image Quality | IQ | `√(blur_score × noise_cleanliness)` |
| CLIP Diversity | CD | Mean pairwise cosine distance (ViT-B/32) |
| Lighting Diversity | LD | Normalized brightness entropy |
| Pose Diversity | PD | Normalized aspect-ratio entropy |
| Class Balance | CB | `1 − Gini coefficient` |

### Architecture

```
f(D) ∈ ℝ⁶
  → StandardScaler
  → PolynomialFeatures(degree=2)  → ℝ²⁸
  → Ridge(α=1.0)
  → predicted mAP@0.5
```

---

## Performance

| Metric | Value |
|--------|-------|
| CV Pearson r (k=5) | **0.929** |
| CV R² | 0.854 |
| Train Pearson r | 0.970 |
| Training samples | 96 |

**SHAP feature importance (mean \|φ\|):**
- CD (CLIP Diversity): 0.0765 ← strongest
- IQ (Image Quality): 0.0211
- AQ (Annotation Quality): 0.0142

---

## Usage

```python
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

model_path = hf_hub_download("EricChenWei/neural-dqs", "neural_dqs_model.pkl")
model = joblib.load(model_path)

# Feature vector: [AQ, IQ, CD, LD, PD, CB]
features = np.array([[0.80, 0.46, 0.49, 0.46, 0.83, 0.92]])
predicted_map50 = model.predict(features)[0]
print(f"Predicted mAP@0.5 = {predicted_map50:.4f}")
```

### Extract features with Auto Dataset Builder

```python
from models.dqs.feature_extractor import extract_features

feats = extract_features(image_dir="path/to/images", label_dir="path/to/labels")
f = [feats.annotation_quality, feats.sharpness, feats.clip_diversity,
     feats.lighting_diversity, feats.pose_diversity, feats.class_balance]

predicted_map50 = model.predict([f])[0]
```

---

## Training Data

[EricChenWei/neural-dqs-benchmark](https://huggingface.co/datasets/EricChenWei/neural-dqs-benchmark) — 96-variant COCO128 degradation benchmark.

## Related

- **GitHub**: [ericchen931209/auto-dataset-builder](https://github.com/ericchen931209/auto-dataset-builder)

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

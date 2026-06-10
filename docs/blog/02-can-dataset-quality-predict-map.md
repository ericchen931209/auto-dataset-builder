# Can Dataset Quality Predict mAP — Before You Train?

*Part of the [Auto Dataset Builder](https://github.com/ericchen931209/auto-dataset-builder) project.*

Every computer vision practitioner has asked some version of this question:
**"Is my dataset good enough?"** Usually the only answer is to train a model,
wait, and look at the mAP. That loop costs hours or days per iteration.

What if you could predict the mAP *before* training — just by looking at six
numbers extracted from your dataset?

## The hypothesis

I built **Neural DQS (Dataset Quality Score)**, a small regression model that
maps a 6-dimensional feature vector describing a dataset to a predicted
mAP@0.5:

```
f(D) = [AQ, IQ, CD, LD, PD, CB] ∈ ℝ⁶

  AQ — Annotation Quality   (label completeness + bbox geometry)
  IQ — Image Quality        (√(blur_score × noise_cleanliness))
  CD — CLIP Diversity       (mean pairwise cosine distance, CLIP ViT-B/32)
  LD — Lighting Diversity   (brightness entropy)
  PD — Pose Diversity       (aspect-ratio entropy)
  CB — Class Balance        (1 − Gini coefficient)
```

The hypothesis: **DQS(D) ↑ ⟹ mAP(YOLO trained on D) ↑**

## The experiment

To test this without needing dozens of different datasets, I generated **96
controlled degradation variants of COCO128** — blurred, noised, darkened,
with missing labels, noisy bounding boxes, and combinations thereof. For each
variant I:

1. Extracted the 6 DQS features
2. Trained YOLOv11n from scratch (CPU only — Intel Core Ultra 9 285H, no GPU)
3. Recorded the actual mAP@0.5

Then I fit `Ridge(α=1.0) + PolynomialFeatures(degree=2)` on the 96 (feature,
mAP) pairs and evaluated with 5-fold cross-validation.

## Result

**CV Pearson r = 0.929 (p < 0.001)**, CV R² = 0.854.

![Neural DQS predicted mAP vs. actual mAP@0.5](../dqs_scatter.png)

To check this wasn't an artifact of under-training (15 epochs per variant,
chosen for throughput), I re-ran the entire 96-variant benchmark at 50
epochs. The correlation barely moved: **r = 0.922** (Δr = 0.007). The
DQS↔mAP relationship is **epoch-invariant** — it reflects something about the
data, not an artifact of training duration.

## Which feature matters most?

A leave-one-out ablation answers this directly:

| Configuration | CV r | Δr |
|---|---|---|
| **Full model (6 features)** | **0.929** | — |
| w/o CLIP Diversity | 0.679 | **−0.250** |
| w/o Lighting Diversity | 0.923 | −0.006 |
| w/o Pose Diversity | 0.925 | −0.004 |
| w/o Image Quality | 0.925 | −0.004 |
| w/o Annotation Quality | 0.928 | −0.001 |
| w/o Class Balance | 0.938 | +0.009 |

Removing **CLIP Diversity** alone drops r from 0.929 to 0.679 — a quarter of
the model's predictive power lives in a single feature: how semantically
diverse the images are in CLIP embedding space.

SHAP confirms it independently:

![SHAP feature importance for Neural DQS](../dqs_shap.png)

CLIP Diversity's mean |SHAP value| (0.0765) is **3.6× larger** than the
runner-up (Image Quality, 0.0211). Interestingly, Annotation Quality ranks
third in SHAP despite a near-zero linear correlation (r = −0.042) — it
contributes through non-linear interactions (the polynomial terms), likely
penalizing datasets only once label completeness drops below some threshold.

## Takeaway

A 6-number summary of a dataset — computable in seconds, without training
anything — predicts the mAP you'll get from a full YOLO training run with
r = 0.929. The single most informative number is how diverse your images are
in CLIP embedding space, *not* how clean your labels or images are.

This is the quality-scoring core of **Auto Dataset Builder (ADB)**, an
end-to-end framework that builds and scores datasets from a natural-language
description. Code, the trained model, and the full benchmark dataset are
public:

- GitHub: [ericchen931209/auto-dataset-builder](https://github.com/ericchen931209/auto-dataset-builder)
- Model: [🤗 EricChenWei/neural-dqs](https://huggingface.co/EricChenWei/neural-dqs)
- Benchmark: [🤗 EricChenWei/neural-dqs-benchmark](https://huggingface.co/datasets/EricChenWei/neural-dqs-benchmark)

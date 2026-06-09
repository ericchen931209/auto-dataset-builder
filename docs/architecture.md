# 系統架構圖（論文等級）

## Figure 1: ADB System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Auto Dataset Builder (ADB)                         │
└─────────────────────────────────────────────────────────────────────────────┘

  User Input (Natural Language)
  "Build a Taiwan motorcycle detection dataset"
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  [Module 1] Natural Language Parser                                         │
│                                                                             │
│   LLM (Qwen-VL / Gemma)                                                     │
│   Input: Free-form text                                                     │
│   Output: {target, task, region, modality, class_list, min_samples}        │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  [Module 2] Dataset Planning Agent                                          │
│                                                                             │
│   • Keyword expansion via LLM                                               │
│   • Search query generation                                                 │
│   • Annotation schema definition (YOLO / COCO)                             │
│   • Diversity target setting                                                │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  [Module 3] Data Collection                                                 │
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐                      │
│  │  YouTube Crawler      │    │  Image Search Engine  │                     │
│  │  yt-dlp              │    │  Google / Bing API    │                     │
│  │  720P / 1080P        │    │  keyword expansion    │                     │
│  └──────────┬───────────┘    └──────────┬────────────┘                     │
│             │                           │                                   │
│             └──────────┬────────────────┘                                   │
│                        ▼                                                    │
│              Deduplication Filter                                           │
│              imagehash.phash() │ cosine sim > 0.95 → discard               │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  [Module 4] Frame Extraction                                                │
│                                                                             │
│  ┌────────────────────┐    ┌──────────────────────────┐                    │
│  │  Fixed-rate        │    │  Adaptive (SSIM-based)    │                   │
│  │  1 / 2 / 5 FPS     │    │  extract if Δ > threshold │                   │
│  └────────────────────┘    └──────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  [Module 5] Three-Stage Auto Annotation Pipeline                            │
│                                                                             │
│  Stage 1: YOLOv11                                                           │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │  Input: raw frame                                         │              │
│  │  Output: {bbox, class, confidence}                        │              │
│  └──────────────────────────────────────────────────────────┘              │
│                    │ conf > θ_detect (e.g., 0.5)                            │
│                    ▼                                                        │
│  Stage 2: SAM2 Refinement                                                  │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │  Input: bbox prompt                                       │              │
│  │  Output: high-quality segmentation mask                   │              │
│  └──────────────────────────────────────────────────────────┘              │
│                    │                                                        │
│                    ▼                                                        │
│  Stage 3: LLM Verification (Vision LLM)                                    │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │  Query: "Does this crop contain a {class}? Yes/No"       │              │
│  │  Output: verified label │ discard if No                  │              │
│  └──────────────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  [Module 6] Dataset Cleaning                                                │
│                                                                             │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────┐    │
│  │  Blur Detection  │  │  Dark Frame Rm.  │  │  Overexposure Rm.      │   │
│  │  Laplacian var   │  │  mean(px) < 20   │  │  Histogram saturation  │   │
│  │  < threshold     │  │                  │  │  > 98%                 │   │
│  └─────────────────┘  └──────────────────┘  └────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  [Module 7] Neural Dataset Quality Score (Neural DQS)                      │
│                                                                             │
│  Feature Extraction → f(D) ∈ ℝ⁵                                            │
│  Regression Model   → DQS(D) = g(f(D); θ)  ∈ [0, 1]                       │
│  Training Signal    → mAP from trained YOLOv11                             │
│                                                                             │
│  (See docs/dqs-model.md for full formulation)                               │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  [Module 8] Active Learning Loop                                            │
│                                                                             │
│  Train → Infer → Select (conf < 0.5) → Re-annotate → Add → Repeat          │
│                                                                             │
│  Termination: DQS > 0.85  OR  max_iterations reached                       │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  [Module 9] Dataset Export                                                  │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────────┐ │
│  │  YOLO Format  │  │  COCO JSON   │  │  Version Snapshot (Git-like)     │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Figure 2: Three-Stage Annotation Pipeline（論文 Figure 細節）

```
 Input Frame
     │
     ▼
 ┌─────────────────────────────────────────────────────┐
 │ Stage 1: Proposal Generation (YOLOv11)              │
 │                                                     │
 │  ┌──────────┐    conf ≥ θ    ┌──────────────────┐  │
 │  │  YOLOv11 │ ─────────────▶ │  Proposal Queue  │  │
 │  └──────────┘                └────────┬─────────┘  │
 │                                       │             │
 └───────────────────────────────────────┼─────────────┘
                                         │
                                         ▼
 ┌─────────────────────────────────────────────────────┐
 │ Stage 2: Mask Refinement (SAM2)                     │
 │                                                     │
 │  bbox prompt → SAM2 → pixel-level mask              │
 │  IoU(mask, bbox) < 0.5 → flag for review            │
 │                                                     │
 └─────────────────────────────────────────────────────┘
                                         │
                                         ▼
 ┌─────────────────────────────────────────────────────┐
 │ Stage 3: Semantic Verification (Vision LLM)         │
 │                                                     │
 │  crop(frame, bbox) + prompt → LLM                   │
 │                                                     │
 │  Prompt template:                                   │
 │  "You are a dataset annotator. Does the image       │
 │   contain a {class_name}? Answer: Yes or No."       │
 │                                                     │
 │  Yes → accept label                                 │
 │  No  → discard sample                               │
 │  Low confidence → queue for human review            │
 │                                                     │
 └─────────────────────────────────────────────────────┘
```

---

## Figure 3: Active Learning Data Flywheel

```
          ┌──────────────────────────────┐
          │    Initial Dataset D₀        │
          │    (auto-annotated, ~500 img) │
          └───────────────┬──────────────┘
                          │
                          ▼
               ┌──────────────────┐
               │  Train YOLOv11   │
               └────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐         ┌────────────────────────┐
               │  Inference on    │────────▶│  Low-confidence samples │
               │  unlabeled pool  │         │  conf(x) < 0.5          │
               └──────────────────┘         └───────────┬────────────┘
                                                        │
                                                        ▼
                                            ┌───────────────────────┐
                                            │  Re-annotation        │
                                            │  (ADB pipeline)       │
                                            └───────────┬───────────┘
                                                        │
                                                        ▼
               ┌──────────────────┐         ┌──────────────────────┐
               │  DQS > 0.85?     │◀────────│  Augmented Dataset   │
               │  Stop            │         │  D_t+1 = D_t ∪ ΔD    │
               └──────────────────┘         └──────────────────────┘
```

---

## 論文圖注範例（LaTeX caption 格式）

```
Figure 1: Overview of the Auto Dataset Builder (ADB) framework.
          The system takes natural language input and produces a
          YOLO-compatible annotated dataset through a nine-stage
          automated pipeline.

Figure 2: Three-stage annotation pipeline. Stage 1 proposes bounding
          boxes via YOLOv11, Stage 2 refines boundaries using SAM2,
          and Stage 3 verifies semantic correctness with a Vision LLM.

Figure 3: Active learning loop forming a Data Flywheel. The model
          iteratively identifies uncertain samples for re-annotation,
          converging when DQS exceeds the quality threshold θ_q.
```

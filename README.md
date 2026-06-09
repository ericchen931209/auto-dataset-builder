# 🤖 Auto Dataset Builder (ADB)

> **Turn natural language into a training-ready computer vision dataset — automatically.**

[![Tests](https://github.com/ericchen931209/auto-dataset-builder/actions/workflows/test.yml/badge.svg)](https://github.com/ericchen931209/auto-dataset-builder/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-red.svg)](https://github.com/ultralytics/ultralytics)
[![Vue 3](https://img.shields.io/badge/Vue-3-42b883.svg)](https://vuejs.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed.svg)](docker-compose.dev.yml)

---

## What is ADB?

**Auto Dataset Builder** is an end-to-end platform that transforms a single natural language description into a fully annotated, YOLO-compatible computer vision dataset — with no manual labeling required.

```
Input:  "Build a Taiwan motorcycle detection dataset"
Output: Annotated dataset (YOLO format) + quality score + training-ready split
```

The system automates every stage of the dataset pipeline:

| Stage | Module | Technology |
|-------|--------|------------|
| Data collection | YouTube + Image Search | `yt-dlp`, Google Custom Search |
| Frame extraction | Fixed-rate / Scene-change | OpenCV SSIM |
| Auto annotation | Three-stage pipeline | YOLOv11 → SAM2 → Vision LLM |
| Quality filtering | Image cleaning | Laplacian, HSV analysis |
| Quality scoring | **Neural DQS** | MLP + CLIP embeddings |
| Active learning | Data flywheel | Uncertainty sampling |

---

## Key Innovation: Neural Dataset Quality Score (Neural DQS)

ADB introduces a **learnable dataset quality metric** that predicts model mAP *before* training:

```
f(D) = [AQ, DS, LD, PD, CB]  ∈ ℝ⁵

  AQ — Annotation Quality     (bbox consistency)
  DS — Diversity Score        (CLIP embedding entropy)
  LD — Lighting Diversity     (brightness distribution)
  PD — Pose Diversity         (aspect ratio distribution)
  CB — Class Balance          (1 - Gini coefficient)

DQS(D) = MLP(f(D))  →  predicted mAP@0.5
```

Hypothesis: **DQS ↑ ⟹ mAP ↑** (Pearson r > 0.85)

---

## Architecture

```
Natural Language Input
        │
        ▼
┌─────────────────────┐
│ NL Parser (LLM)     │  → {target, task, region, classes}
└─────────────────────┘
        │
        ▼
┌─────────────────────┐     ┌─────────────────────┐
│ YouTube Crawler     │     │ Image Search         │
│ (CC license only)   │     │ (Google Custom API)  │
└─────────┬───────────┘     └──────────┬───────────┘
          └──────────┬─────────────────┘
                     ▼
          Frame Extraction (SSIM adaptive)
                     │
                     ▼
        ┌────────────────────────┐
        │ Three-Stage Annotation  │
        │  1. YOLOv11 proposal    │
        │  2. SAM2 refinement     │
        │  3. Vision LLM verify   │
        └────────────┬───────────┘
                     │
                     ▼
          Image Cleaning (blur/dark/overexposed)
                     │
                     ▼
          Neural DQS Evaluation
                     │
              DQS < threshold?
                ┌────┴────┐
               Yes        No
                │          │
         Active Learning  Export
         (re-annotate)    (YOLO/COCO)
```

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- (Optional) GPU for YOLO / SAM2 inference

### Run with Docker

```bash
git clone https://github.com/YOUR_USERNAME/auto-dataset-builder
cd auto-dataset-builder

cp .env.example .env
# Edit .env to set your Google Search API key (optional)

docker-compose -f docker-compose.dev.yml up -d
```

Services started:

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | FastAPI backend |
| Docs | http://localhost:8000/docs | Swagger UI |
| Flower | http://localhost:5555 | Celery task monitor |

### Create a Dataset via API

```bash
curl -X POST http://localhost:8000/api/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{"query": "Build a Taiwan motorcycle detection dataset"}'
```

### Evaluate Dataset Quality (DQS)

```bash
curl -X POST http://localhost:8000/api/v1/datasets/1/evaluate-dqs
```

Response:

```json
{
  "dataset_id": 1,
  "dqs_score": 0.74,
  "features": {
    "annotation_quality": 0.82,
    "diversity": 0.68,
    "lighting_diversity": 0.71,
    "pose_diversity": 0.55,
    "class_balance": 1.0
  }
}
```

---

## Project Structure

```
auto-dataset-builder/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # FastAPI endpoints
│   │   ├── core/            # Config, settings
│   │   ├── db/              # SQLAlchemy models
│   │   └── main.py
│   └── requirements.txt
├── workers/
│   ├── collector/           # YouTube + image download, dedup
│   ├── extractor/           # Frame extraction (fixed/adaptive)
│   ├── annotator/           # YOLO annotation pipeline
│   └── cleaner/             # Image quality filtering
├── models/
│   └── dqs/                 # Neural DQS feature extraction + MLP
├── docs/
│   ├── architecture.md      # System diagrams (paper level)
│   ├── dqs-model.md         # Mathematical formulation
│   ├── research-design.md   # Hypotheses & experiments
│   ├── roadmap.md           # V0.1 → V1.0 milestones
│   └── paper-draft.md       # Paper outline
└── docker-compose.dev.yml
```

---

## Roadmap

| Version | Status | Feature |
|---------|--------|---------|
| V0.1 | ✅ Done | Project scaffold (FastAPI + Celery + PostgreSQL) |
| V0.2 | ✅ Done | Data collection (YouTube CC-licensed + Image Search + Dedup) |
| V0.3 | ✅ Done | Frame extraction (SSIM adaptive) + YOLO auto annotation |
| V0.4 | ✅ Done | Dataset cleaning (blur / dark / overexposed) |
| V0.5 | ✅ Done | SAM2 bbox refinement + Vision LLM verification |
| V0.6 | ✅ Done | Neural DQS (5-feature MLP + SHAP explainability) |
| V0.7 | ✅ Done | Active learning loop (uncertainty sampling + DQS convergence) |
| V0.8 | ✅ Done | Web dashboard (Vue 3 + Chart.js radar chart) |
| V0.9 | ✅ Done | Version control + YOLO/COCO export + download API |
| V1.0 | ✅ Done | Integration tests + benchmark + production Docker + CI |

---

## Research

This project accompanies the paper:

> **Auto Dataset Builder: An LLM-Assisted Framework for Automatic Dataset Construction with Neural Dataset Quality Scoring**

Core research contributions:
1. End-to-end NL→Dataset pipeline with no manual labeling
2. Three-stage annotation (YOLO + SAM2 + Vision LLM)
3. **Neural DQS**: first learnable predictor of dataset-level mAP
4. Active learning loop terminated by DQS threshold

See [docs/paper-draft.md](docs/paper-draft.md) for the full paper outline and [docs/dqs-model.md](docs/dqs-model.md) for the mathematical formulation.

---

## Citation

If you use ADB in your research, please cite:

```bibtex
@software{chen2026adb,
  author  = {Chen, Wei},
  title   = {Auto Dataset Builder: An LLM-Assisted Framework for
             Automatic Dataset Construction},
  year    = {2026},
  url     = {https://github.com/ericchen931209/auto-dataset-builder},
  license = {MIT}
}
```

Or use the [CITATION.cff](CITATION.cff) file directly — GitHub will render a "Cite this repository" button automatically.

---

## Tests

```bash
# Unit tests (47 tests)
python3 tests/test_all.py
# Results: 47 passed, 0 failed / 47 total ✓

# Integration tests (8 end-to-end tests)
python3 tests/test_integration.py
# Results: 8 passed, 0 failed / 8 total ✓

# Benchmark
python3 tools/benchmark.py --images 100 --outfile results.json
```

CI runs automatically on every push via GitHub Actions.

---

## ⚠️ Copyright Notice

YouTube videos are downloaded for **research and academic use only**.
By default, only **Creative Commons licensed** videos are downloaded (`license_filter="creativecommons"`).

Do **not** redistribute downloaded videos or frames publicly without verifying their licenses.

---

## Contributing

Issues, experiments, and PRs are welcome!
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and areas that need help.

---

## License

MIT — see [LICENSE](LICENSE)

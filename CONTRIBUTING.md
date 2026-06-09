# Contributing to Auto Dataset Builder

Thank you for your interest in contributing! ADB is an active research project and we welcome contributions of all kinds.

## How to Contribute

### Bug Reports & Feature Requests
Open an issue with one of these labels:
- `bug` — something is broken
- `enhancement` — new feature request
- `research` — research direction / experiment idea
- `good first issue` — a great place to start

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes and run tests (`python3 tests/test_all.py`)
4. Commit with a clear message (`feat: add X`, `fix: Y`, `docs: Z`)
5. Open a PR against `main`

### Development Setup

```bash
git clone https://github.com/ericchen931209/auto-dataset-builder
cd auto-dataset-builder
docker-compose -f docker-compose.dev.yml up -d
```

Frontend (Vue 3):
```bash
cd frontend && npm install && npm run dev
```

Run tests:
```bash
python3 tests/test_all.py
```

## Areas Most Needing Help

| Area | Difficulty | Description |
|------|-----------|-------------|
| V0.5 SAM2 Integration | Medium | Wire SAM2 into annotation pipeline |
| V0.5 Vision LLM Verification | Medium | Integrate Qwen-VL / LLaVA for label verification |
| V0.7 Active Learning | Medium | Implement uncertainty sampling loop |
| DQS Training Data | Easy | Contribute (features, mAP) pairs to improve Neural DQS |
| Benchmark Datasets | Easy | Test on VisDrone, COCO subsets, custom domains |
| Unit Tests | Easy | Expand test coverage |
| Documentation | Easy | Improve docs, add examples |

## Research Contributions

If you run experiments and want to contribute results to the DQS training dataset, please open an issue with:
- Dataset description
- Feature vector `[AQ, DS, LD, PD, CB]`
- mAP@0.5 achieved
- Model architecture (YOLO version)

## Code Style

- Python: follow PEP8, type hints on all public functions
- Vue: Composition API + `<script setup>`
- No comments explaining *what* the code does — only *why* when non-obvious

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

# GitHub Roadmap — Auto Dataset Builder

## V0.1 → V1.0 里程碑規劃

---

## 總覽

```
V0.1  ████░░░░░░░░░░░░░░░░  Project Scaffold
V0.2  ████████░░░░░░░░░░░░  Data Collection
V0.3  ████████████░░░░░░░░  Auto Annotation (YOLO)
V0.4  ████████████████░░░░  Dataset Cleaning
V0.5  ██████████████████░░  SAM2 + LLM Verification
V0.6  ████████████████████  Neural DQS
V0.7  ████████████████████  Active Learning
V0.8  ████████████████████  Web Dashboard
V0.9  ████████████████████  Version Control + Export
V1.0  ████████████████████  Full Release + Paper
```

---

## V0.1 — Project Scaffold ✅ DONE (2026-06-09)

**目標**：建立基礎框架，讓所有模組有共同的介面。

### Issues / Tasks

- [x] 初始化 FastAPI 後端（`/api/v1`）
- [x] 初始化 Celery + Redis 工作佇列
- [x] 建立 PostgreSQL schema（datasets, jobs, images, annotations）
- [x] 建立 Docker Compose（dev 環境）
- [x] 建立 Python 專案結構（`backend/`, `workers/`, `models/`）
- [x] 基本 health check endpoint

### Deliverables

```
backend/
  app/
    api/
    models/
    workers/
  Dockerfile
docker-compose.dev.yml
requirements.txt
```

---

## V0.2 — Data Collection Module ✅ DONE (2026-06-09)

**目標**：能自動從 YouTube 和圖片搜尋引擎蒐集原始資料。

### Issues / Tasks

- [x] YouTube 下載器（`yt-dlp`，支援 720P/1080P）
- [x] 關鍵字自動擴充（rule-based，V0.5 升級為 LLM）
- [x] Google Image Search 爬取（Custom Search API）
- [x] imagehash.phash() 去重模組（threshold 可設定）
- [x] 進度回報到 Redis（Celery task state PROGRESS）
- [ ] 單元測試（下一輪補充）

### Deliverables

```
workers/
  collector/
    youtube_downloader.py
    image_searcher.py
    deduplicator.py
```

---

## V0.3 — Frame Extraction + YOLO Annotation ✅ DONE (2026-06-09)

**目標**：將影片轉換為標註好的訓練資料（第一階段）。

### Issues / Tasks

- [x] Fixed-rate frame extraction（1/2/5 FPS）
- [x] SSIM-based adaptive extraction
- [x] YOLOv11 inference wrapper（batch processing, class filter）
- [x] Confidence threshold 設定介面
- [x] 輸出 YOLO format（.txt labels + dataset.yaml）
- [x] 批次處理（batch_size=32）

### Deliverables

```
workers/
  extractor/
    frame_extractor.py   (fixed + adaptive)
  annotator/
    yolo_annotator.py
    yolo_formatter.py
```

---

## V0.4 — Dataset Cleaning Module ✅ DONE (2026-06-09)

**目標**：自動移除低品質影像。

### Issues / Tasks

- [x] 模糊偵測（Laplacian variance，可設 threshold）
- [x] 黑畫面偵測（mean pixel < 20）
- [x] 過曝偵測（histogram saturation > 98%）
- [x] 批次清理 pipeline（同時刪除對應 label 檔）
- [x] 清理報告（breakdown by reason）
- [x] dry_run 模式（只分析不刪除）

### Deliverables

```
workers/
  cleaner/
    blur_detector.py
    darkness_detector.py
    overexposure_detector.py
    cleaning_pipeline.py
```

---

## V0.5 — SAM2 + LLM Verification

**目標**：完成三階段標註流程，大幅提升標註品質。

### Issues / Tasks

- [ ] SAM2 整合（bbox prompt → mask）
- [ ] 從 mask 回推精修 bbox
- [ ] IoU 一致性檢查
- [ ] Vision LLM 驗證（Qwen-VL 或 LLaVA）
- [ ] Prompt template 管理
- [ ] Low confidence → 人工審核佇列
- [ ] 整合測試（三階段流程 end-to-end）

### Deliverables

```
workers/
  annotator/
    sam2_refiner.py
    llm_verifier.py
    annotation_pipeline.py   (three-stage)
```

---

## V0.6 — Neural DQS ✅ DONE (2026-06-09)

**目標**：實作資料集品質評估模組，並訓練 Neural DQS 模型。

### Issues / Tasks

- [x] AQ 計算（bbox area heuristic；V0.5 升級為 IoU）
- [x] DS 計算（CLIP embedding diversity + pixel fallback）
- [x] LD 計算（brightness entropy, 3 buckets）
- [x] PD 計算（aspect ratio entropy, 3 buckets）
- [x] CB 計算（1 - normalized Gini coefficient）
- [x] MLP Regressor 訓練腳本（sklearn Pipeline + StandardScaler）
- [x] SHAP 視覺化（KernelExplainer，需安裝 shap）
- [x] DQS API endpoint（POST /datasets/{id}/evaluate-dqs）
- [x] Heuristic fallback（geometric mean，未訓練時使用）

### Deliverables

```
models/
  dqs/
    feature_extractor.py
    neural_dqs.py
    shap_explainer.py
    train_dqs.py
```

---

## V0.7 — Active Learning Loop

**目標**：實現自動化資料飛輪。

### Issues / Tasks

- [ ] 低信心樣本篩選（conf < 0.5）
- [ ] 觸發重新標註流程
- [ ] DQS threshold 達標停止條件
- [ ] 迭代歷史記錄（每輪 mAP + DQS）
- [ ] max_iterations 安全上限

### Deliverables

```
workers/
  active_learning/
    uncertainty_sampler.py
    al_loop.py
    convergence_checker.py
```

---

## V0.8 — Web Dashboard ✅ DONE (2026-06-09)

**目標**：可視化操作介面，讓使用者監控整個 pipeline。

### Issues / Tasks

- [x] Vue 3 + Vite + TypeScript + Pinia 架構
- [x] Dashboard（dataset 列表 + 統計卡）
- [x] 建立任務（自然語言輸入 + example chips）
- [x] Dataset Detail（info + pipeline status）
- [x] DQS 雷達圖（Chart.js Radar，5 維度）
- [x] DQS 各維度進度條
- [x] Pipeline 步驟視覺化（done/active/pending）
- [x] Docker Compose frontend service（node:20-alpine）
- [ ] WebSocket 即時進度（V0.9 補充）

### Deliverables

```
frontend/
  src/
    views/
      CreateDataset.vue
      DatasetDetail.vue
      Dashboard.vue
    components/
      DQSRadarChart.vue
      BboxPreview.vue
      ProgressTimeline.vue
```

---

## V0.9 — Version Control + Multi-format Export

**目標**：資料集版本管理與輸出。

### Issues / Tasks

- [ ] Dataset snapshot（壓縮 + hash 記錄）
- [ ] Version diff（新增/刪除了哪些樣本）
- [ ] YOLO format export
- [ ] COCO JSON export
- [ ] dataset.yaml 自動生成
- [ ] Download API（zip 打包）

### Deliverables

```
backend/
  app/
    services/
      version_control.py
      exporter.py
```

---

## V1.0 — Full Release

**目標**：完整可用版本 + 論文提交。

### Issues / Tasks

- [ ] End-to-end integration test
- [ ] Performance benchmark（vs Manual labeling）
- [ ] README + Documentation
- [ ] GitHub Actions CI/CD
- [ ] Docker 生產環境配置
- [ ] 論文 Figure 與 Table 生成腳本
- [ ] Demo video

---

## 里程碑時間線（推甄導向）

| 版本  | 預計完成     | 核心功能                     |
| ----- | ------------ | ---------------------------- |
| V0.1  | Week 1       | 基礎架構                     |
| V0.2  | Week 2       | 資料蒐集                     |
| V0.3  | Week 3       | YOLO 自動標註                |
| V0.4  | Week 4       | 清理模組                     |
| V0.5  | Week 5~6     | SAM2 + LLM 驗證              |
| V0.6  | Week 7~8     | Neural DQS                   |
| V0.7  | Week 9       | Active Learning              |
| V0.8  | Week 10~11   | Web Dashboard                |
| V0.9  | Week 12      | 匯出與版本管理               |
| V1.0  | Week 13~14   | 完整整合 + Demo              |

**建議優先順序（最小可展示版本）**：

```
V0.1 → V0.2 → V0.3 → V0.6 → V0.8
```

先讓 DQS + Dashboard 可以跑起來，推甄資料最有說服力。

---

## GitHub Labels 建議

| Label         | 顏色    | 用途                       |
| ------------- | ------- | -------------------------- |
| `module`      | blue    | 對應各功能模組              |
| `paper`       | purple  | 與論文相關的實驗/圖表       |
| `demo`        | green   | 推甄展示必要功能            |
| `research`    | yellow  | 研究性質，非工程            |
| `good first`  | pink    | 容易入手的 issue           |

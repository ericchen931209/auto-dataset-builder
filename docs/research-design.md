# 研究假設與實驗設計

## 1. 研究問題（Research Questions）

**RQ1**: 自動化標註流程（YOLO + SAM2 + LLM）與人工標註相比，能否達到可接受的標註品質？

**RQ2**: Neural DQS 能否有效預測資料集訓練後的 mAP？

**RQ3**: 使用 ADB 建立的資料集，相比同等人工成本建立的資料集，訓練效果是否具有競爭力？

---

## 2. 研究假設（Hypotheses）

### H1: 標註品質假設

```
H1_null: ADB 自動標註與人工標註的 mAP 無顯著差異
H1_alt:  ADB 自動標註達到人工標註 mAP 的 90% 以上
```

**驗證方式**：
- 取 100 張影像，分別用 ADB 自動標註與人工標註
- 各訓練 YOLOv11（相同架構、相同 epoch）
- 比較 mAP@0.5

---

### H2: DQS 預測力假設

```
H2_null: DQS 與 mAP 之間無顯著線性相關
H2_alt:  Pearson r(DQS, mAP) > 0.80  (p < 0.05)
```

**驗證方式**：
- 建立 M ≥ 20 個不同品質的資料集（人為控制各維度）
- 計算每個資料集的 f(D)
- 訓練 YOLO 取得 mAP
- 計算 Pearson 相關係數

---

### H3: Active Learning 效益假設

```
H3_null: Active Learning 迭代無法顯著提升 mAP
H3_alt:  2 輪 Active Learning 後，mAP 提升 ≥ 5%
```

**驗證方式**：
- 初始資料集 D₀（~300 張）
- 執行 2 輪 active learning
- 記錄每輪 mAP 變化

---

## 3. 實驗設計（Proof of Concept）

### 3.1 資料集選擇

**主資料集**：台灣機車辨識（Taiwan Motorcycle Detection）

| 屬性      | 設定          |
| --------- | ------------- |
| 類別數    | 1（motorcycle）|
| 目標數量  | 500 ~ 800 張  |
| 來源      | YouTube + Google Images |
| 標註格式  | YOLO           |

---

### 3.2 實驗條件（Exp. Conditions）

為驗證 H2，建立 4 種品質等級的資料集版本：

| Version | 說明                             | 預期 DQS |
| ------- | -------------------------------- | -------- |
| D_low   | 模糊圖多、標註粗糙、無多樣性     | < 0.4    |
| D_mid   | 中等品質                         | 0.4~0.6  |
| D_high  | 高品質、通過全部清理流程          | 0.6~0.8  |
| D_full  | D_high + Active Learning         | > 0.8    |

---

### 3.3 Baseline 設定

| Baseline         | 說明                                     |
| ---------------- | ---------------------------------------- |
| Manual           | 人工用 LabelImg 標註 100 張              |
| YOLO-only        | 只用 YOLO 自動標註，無 SAM2/LLM 驗證    |
| ADB (ours)       | 完整三階段流程                           |
| ADB + AL         | 完整流程 + 2 輪 Active Learning          |

---

### 3.4 評估指標

| 指標             | 說明                                      |
| ---------------- | ----------------------------------------- |
| mAP@0.5          | 主要偵測效能指標                          |
| mAP@0.5:0.95     | COCO 標準，更嚴格                         |
| Precision        | 誤報率                                    |
| Recall           | 漏報率                                    |
| Annotation Time  | 每 100 張所需時間（人工 vs 自動）         |
| DQS              | 本文提出的品質分數                        |

---

### 3.5 實驗流程

```
Phase 1: 資料蒐集
─────────────────
  1. ADB 下載 YouTube 影片（台灣機車相關，10 部）
  2. 擷取 frame（SSIM adaptive，目標 600 frames）
  3. 去重、清理
  Result: ~500 raw frames

Phase 2: 自動標註
─────────────────
  1. YOLOv11n（pretrained on COCO）做 proposal
  2. SAM2 精修 bbox
  3. Qwen-VL / LLaVA 語意驗證
  Result: ~400 labelled samples (D_high)

Phase 3: 品質評估
─────────────────
  1. 計算 f(D) = [AQ, DS, LD, PD, CB]
  2. 計算 DQS
  3. 與人工標註版本對比

Phase 4: 模型訓練
─────────────────
  Train YOLOv11n on:
    - D_low / D_mid / D_high / D_full
    - Manual (100 imgs)
    - YOLO-only baseline

  Settings:
    epochs: 100
    imgsz: 640
    batch: 16
    split: 80/10/10 (train/val/test)

Phase 5: 分析
─────────────────
  1. mAP vs DQS 散點圖 → Pearson r
  2. 各 Baseline 比較表
  3. Active Learning 收斂曲線
  4. SHAP 視覺化（哪個維度影響最大）
```

---

## 4. 預期結果（Expected Results）

### Table 1: Annotation Method Comparison

| Method      | mAP@0.5 | Time (100 imgs) | Cost  |
| ----------- | ------- | --------------- | ----- |
| Manual      | ~0.82   | ~4 hrs          | High  |
| YOLO-only   | ~0.65   | ~5 min          | Low   |
| ADB (ours)  | ~0.78   | ~20 min         | Low   |
| ADB + AL    | ~0.81   | ~40 min         | Low   |

---

### Table 2: DQS Correlation

| Dataset | DQS  | mAP@0.5 |
| ------- | ---- | ------- |
| D_low   | 0.32 | 0.51    |
| D_mid   | 0.55 | 0.64    |
| D_high  | 0.71 | 0.76    |
| D_full  | 0.84 | 0.81    |

```
Pearson r ≈ 0.97   (p < 0.01)   ← 預期強相關
```

---

### Figure: DQS vs mAP Scatter Plot

```
mAP
1.0 │                              ● D_full
    │
0.8 │                    ● D_high
    │
0.6 │          ● D_mid
    │
0.4 │  ● D_low
    │
0.2 │
    └──────────────────────────────────── DQS
       0.2   0.4   0.6   0.8   1.0
```

---

## 5. 限制與未來工作（Limitations）

| 限制                               | 未來方向                          |
| ---------------------------------- | --------------------------------- |
| 僅驗證單一類別（機車）              | 擴展至多類別、多場景               |
| Neural DQS 訓練樣本少（M~20）       | 收集更多資料集建立 meta-dataset    |
| LLM 驗證 API 成本                   | Fine-tune 小型 open-source VLM    |
| 沒有跨資料集泛化實驗               | 在 VisDrone / COCO 子集重做實驗   |

---

## 6. 統計分析計畫

| 假設 | 統計方法         | 顯著水準 |
| ---- | ---------------- | -------- |
| H1   | Paired t-test    | p < 0.05 |
| H2   | Pearson r, 95%CI | p < 0.05 |
| H3   | Paired t-test    | p < 0.05 |

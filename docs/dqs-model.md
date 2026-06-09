# Dataset Quality Score (DQS) — 數學模型

## 1. 問題定義

給定一個資料集 **D = {(xᵢ, yᵢ)}ᵢ₌₁ᴺ**，其中：

- xᵢ：影像
- yᵢ：標註（bounding box + class label）

目標：學習一個函數 **g : D → [0, 1]**，使得：

```
g(D) ≈ mAP(YOLO trained on D)
```

---

## 2. Feature Extraction

從資料集 D 中萃取五個維度的特徵向量：

### f(D) = [AQ, DS, LD, PD, CB] ∈ ℝ⁵

---

### 2.1 Annotation Quality (AQ)

**動機**：標註品質差（bbox 太大/太小、邊界不準）會直接拉低模型 mAP。

**計算方法（Self-consistency check）**：

```
AQ(D) = (1/N) Σᵢ IoU( bbox_yolo(xᵢ), bbox_sam2(xᵢ) )
```

- bbox_yolo(xᵢ)：YOLOv11 預測框
- bbox_sam2(xᵢ)：SAM2 精修後的框（mask 最小外接矩形）
- IoU：Intersection over Union

**值域**：[0, 1]，越高表示標註一致性越好。

---

### 2.2 Diversity Score (DS)

**動機**：多樣性不足的資料集容易過擬合，泛化能力差。

**計算方法（CLIP Embedding Entropy）**：

```
eᵢ = CLIP_vision_encoder(xᵢ) ∈ ℝ⁵¹²

DS(D) = 1 - (1 / (N(N-1))) Σᵢ≠ⱼ cosine_similarity(eᵢ, eⱼ)
```

**值域**：[0, 1]，越高表示影像越多樣。

---

### 2.3 Lighting Diversity (LD)

**動機**：只有白天場景的資料集無法應對夜晚或室內環境。

**計算方法（Brightness Entropy）**：

將每張影像轉換至 HSV 色彩空間，取 V（brightness）通道：

```
bᵢ = mean(V channel of xᵢ)

將 bᵢ 分為 K=3 個 bucket：
  dark:    bᵢ < 85
  normal:  85 ≤ bᵢ < 170
  bright:  bᵢ ≥ 170

pₖ = proportion of images in bucket k

LD(D) = -Σₖ pₖ log(pₖ)  / log(K)   (normalized entropy)
```

**值域**：[0, 1]，均勻分布時等於 1。

---

### 2.4 Pose Diversity (PD)

**動機**：只有正面拍攝的物件資料集，對側面或背面角度的偵測能力差。

**計算方法（Aspect Ratio Entropy）**：

```
rᵢ = bbox_width(yᵢ) / bbox_height(yᵢ)

將 rᵢ 分為 K=3 個 bucket：
  front/back: 0.5 ≤ rᵢ ≤ 2.0
  side:       rᵢ > 2.0 or rᵢ < 0.5
  overhead:   (reserved for future)

PD(D) = -Σₖ pₖ log(pₖ)  / log(K)
```

**值域**：[0, 1]

**Note**：若有 pose estimation 模型可替換為更精確的姿態角度分布。

---

### 2.5 Class Balance (CB)

**動機**：類別不均衡（imbalanced classes）會導致 minority class 的 mAP 極低。

**計算方法（1 - Gini Coefficient）**：

```
nₖ = number of instances of class k
pₖ = nₖ / Σₖ nₖ

Gini(D) = 1 - Σₖ pₖ²

CB(D) = Gini(D) / (1 - 1/C)   (normalized, C = num classes)
```

**值域**：[0, 1]，完全均衡時等於 1。

---

## 3. Neural DQS Model

### 3.1 架構

```
f(D) ∈ ℝ⁵
   │
   ▼
┌────────────────────────────┐
│  MLP Regressor             │
│  FC(5→32) → ReLU           │
│  FC(32→16) → ReLU          │
│  FC(16→1) → Sigmoid        │
└────────────────────────────┘
   │
   ▼
DQS(D) ∈ [0, 1]
```

### 3.2 訓練

**訓練資料**：

收集 M 個資料集（合成生成或公開資料集切片）：

```
{(f(Dⱼ), mAPⱼ)}ⱼ₌₁ᴹ
```

**損失函數**：

```
L(θ) = (1/M) Σⱼ (DQS(Dⱼ; θ) - mAPⱼ)²   (MSE)
```

**正規化**：

```
L_reg(θ) = L(θ) + λ ||θ||²
```

### 3.3 推論

給定新資料集 D_new：

```
Step 1: Compute f(D_new) = [AQ, DS, LD, PD, CB]
Step 2: DQS(D_new) = g(f(D_new); θ*)
Step 3: If DQS < θ_q → trigger active learning
        If DQS ≥ θ_q → export dataset
```

---

## 4. Interpretability

除了純 Neural 版本，亦提供 **Explainable DQS (xDQS)**：

```
xDQS(D) = SHAP(g, f(D))
```

輸出每個維度對最終分數的貢獻：

```
ΔmAP ≈ φ_AQ + φ_DS + φ_LD + φ_PD + φ_CB
```

- φₖ：SHAP value，正值表示提升、負值表示拖累

**優點**：
- 告訴使用者哪個維度不足
- 提供可解釋的資料改善建議
- 增加論文說服力

---

## 5. Baseline Comparison

| Method          | Type          | Interpretable | Predicts mAP |
| --------------- | ------------- | ------------- | ------------ |
| Manual QA       | Human         | Yes           | No           |
| Simple Average  | Linear        | Partial       | No           |
| Weighted Sum    | Linear        | Yes           | Partial      |
| **Neural DQS**  | Learned (MLP) | via SHAP      | **Yes**      |

---

## 6. 預期論文結論

```
Hypothesis: DQS(D) is a statistically significant predictor of mAP.
            Pearson r(DQS, mAP) > 0.85  (p < 0.05)
```

---

## 7. 符號整理（供論文使用）

| Symbol  | Definition                            |
| ------- | ------------------------------------- |
| D       | Dataset {(xᵢ, yᵢ)}                    |
| N       | Number of samples                     |
| C       | Number of classes                     |
| f(D)    | Feature vector ∈ ℝ⁵                   |
| g(·;θ)  | Neural DQS regressor                  |
| DQS(D)  | Predicted dataset quality score       |
| mAP(D)  | Mean Average Precision after training |
| θ_q     | Quality threshold (default: 0.75)     |
| φₖ      | SHAP value for feature k              |

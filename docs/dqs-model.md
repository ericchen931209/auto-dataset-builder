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

從資料集 D 中萃取六個維度的特徵向量：

### f(D) = [AQ, IQ, CD, LD, PD, CB] ∈ ℝ⁶

---

### 2.1 Annotation Quality (AQ)

**動機**：標註品質差（missing labels、bbox 偏移）會直接拉低模型 mAP。

**計算方法**：

```
completeness = fraction of images that have at least one annotation

geometry = (1/N) Σᵢ mean_IoU_within_image(yᵢ)
           (self-consistency: aspect ratio variance penalty)

AQ(D) = 0.6 × completeness + 0.4 × geometry
```

**值域**：[0, 1]，越高表示標註越完整且幾何一致。

---

### 2.2 Image Quality (IQ)

**動機**：模糊影像與高雜訊影像都會降低模型對特徵的學習能力；兩者需要同時懲罰。

**計算方法（Composite metric）**：

```
blur_score       = Var(Laplacian(medianBlur(x, 3))) / 500  (clipped to [0,1])

noise_residual   = x - GaussianBlur(x, (5,5), σ=1.5)
noise_std        = std(noise_residual)
noise_cleanliness = 1 - min(noise_std / 25.0, 1.0)

IQ(D) = mean over images of √(blur_score × noise_cleanliness)
```

**值域**：[0, 1]，同時高 sharpness 且低 noise 時趨近 1。

---

### 2.3 CLIP Diversity (CD)

**動機**：影像在語意層面（CLIP embedding space）的多樣性比像素層面更能反映模型泛化能力；模糊、亮度退化等降質在 CLIP 空間中會造成 embedding 聚集，使 CD 下降。

**計算方法（Mean pairwise cosine distance）**：

```
eᵢ = normalize( CLIP_ViT-B/32(xᵢ) )  ∈ ℝ⁵¹²

CD(D) = 1 - (2 / (N(N-1))) Σᵢ<ⱼ eᵢ · eⱼ
```

**值域**：[0, 1]，越高表示影像在語意空間越多樣。

**實驗觀察**：Pearson r(CD, mAP) = **0.892**（n=96），是六個特徵中最強的預測因子。

---

### 2.4 Lighting Diversity (LD)

**動機**：只有白天場景的資料集無法應對夜晚或室內環境。

**計算方法（Brightness Entropy）**：

```
bᵢ = mean(V channel of HSV(xᵢ))

將 bᵢ 分為 K=3 個 bucket：
  dark:    bᵢ < 85
  normal:  85 ≤ bᵢ < 170
  bright:  bᵢ ≥ 170

pₖ = proportion of images in bucket k

LD(D) = -Σₖ pₖ log(pₖ)  / log(K)   (normalized entropy)
```

**值域**：[0, 1]，均勻分布時等於 1。

---

### 2.5 Pose Diversity (PD)

**動機**：只有正面拍攝的物件資料集，對側面或背面角度的偵測能力差。

**計算方法（Aspect Ratio Entropy）**：

```
rᵢ = bbox_width(yᵢ) / bbox_height(yᵢ)

PD(D) = -Σₖ pₖ log(pₖ)  / log(K)
```

**值域**：[0, 1]

---

### 2.6 Class Balance (CB)

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

### 3.1 架構（small-sample regime）

對於 n < 100 個訓練樣本，使用 Ridge Regression with Polynomial Features（degree=2）以避免過擬合：

```
f(D) ∈ ℝ⁶
   │
   ▼
StandardScaler
   │
   ▼
PolynomialFeatures(degree=2)  →  ℝ²⁸
   │
   ▼
Ridge(α=1.0)
   │
   ▼
DQS(D) ∈ ℝ  (predicted mAP@0.5)
```

對於 n ≥ 100 個訓練樣本，改用 MLP (64, 32) with ReLU。

### 3.2 訓練

**訓練資料**：

```
{(f(Dⱼ), mAPⱼ)}ⱼ₌₁ᴹ   M = 96 (COCO128 controlled degradation variants)
```

**降質類型**（10 categories）：
- Label missing（10%–90%）
- Label noise（bbox shift 3%–20%）
- Blur（kernel 3–61）
- Gaussian noise（σ=2–100）
- Brightness（factor 0.05–2.0）
- Combined degradations（blur+dark, noise+dark, noise+blur）

### 3.3 實驗結果

| Metric         | Value  |
| -------------- | ------ |
| Train r        | 0.970  |
| CV r (k=5)     | **0.929** |
| CV R²          | 0.854  |
| CV MSE         | 0.0033 |
| n              | 96     |

Feature–mAP Pearson correlations:
| Feature | r |
|---------|---|
| CD (CLIP Diversity) | +0.892 |
| IQ (Image Quality)  | +0.661 |
| LD (Lighting)       | +0.264 |
| CB (Class Balance)  | -0.140 |
| PD (Pose)           | -0.067 |
| AQ (Annotation)     | -0.042 |

> Note: AQ 在 COCO128 實驗中相關性低，因為 COCO128 標註品質本身就高且固定；missing label variant 已由 completeness component 捕捉。

### 3.4 推論

給定新資料集 D_new：

```
Step 1: Compute f(D_new) = [AQ, IQ, CD, LD, PD, CB]
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
ΔmAP ≈ φ_AQ + φ_IQ + φ_CD + φ_LD + φ_PD + φ_CB
```

- φₖ：SHAP value，正值表示提升、負值表示拖累

---

## 5. Baseline Comparison

| Method          | Type            | Interpretable | Predicts mAP |
| --------------- | --------------- | ------------- | ------------ |
| Manual QA       | Human           | Yes           | No           |
| Simple Average  | Linear          | Partial       | No           |
| Weighted Sum    | Linear          | Yes           | Partial      |
| **Neural DQS**  | Ridge+Poly(n<100) / MLP(n≥100) | via SHAP | **Yes** |

---

## 6. 論文核心假設

```
Hypothesis: DQS(D) is a statistically significant predictor of mAP.
            Pearson r(DQS, mAP) > 0.85  (p < 0.05)

Result:     CV Pearson r = 0.929  (n=96, p < 0.001)  ✓
```

---

## 7. 符號整理（供論文使用）

| Symbol  | Definition                            |
| ------- | ------------------------------------- |
| D       | Dataset {(xᵢ, yᵢ)}                    |
| N       | Number of samples                     |
| C       | Number of classes                     |
| f(D)    | Feature vector ∈ ℝ⁶                   |
| g(·;θ)  | Neural DQS regressor                  |
| DQS(D)  | Predicted dataset quality score       |
| mAP(D)  | Mean Average Precision after training |
| θ_q     | Quality threshold (default: 0.75)     |
| φₖ      | SHAP value for feature k              |

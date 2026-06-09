# Auto Dataset Builder - Exposure & Growth Plan

## 專案曝光與成長規劃

---

# 目標

本計畫旨在提升 Auto Dataset Builder (ADB) 的：

* GitHub Star 數量
* 使用者數量
* 學術影響力
* 開源社群影響力
* 推甄與研究價值

最終目標：

> 將 Auto Dataset Builder 打造成 AI Dataset Construction 領域的重要開源專案。

---

# 第一階段：產品可用性優化

## 目標

讓陌生人能在 10 分鐘內成功使用系統。

---

## 必須完成

### Docker 一鍵部署

```bash
git clone https://github.com/ericchen931209/auto-dataset-builder

cd auto-dataset-builder

docker compose up
```

即可啟動。

---

### 建立 Quick Start

README 最前方新增：

```text
1. Install Docker
2. Clone Repository
3. Run Docker Compose
4. Open Browser
5. Create Dataset
```

---

### 建立範例 Dataset

提供：

* Taiwan Scooter Dataset
* Taiwan Helmet Dataset
* Taiwan Traffic Dataset

供使用者快速測試。

---

# 第二階段：影片推廣

## 目標

讓使用者快速理解專案價值。

---

## Demo Short (60 秒)

展示流程：

```text
輸入需求

↓

自動蒐集資料

↓

自動標註

↓

資料清理

↓

資料集輸出
```

---

## Full Demo (5 分鐘)

內容：

### 問題背景

目前資料集建立流程耗時且昂貴。

---

### 系統介紹

展示系統架構。

---

### 實際操作

建立：

```text
Taiwan Scooter Dataset
```

---

### 結果展示

輸出：

```text
YOLO Dataset
```

---

### 未來規劃

介紹：

* Active Learning
* Neural DQS
* Agent System

---

# 第三階段：建立官方網站

## 目標

增加專業度與搜尋曝光。

---

## 首頁

標語：

```text
Build AI Datasets Automatically.
```

---

## 功能展示

展示：

* Collect
* Annotate
* Clean
* Train
* Export

流程動畫。

---

## 效能比較

| 方法                   | 所需時間  |
| -------------------- | ----- |
| 人工建立資料集              | 12 小時 |
| Auto Dataset Builder | 45 分鐘 |

---

## 技術架構

展示：

* YOLO
* SAM2
* LLM
* Active Learning
* Neural DQS

---

# 第四階段：GitHub 優化

## README 優化

加入：

### 專案 Logo

### 架構圖

### GIF Demo

### 安裝教學

### 使用案例

### Benchmark

### Roadmap

---

## GitHub Topics

新增：

```text
dataset-builder
computer-vision
yolo
sam2
llm
active-learning
dataset-generation
ai-agent
```

---

## Releases

建立：

```text
v0.1
v0.2
v0.3
v1.0
```

正式版本。

---

# 第五階段：Hugging Face 生態系

## 建立 Organization

```text
Auto Dataset Builder
```

---

## 公開 Dataset

建立：

### Taiwan Scooter Dataset

### Taiwan Helmet Dataset

### Taiwan Traffic Dataset

### Taiwan License Plate Dataset

---

## 公開模型

建立：

```text
ADB-YOLO-Scooter
```

```text
ADB-YOLO-Helmet
```

等模型。

---

# 第六階段：技術文章

## Medium

撰寫：

### Article 1

```text
How I Built an Automatic Dataset Generator
```

---

### Article 2

```text
Can Dataset Quality Predict mAP?
```

---

### Article 3

```text
Building Active Learning Pipelines with YOLO and SAM2
```

---

## Hackernoon

同步發佈。

---

## 個人部落格

建立：

```text
research.ericchen.tw
```

記錄開發歷程。

---

# 第七階段：社群推廣

## Reddit

發佈至：

### r/MachineLearning

介紹專案。

---

### r/computervision

分享技術細節。

---

### r/OpenSource

分享開源成果。

---

## Discord 社群

加入：

* Computer Vision
* Open Source
* Machine Learning

相關社群。

---

# 第八階段：研究與論文

## 預印本論文

投稿：

### arXiv

論文題目：

```text
Auto Dataset Builder:
An LLM-Assisted Framework for Automatic Dataset Construction
with Neural Dataset Quality Scoring
```

---

## 論文核心

研究：

```text
Neural DQS
```

與：

```text
mAP
```

之間的關聯性。

---

## Benchmark Dataset

使用：

* COCO
* Pascal VOC
* VisDrone
* BDD100K
* Roboflow100

進行驗證。

---

# 第九階段：建立學術影響力

## 發表海報

參加：

* AI Workshop
* Computer Vision Workshop
* 學生論文競賽

---

## 研討會論文

目標：

### 國內

* TANET
* 資訊工程研討會

---

### 國際

* ICCE
* ICCE-TW
* ICIA

---

# 第十階段：打造研究品牌

## 建立個人網站

內容：

### About

### Projects

### Publications

### Datasets

### Contact

---

## 建立研究品牌

統一名稱：

```text
Eric Chen
AI Researcher
Computer Vision
Dataset Intelligence
```

---

# 成功指標 (KPI)

## GitHub

| 階段 | 目標 |
|------|------|
| 第一階段 | 100 Stars |
| 第二階段 | 500 Stars |
| 長期 | 1000 Stars |

---

## Hugging Face

```text
5 Public Datasets
3 Public Models
```

---

## 論文

```text
1 Preprint (arXiv)
1 Conference Paper
```

---

## 使用者

```text
100+ Users
1000+ Dataset Builds
```

---

# 時間規劃

| 時間 | 目標 |
|------|------|
| 2026 Q1 | Docker Deploy、Demo Video、Website |
| 2026 Q2 | Hugging Face Dataset、Technical Blogs、Reddit |
| 2026 Q3 | Neural DQS Research、Benchmark、Paper Draft |
| 2026 Q4 | Conference Submission、GitHub 100+ Stars、推甄資料整理 |

---

# 最終願景

建立一個讓任何人只需輸入：

```text
Build a Taiwan Scooter Dataset
```

即可自動產生高品質 AI 訓練資料集的平台。

讓 Auto Dataset Builder 成為：

> Dataset Construction 領域的代表性開源工具。

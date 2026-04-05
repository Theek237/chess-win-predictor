# ♟️ Chess Win Predictor

A machine learning web application that predicts the outcome of a chess game — **White Win**, **Black Win**, or **Draw** — based on player ratings, opening choice, and time control. Trained on **6.26 million real games** from Lichess.

---

## Live Demo

> Run locally with `streamlit run app.py`

![App Screenshot](https://raw.githubusercontent.com/Theek237/chess-win-predictor/Theenuka/assets/app_preview.png)

---

## Overview

| | |
|---|---|
| **Problem Type** | Supervised Classification (3 classes) |
| **Dataset** | [Lichess Chess Games — Kaggle](https://www.kaggle.com/datasets/arevel/chess-games) |
| **Dataset Size** | 6.26 million games |
| **Best Model** | XGBoost (63% accuracy) |
| **Baseline** | Random Forest (46% accuracy) |
| **Tech Stack** | Python, Scikit-Learn, XGBoost, Streamlit |

---

## Features

### Input Features Used
| Feature | Description |
|---|---|
| `WhiteElo` | White player's Elo rating (500–3500) |
| `BlackElo` | Black player's Elo rating (500–3500) |
| `ECO` | Opening code (A00–E99, 493 unique openings) |
| `TimeControl` | Game time format (e.g. 300+5, 180+0) |
| `EloDifference` | White Elo − Black Elo (engineered) |
| `AbsEloDiff` | Absolute Elo difference (engineered) |
| `CloseMatch` | 1 if Elo gap < 100 pts (draw signal) |
| `ECO_Family` | Opening family group (A/B/C/D/E) |
| `TimeCategory` | Bullet / Blitz / Rapid / Classical |

### Target Variable
`Result` → **0** = Black Win · **1** = White Win · **2** = Draw

---

## ML Pipeline

```
Raw Data (6.26M games)
        │
        ▼
 Data Cleaning & Filtering
 (remove invalid TimeControl/Result)
        │
        ▼
 Feature Engineering
 (EloDifference, AbsEloDiff, CloseMatch,
  ECO_Family, TimeCategory)
        │
        ▼
 Label Encoding (ECO, TimeControl, Result)
        │
        ▼
 Class Balancing (equal 3-way sampling)
        │
        ▼
 Train/Test Split (80/20)
        │
        ▼
 Feature Scaling (StandardScaler)
        │
        ▼
 Model Training & Comparison
 ┌──────────────────┬──────────────────┐
 │ Logistic         │ Random Forest    │
 │ Regression       │ (Tuned)          │
 │ Accuracy: 46%    │ Accuracy: 46%    │
 └──────────────────┴──────────────────┘
        │
        ▼
 Improvements (Section 11)
 ┌──────────────────┬──────────────────┐
 │ RF + New         │ XGBoost +        │
 │ Features         │ New Features     │
 │ Accuracy: 50%    │ Accuracy: 63% ✓  │
 └──────────────────┴──────────────────┘
        │
        ▼
 Best Model Saved → Streamlit App
```

---

## Model Results

### Baseline Models

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 46% | Linear baseline |
| Random Forest (default) | 42% | Underfitting |
| Random Forest (tuned) | 46% | `max_depth=12`, `n_estimators=100` |

### Improved Models (Section 11 — 9 features)

| Model | Accuracy | Black Win F1 | White Win F1 | Draw F1 |
|---|---|---|---|---|
| RF + New Features | 50% | 0.57 | 0.58 | 0.09 |
| **XGBoost** | **63%** | **0.62** | **0.66** | **—** |

> **Note:** Draws have low recall because they naturally occur far less often in real chess (~28% of games vs ~37% white wins). This is realistic, not a bug — the model faithfully reflects actual game distributions.

### Why 63% is Good
- Random baseline = 33% (3 classes)
- Our best model = 63% → **30 percentage points above random**
- Chess outcomes are inherently noisy — even grandmasters can't perfectly predict results

---

## App Features

- **Live Elo Difference metric** — shows white/black advantage in real time
- **Time Format detection** — auto-classifies Bullet / Blitz / Rapid / Classical
- **Opening Family display** — groups ECO codes into A/B/C/D/E families
- **Confidence indicator** — High / Medium / Low with color coding
- **Context hints** — warns when players are evenly matched (draw more likely) or Elo gap is large
- **Probability breakdown** — visual progress bars for all 3 outcomes
- **Crash-safe encoding** — graceful fallback for unseen ECO/TimeControl values

---

## Project Structure

```
chess-win-predictor/
├── app.py               # Streamlit web application
├── chess.ipynb          # Full ML pipeline (data → training → evaluation)
├── requirements.txt     # Python dependencies
├── chess_rf_model.pkl   # Trained model (XGBoost, 63% accuracy)
├── scaler.pkl           # StandardScaler for feature normalization
├── le_eco.pkl           # LabelEncoder for ECO opening codes (493 classes)
└── le_time.pkl          # LabelEncoder for time controls (840 classes)
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Theek237/chess-win-predictor.git
cd chess-win-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Dataset

**Source:** [arevel/chess-games on Kaggle](https://www.kaggle.com/datasets/arevel/chess-games)

| Property | Value |
|---|---|
| Total games | 6,256,184 |
| Original features | 14 |
| Engineered features | 4 |
| Platform | Lichess (online chess) |
| Time period | 2013–2017 |

Raw columns: `Event`, `White`, `Black`, `Result`, `UTCDate`, `UTCTime`, `WhiteElo`, `BlackElo`, `WhiteRatingDiff`, `BlackRatingDiff`, `ECO`, `Opening`, `TimeControl`, `Termination`, `AN`

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | 1.32.0 | Web UI |
| `scikit-learn` | 1.6.1 | ML models, preprocessing, evaluation |
| `xgboost` | 2.0.3 | Best performing model |
| `pandas` | 2.2.1 | Data manipulation |
| `numpy` | 1.26.4 | Numerical operations |
| `joblib` | 1.3.2 | Model serialization |
| `matplotlib` + `seaborn` | — | Visualization |

---

## Author

**Theenuka Bandara**
- GitHub: [@Theek237](https://github.com/Theek237)
- Email: theenukabandara@gmail.com

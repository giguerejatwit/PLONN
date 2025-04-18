# 🧠 PLONN: Parlay Learning Optimization Neural Network

**Welcome to PLONN**, a unified sports betting research platform designed to forecast game outcomes and player performance across NBA and MLB using advanced machine learning and deep learning techniques. This repo powers daily predictions, strategy testing, and betting edge detection through statistical modeling.

---

## 🏀 NBA Total Points Model (PLONN-NBA)

This module predicts final game scores for NBA matchups using an MLP-based neural network. The predictions are benchmarked against DK lines for O/U evaluation and betting optimization.

### 🔧 Model Overview:
- **Input**: Twin vectors for each team (home/away), capturing season averages
- **Target**: Final game total and individual team points
- **Optimized With**: Feature engineering + backtesting vs DK lines

### 💡 Strategy Layer:
- Post-processing module to determine betting edge (O/U bets)
- Integrates reinforcement learning logic (via OpenAI Gym) for parlay simulation and bankroll management

### 🛠 Key Files:
- `nba_total_pts.py` – daily prediction generator
---

## ⚾ MLB Strikeout Prediction Pipeline (PLONN-MLB)

This system predicts daily MLB starting pitcher strikeouts using both LSTM and MLP neural nets. It leverages rolling stats, team batting encodings, and DK line comparisons.

### 🔁 LSTM Model
- **Input**: Past 2–3 games of pitcher performance
- **Features**: `K_prev`, `K_avg_3`, `IP_avg_3`, `BB`, `BF`, `encoded_0...7`
- **Output**: Strikeout prediction (raw)

### 🧠 MLP Model (Scaled)
- **Input**: Latest appearance using 11 features
- **Features**: `K_prev`, `K_avg_3`, `IP_avg_3`, `encoded_0...7`
- **Target**: `SO` (scaled)
- **Regularization**: Dropout + inverse-transform prediction

### 📈 Supporting Tools:
- `preprocess.py`: Adds rolling metrics + opponent encoding
- `lstm_strikeouts.py` / `mlp_strikeouts.py`
- `predict_today.py` / `predict_mlp_scaled.py`
- `get_games.py`: Gets daily matchups from Baseball Reference

### 🔍 Betting Evaluation:
- Daily predictions matched to DK strikeout props
- Evaluates prediction margin (confidence), edge detection
- Ensemble voting logic and confidence threshold filtering

---

## 📁 Folder Structure
```bash
Leagues/
├── NBA/
│   ├── nba_total_pts.py
│   ├── data/
│   ├── models/
├── MLB/
│   ├── train_lstm.py
│   ├── train_mlp_scaled.py
│   ├── predict_today.py
│   ├── scrape_games_bs4.py
│   ├── data/
│   ├── models/
```


## 🚧 Roadmap
- ✅ Daily prediction integration for both NBA + MLB
- 🔁 Confidence-weighted ensemble voting
- 🧠 RL for parlay optimization (NBA)
- ☁️ Cloud deployment + Google Sheets/DB hooks (WIP)

---

## 👨‍💻 Created by: Jake Giguere  
– Reinforcement Learning + Sports Analytics


# ⚾ MLB Strikeout Prediction Pipeline (PLONN-MLB)

This system is designed to predict daily MLB starting pitcher strikeouts using a hybrid deep learning approach. We utilize both LSTM and MLP architectures and optimize predictions with engineered features, autoencoded batting stats, and betting line comparisons.

---

## 🧠 Models Used

### 🔁 LSTM Model
- **Input**: Past 2–3 games of pitcher performance
- **Features**: `K_prev`, `K_avg_3`, `IP_avg_3`, `BB`, `BF`, `encoded_0...7`
- **Output**: Predicted strikeouts (Ks) for today's game
- **Note**: Input features are normalized, target is not scaled

### 🧠 MLP Model (Scaled)
- **Input**: Most recent game stats using 11 core features
- **Features**: `K_prev`, `K_avg_3`, `IP_avg_3`, `encoded_0...7`
- **Target**: Strikeouts (`SO`), standard-scaled
- **Output**: Inverse-transformed Ks prediction
- **Regularization**: Dropout layer included to reduce overfitting

---

## 📊 Data Sources
- **Batting Stats**: 5-game rolling spans, encoded using an autoencoder neural network
- **Pitcher Game Logs**: Pulled from historical data, enhanced with rolling performance metrics
- **Today’s Starters**: Scraped daily from [Baseball Reference Previews](https://www.baseball-reference.com/previews/)
- **DK Betting Lines**: Manually input for validation and profit tracking

---

## 🛠 Key Components

### `train_lstm.py`
- Trains the LSTM model using sequential (temporal) pitcher appearance data

### `train_mlp_scaled.py`
- Trains the MLP model on most recent pitcher appearances
- Scales both input features and target strikeouts (using `StandardScaler`)
- Saves fitted scalers (`mlp_x_scaler.pkl`, `mlp_y_scaler.pkl`)

### `predict_today.py`
- Predicts using LSTM or MLP models
- Pads inputs to 3-game format for LSTM
- Fuzzy name matching handles inconsistencies between sources

### `predict_mlp_scaled.py`
- Loads the MLP model + scalers
- Scales input, runs prediction, and inverse-transforms output
- Output is saved to `today_predictions_mlp_scaled.csv`

### `preprocess.py`
- Cleans dates and columns
- Adds rolling pitching features: `K_avg_3`, `IP_avg_3`, etc.
- Encodes opponent batting spans
- Generates the combined `gamelog_final.csv`

### `scrape_games_bs4.py`
- Pulls today's scheduled matchups and probable pitchers using BeautifulSoup
- Scrapes directly from Baseball Reference's preview page

---

## 📁 Folder Structure
```bash
Leagues/
├── MLB/
│   ├── data/
│   │   ├── gamelog_final.csv
│   │   ├── today_pitchers.csv
│   │   ├── today_predictions.csv
│   │   ├── today_predictions_mlp_scaled.csv
│   ├── models/
│   │   ├── lstm_strikeout_model.h5
│   │   ├── mlp_strikeout_model_scaled.keras
│   │   ├── mlp_x_scaler.pkl
│   │   ├── mlp_y_scaler.pkl
```

---

## 📋 Logging
Daily predictions, model outputs, and DK comparisons will be logged using Python's `logging` module.
- Logs will be rotated daily (e.g., `mlb_predictions_2025-04-18.log`)
- Includes player name, prediction value, model used, DK line, and result (W/L)
- Useful for historical auditing and long-term model tracking

---

## ✅ Next Steps
- Schedule daily runs via `cron`
- Scrape DK lines automatically
- Implement ensemble voting + confidence thresholds
- Push daily logs + output to Google Sheets or database for tracking
- Visualizations and dashboard coming soon

---

## 👨‍💻 Created by: Jake Giguere 🚀


import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
from difflib import get_close_matches
import joblib

# === Load Data ===
gamelog = pd.read_csv("Leagues/MLB/data/gamelog_final.csv")
today_pitchers = pd.read_csv("Leagues/MLB/data/today_pitchers.csv")
mlp = load_model("Leagues/MLB/models/mlp_strikeout_model_scaled.keras")
x_scaler = joblib.load("Leagues/MLB/models/mlp_x_scaler.pkl")
y_scaler = joblib.load("Leagues/MLB/models/mlp_y_scaler.pkl")

# === MLP Feature Columns ===
mlp_feature_cols = ['K_prev', 'K_avg_3', 'IP_avg_3'] + [f'enc_{i}' for i in range(8)]

# === Ensure datetime ===
gamelog['Date'] = pd.to_datetime(gamelog['Date'])
today = pd.to_datetime(datetime.now().date())

# === Fuzzy match function ===
def match_player(name, all_names):
    matches = get_close_matches(name, all_names, n=1, cutoff=0.8)
    return matches[0] if matches else None

# === Collect Predictions ===
predictions = []

for _, row in today_pitchers.iterrows():
    name = row['Player']
    team = row['Team']
    opponent = row['Opponent']

    matched_name = match_player(name, gamelog['Player'].unique())
    if not matched_name:
        continue

    player_games = gamelog[gamelog['Player'] == matched_name].sort_values(by='Date')
    if len(player_games) < 2:
        continue

    recent_games = player_games.iloc[-3:]
    X = recent_games[mlp_feature_cols].values
    X_flat = X[-1]
    X_scaled = x_scaler.transform([X_flat])

    y_pred_scaled = mlp.predict(X_scaled)[0][0]
    y_pred = y_scaler.inverse_transform([[y_pred_scaled]])[0][0]

    predictions.append({
        'Date': today.date(),
        'Player': name,
        'Team': team,
        'Opponent': opponent,
        'Predicted_Ks': round(y_pred, 2)
    })

# === Save Results ===
pred_df = pd.DataFrame(predictions)
pred_df.to_csv("Leagues/MLB/data/today_predictions_mlp_scaled.csv", index=False)
print(pred_df)
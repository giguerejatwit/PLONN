import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
from difflib import get_close_matches
from utils.logger import setup_logger

logger = setup_logger(name='LSTM')
# === Load Data ===
gamelog = pd.read_csv("Leagues/MLB/data/gamelog_final.csv")
today_pitchers = pd.read_csv("Leagues/MLB/data/today_pitchers.csv")
lstm = load_model("Leagues/MLB/models/lstm_strikeout_model.h5")
mlp = load_model("Leagues/MLB/models/mlp_strikeout_model.h5")

# === Define Feature Columns ===
feature_cols = ['K_prev', 'K_avg_3', 'BB', 'BF', 'IP_avg_3'] + [f'enc_{i}' for i in range(8)]

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

    # Fuzzy match player name
    matched_name = match_player(name, gamelog['Player'].unique())
    if not matched_name:
        continue

    player_games = gamelog[gamelog['Player'] == matched_name].sort_values(by='Date')
    if len(player_games) < 2:
        continue
    
    recent_games = player_games.iloc[-3:]
    X = recent_games[feature_cols].values

    # Pad to [3 x features] if only 2 games
    if X.shape[0] == 2:
        padding = np.zeros((1, X.shape[1]))
        X = np.vstack([padding, X])
    elif X.shape[0] == 1:
        continue  # still too short

    X_input = np.expand_dims(X, axis=0)
    predicted_ks = lstm.predict(X_input)[0][0]

    predictions.append({
        'Date': today.date(),
        'Player': name,
        'Team': team,
        'Opponent': opponent,
        'Predicted_Ks': round(predicted_ks, 2)
    })

# === Save Results ===
pred_df = pd.DataFrame(predictions)
# pred_df.to_csv("Leagues/MLB/data/today_predictions.csv", index=False)
logger.info(pred_df)
print(pred_df)

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor

# Step 1: Load data
game_logs = pd.read_csv('Leagues/MLB/data/2024/gamelog_final.csv')
batting = pd.read_csv('Leagues/MLB/data/2024/encoded_5game_spans.csv')

# Step 2: Parse Result column
def parse_result(result):
    match = re.match(r'([WL]), (\d+)-(\d+)', str(result))
    if match:
        outcome, team_runs, opp_runs = match.groups()
        return pd.Series([1 if outcome == 'W' else 0, int(team_runs), int(opp_runs)])
    return pd.Series([None, None, None])

game_logs[['Win', 'Runs_Scored', 'Runs_Allowed']] = game_logs['Result'].apply(parse_result)

# Step 3: Convert dates
game_logs['Date'] = pd.to_datetime(game_logs['Date'])
batting['Span Ended'] = pd.to_datetime(batting['Span Ended'], format='mixed', errors='coerce')

# Step 4-6: Filter & calculate features
starters = game_logs[game_logs['App,Dec'].str.contains("GS", na=False)].copy()
team_game_logs = starters.groupby(['Team', 'Date']).first().reset_index()

team_game_logs['SO9'] = team_game_logs.apply(lambda row: (row['SO'] / row['IP'] * 9) if row['IP'] > 0 else 0, axis=1)
team_game_logs['ERA_game'] = team_game_logs.apply(lambda row: (row['ER'] / row['IP']) * 9 if row['IP'] > 0 else 0, axis=1)
team_game_logs['WHIP_game'] = team_game_logs.apply(lambda row: (row['BB'] + row['H']) / row['IP'] if row['IP'] > 0 else 0, axis=1)
team_game_logs['BB_per9'] = team_game_logs.apply(lambda row: (row['BB'] / row['IP']) * 9 if row['IP'] > 0 else 0, axis=1)
team_game_logs['H_per9'] = team_game_logs.apply(lambda row: (row['H'] / row['IP']) * 9 if row['IP'] > 0 else 0, axis=1)
team_game_logs = team_game_logs.sort_values(by=['Player', 'Date'])
team_game_logs['days_rest'] = team_game_logs.groupby('Player')['Date'].diff().dt.days.fillna(999)

# Step 7: Merge encoded batting spans
batting_sorted = batting.sort_values(['Team', 'Span Ended'])
def find_latest_span(team, date):
    spans = batting_sorted[(batting_sorted['Team'] == team) & (batting_sorted['Span Ended'] <= date)]
    return spans.iloc[-1] if not spans.empty else pd.Series([np.nan]*len(batting.columns), index=batting.columns)

merged_rows = team_game_logs.apply(lambda row: pd.concat([row, find_latest_span(row['Team'], row['Date'])]), axis=1)

# Step 8: Build dataset
model_data = merged_rows[[
    'Team', 'Date', 'Opp', 'Runs_Scored',  'SO9', 'ERA_game', 'H_per9', 'days_rest',
    'encoded_0', 'encoded_1', 'encoded_2', 'encoded_3', 'encoded_4', 'encoded_5', 'encoded_6', 'encoded_7'
]]
model_data.replace([np.inf, -np.inf], np.nan, inplace=True)
model_data.dropna(inplace=True)

# --- XGBoost Model ---
X = model_data.drop(columns=['Team', 'Date', 'Opp', 'Runs_Scored'])
y = model_data['Runs_Scored']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(random_state=42, n_estimators=30, learning_rate=0.1, max_depth=3)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print(f"✅ XGB MAE: {mean_absolute_error(y_test, y_pred_xgb):.2f}")
print(f"✅ XGB R²: {r2_score(y_test, y_pred_xgb):.2f}")

# --- TensorFlow MLP ---
top_features = ['encoded_1', 'encoded_5', 'H_per9', 'encoded_3', 'SO9', 'days_rest',
                'ERA_game', 'encoded_0', 'encoded_2', 'encoded_7', 'encoded_4']
X = model_data[top_features]
y = model_data['Runs_Scored']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=1, callbacks=[early_stop])

y_pred = model.predict(X_test).flatten()
if np.isnan(y_pred).any():
    raise ValueError("🚫 TensorFlow model output contains NaN.")
print(f"✅ TF MLP MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"✅ TF MLP R²: {r2_score(y_test, y_pred):.2f}")

# --- Predict Today’s Games ---
today_games = pd.read_csv('Leagues/MLB/data/today_pitchers.csv')
today_games['Date'] = pd.to_datetime(today_games['Date'])

# Get latest pitcher data
latest_pitch = game_logs.sort_values('Date').dropna(subset=['Player'])
latest_pitch = latest_pitch.groupby('Player').last().reset_index()
latest_pitch['SO9'] = latest_pitch.apply(lambda row: (row['SO'] / row['IP'] * 9) if row['IP'] > 0 else 0, axis=1)
latest_pitch['ERA_game'] = latest_pitch.apply(lambda row: (row['ER'] / row['IP']) * 9 if row['IP'] > 0 else 0, axis=1)
latest_pitch['H_per9'] = latest_pitch.apply(lambda row: (row['H'] / row['IP']) * 9 if row['IP'] > 0 else 0, axis=1)
latest_pitch['days_rest'] = (pd.to_datetime('today') - latest_pitch['Date']).dt.days

# Merge pitcher and team features
# Merge pitcher and team features
merged = today_games.merge(latest_pitch[['Player', 'SO9', 'ERA_game', 'H_per9', 'days_rest']], on='Player', how='left')
encoded_data = today_games.apply(lambda row: find_latest_span(row['Team'], row['Date']), axis=1)
encoded_features = encoded_data[[f'encoded_{i}' for i in range(8)]].reset_index(drop=True)

# Combine and fill missing values
features = pd.concat([merged.reset_index(drop=True), encoded_features], axis=1)
features[top_features] = features[top_features].fillna(0)  # KEEP all teams

# Predict
X_today = features[top_features]
X_today_scaled = scaler.transform(X_today)
features['Predicted_Runs'] = model.predict(X_today_scaled).flatten()

# Create matchup key (sorted team names)
features['Matchup'] = features.apply(
    lambda row: ' vs '.join(sorted([row['Team'], row['Opponent']])), axis=1
)

# Group to one row per game
game_results = features.sort_values('Matchup').groupby(['Date', 'Matchup']).apply(
    lambda g: pd.Series({
        'Team_1': g.iloc[0]['Team'],
        'Runs_1': g.iloc[0]['Predicted_Runs'],
        'Team_2': g.iloc[1]['Team'],
        'Runs_2': g.iloc[1]['Predicted_Runs'],
        'Total_Runs': g.iloc[0]['Predicted_Runs'] + g.iloc[1]['Predicted_Runs']
    })
).reset_index()

# Final output
print(game_results.sort_values('Total_Runs', ascending=False))


import argparse
import re
import sys

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

from Leagues.MLB.gsheets_logger import runs_to_gsheets
from Leagues.MLB.odds_api import get_dk_lines
from Leagues.MLB.stats.bins import accuracy_bins
from Leagues.MLB.utils.logger import setup_logger


logger = setup_logger(name='Runs', log_dir="Leagues/MLB/logs/Runs")
parser = argparse.ArgumentParser(description='MLB Total Runs Prediction')
parser.add_argument('-sb', required=False, choices=['draftkings', 'fanduel'], help='Check prefered sportsbook')
parser.add_argument('-gs', required=False, type=bool, help='Post to google sheets')
args = parser.parse_args()
sportbook = args.sb
isgsheets = args.gs

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
sp_logs = setup_logger(name='Today_Pitchers', log_dir="Leagues/MLB/logs/StartingPitchers")
sp_logs.info(today_games)
today_games['Date'] = pd.to_datetime(today_games['Date'])

# Get latest pitcher data
latest_pitch = game_logs.sort_values('Date').dropna(subset=['Player'])
latest_pitch = latest_pitch.groupby('Player').last().reset_index()
latest_pitch['SO9'] = latest_pitch.apply(lambda row: (row['SO'] / row['IP'] * 9) if row['IP'] > 0 else 0, axis=1)
latest_pitch['ERA_game'] = latest_pitch.apply(lambda row: (row['ER'] / row['IP']) * 9 if row['IP'] > 0 else 0, axis=1)
latest_pitch['H_per9'] = latest_pitch.apply(lambda row: (row['H'] / row['IP']) * 9 if row['IP'] > 0 else 0, axis=1)
latest_pitch['days_rest'] = (pd.to_datetime('today') - latest_pitch['Date']).dt.days

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
# --- Robust matchup + game_results creation ---
# Detect opponent column name (today_games may have 'Opponent' or 'Opp')
if 'Opponent' in features.columns:
    opp_col = 'Opponent'
elif 'Opp' in features.columns:
    opp_col = 'Opp'
else:
    raise ValueError("No opponent column found in 'features'. Expected 'Opponent' or 'Opp'.")

# Create matchup key consistently (alphabetical order so both teams map to same matchup string)
features['Matchup'] = features.apply(
    lambda row: ' vs '.join(sorted([str(row['Team']), str(row[opp_col])])),
    axis=1
)

# Optional: inspect group sizes to find problems
# print(features.groupby('Matchup').size().value_counts())

def make_game_result(g):
    # Ensure deterministic ordering: alphabetical by team name (same logic used to create Matchup)
    teams = list(map(str, g['Team'].tolist()))
    runs = g['Predicted_Runs'].tolist()

    if len(teams) >= 2:
        # pick two unique teams (if duplicates exist, we still pick the first two alphabetical unique teams)
        # create tuples and sort by team name so ordering is consistent
        paired = sorted(zip(teams, runs), key=lambda t: t[0])
        team1, runs1 = paired[0]
        team2, runs2 = paired[1]
        total = float(runs1) + float(runs2)
        return pd.Series({
            'Team_1': team1,
            'Runs_1': runs1,
            'Team_2': team2,
            'Runs_2': runs2,
            'Total_Runs': total
        })
    elif len(teams) == 1:
        # missing opponent row — return a single-sided result so downstream code doesn't crash
        return pd.Series({
            'Team_1': teams[0],
            'Runs_1': runs[0],
            'Team_2': None,
            'Runs_2': np.nan,
            'Total_Runs': float(runs[0])  # or np.nan if you prefer
        })
    else:
        # empty group — shouldn't happen, return empties
        return pd.Series({
            'Team_1': None,
            'Runs_1': np.nan,
            'Team_2': None,
            'Runs_2': np.nan,
            'Total_Runs': np.nan
        })

game_results = features.groupby(['Date', 'Matchup']).apply(make_game_result).reset_index()


if sportbook is not None:
    print(f"🔍 Checking {sportbook.capitalize()} lines for today's games...")
    dk_lines = get_dk_lines(sportbook=sportbook)

    if dk_lines is None:
        fallback = 'draftkings' if sportbook == 'fanduel' else 'fanduel'
        print(f"🚫 No {sportbook.capitalize()} lines found. Trying {fallback.capitalize()}...")
        dk_lines = get_dk_lines(sportbook=fallback)

    if dk_lines is None:
        print("🚫 No DraftKings or FanDuel lines found. Continuing without lines.")
    else:
        print("✅ Sportsbook lines found for today's games.")
        # Create matchup keys for merging
        game_results['Matchup_Key'] = game_results.apply(
            lambda row: frozenset([row['Team_1'], row['Team_2']]), axis=1
        )
        dk_lines['Matchup_Key'] = dk_lines.apply(
            lambda row: frozenset([row['home_team'], row['away_team']]), axis=1
        )

        # Merge DK lines
        game_results = game_results.merge(dk_lines, on='Matchup_Key', how='left')
        game_results = game_results.fillna("")  # Fill missing DK lines with empty string for manual review
        # dk_diff = np.abs(game_results['DK_Line'] - game_results['Total_Runs'])
        # game_results['DK_Diff'] = dk_diff
        # bins = accuracy_bins()
else:
    print("🚫 No sportsbook specified. Continuing without DK lines.")

if isgsheets:
    print("📊 Posting results to Google Sheets...")
    try:
        runs_to_gsheets(game_results, sheetname="Runs")
    except Exception as e:
        print(f"🚫 Error posting to Google Sheets: {e} \n printing results:")
        print(game_results.sort_values('Total_Runs', ascending=False))
            
else:
    print("📊 Skipping Google Sheets post as per user request.")


logger.info(game_results.sort_values('Total_Runs', ascending=False))

print(game_results.sort_values('Total_Runs', ascending=False))
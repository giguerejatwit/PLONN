def compute_pace(stats: dict, opp_stats: dict, suffix: str = "") -> float:
    fga = stats.get(f'FGA{suffix}', 0)
    fta = stats.get(f'FTA{suffix}', 0)
    fg_pct = stats.get(f'FG%{suffix}', 0)
    fg = fg_pct * fga

    orb = stats.get(f'ORB{suffix}', 0)
    trb = stats.get(f'TRB{suffix}', 0)
    tov = stats.get(f'TOV{suffix}', 0)

    # Defensive rebounds for the opponent
    drb_opp = opp_stats.get(f'TRB{".1" if suffix == "" else ""}', 0) - opp_stats.get(f'ORB{".1" if suffix == "" else ""}', 0)
    orb_denom = orb + drb_opp
    orb_factor = orb / orb_denom if orb_denom else 0

    pace = fga + 0.4 * fta - 1.07 * orb_factor * (fga - fg) + tov
    return pace
import argparse
import sys
import time
import os
from datetime import datetime
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from bs4 import BeautifulSoup
from ghsheets_logger import data_to_googlesheets
from keras import callbacks, layers
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import Huber
from utils.odds_api import get_nba_dk_lines
from get_data import get_team_per_game_stats, get_today_games

from utils.abbr_map import TEAM_MAP

def to_team_abbr(name_or_abbr: str) -> str:
    """Normalize an input string (full name or abbr) to a team abbreviation.

    Resolution order:
      1) If it already matches a Team value in the TPG sheet, return as uppercased.
      2) If it matches a TEAM_MAP key (assumed to be an abbreviation in some setups), return that key uppercased.
      3) If it matches a TEAM_MAP value (full name), return the corresponding key.
      4) Fallback: best-effort uppercased input.
    """
    if not name_or_abbr:
        return ""

    s = str(name_or_abbr).strip()
    u = s.upper()

    # 1) Check directly against TPG 'Team' column (abbreviations)
    tpg_df = load_tpg_table()
    if 'Team' in tpg_df.columns:
        tpg_abbrs = tpg_df['Team'].astype(str).str.strip().str.upper().unique()
        if u in tpg_abbrs:
            return u

    # 2) If TEAM_MAP keys are abbreviations, use them
    if u in TEAM_MAP:
        return u

    # 3) Try to match full name against TEAM_MAP values
    lower = s.lower()
    for k, v in TEAM_MAP.items():
        if str(v).strip().lower() == lower:
            return str(k).upper()

    # 4) Fallback: best-effort
    return u


print("TensorFlow version:", tf.__version__)

# Global cache for TPG (team per-game) table
_TPG_TABLE = None

def load_tpg_table(path: str = 'leagues/NBA/data/tpgOct26.xlsx', sheet: str = 'TPG') -> pd.DataFrame:
    """Lazy-load the TPG sheet that contains per-team season averages.

    Expected columns include at least:
      - 'Team' (abbreviation, e.g. OKC, LAL, BOS)
      - ORtg, DRtg, eFG%, 2P%, 3P%, FT%, FG%, FGA, 3PA, ORB, TRB,
        AST, TOV, STL, PF, FTA, FT%, PTS, etc.
    """
    global _TPG_TABLE
    if _TPG_TABLE is not None:
        return _TPG_TABLE

    try:
        df = pd.read_excel(path, sheet_name=sheet, header=0)
        _TPG_TABLE = df
    except Exception as e:
        print(f"[WARN] Failed to load TPG sheet from {path} ({sheet}): {e}")
        _TPG_TABLE = pd.DataFrame()
    return _TPG_TABLE


def get_tpg_stats(team_abbr: str) -> dict:
    """Return a dict of per-team stats for the given team abbreviation
    using the TPG sheet. If the team or sheet is missing, return zeros
    for the expected stat keys.
    """
    df = load_tpg_table()

    # Columns we want to feed into the matchup vector (home side).
    # Away side will be suffixed with '.1' later.
    wanted_cols = [
        'PTS', 'FG%', 'FGA', '3P%', '3PA', 'ORB', 'TRB',
        'AST', 'TOV', 'STL', 'PF', 'ORtg', 'DRtg', 'FTA', 'FT%'
    ]

    if df.empty or 'Team' not in df.columns:
        print("[WARN] TPG sheet empty or missing 'Team' column; returning zeros.")
        return {col: 0.0 for col in wanted_cols}

    abbr = str(team_abbr).strip().upper()
    mask = df['Team'].astype(str).str.strip().str.upper() == abbr
    rowset = df[mask]

    if rowset.empty:
        print(f"[WARN] TPG: no row found for team '{team_abbr}'")
        return {col: 0.0 for col in wanted_cols}

    row = rowset.iloc[0]
    stats = {}
    for col in wanted_cols:
        val = row[col] if col in row else 0.0
        if pd.isna(val):
            val = 0.0
        stats[col] = float(val)
    return stats


parser = argparse.ArgumentParser(
    prog='PLONN-NBA1.0',
    description='This model takes basic data nba data, and computes TeamA score when matched up with TeamB'
)
parser.add_argument('-g', '--graph', action='store_true',
                    help="Enable graph mode")
parser.add_argument('-t', '--train', action='store_true',
                    help="Enable training mode")
parser.add_argument('-m', '--model', type=str,
                    help="select model 'adv' | '30dw'")
args = parser.parse_args()

feature_columns = [
    'PTS.1',
    'FG%', 'FG%.1', 'FGA', 'FGA.1',
    '3P%', '3P%.1', '3PA', '3PA.1',
    'ORB', 'ORB.1', 'TRB', 'TRB.1',
    'AST', 'AST.1', 'TOV', 'TOV.1',
    'STL', 'STL.1', 'PF', 'PF.1',
    'ORtg', 'ORtg.1', 'DRtg', 'DRtg.1',
    'FT%', 'FT%.1', 'FTA', 'FTA.1',
    'Pace', 'Pace.1'
]



def calculate_team_pace(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # Estimate FG if missing
    if 'FG' not in data.columns:
        data['FG'] = data['FG%'] * data['FGA']
    if 'FG.1' not in data.columns:
        data['FG.1'] = data['FG%.1'] * data['FGA.1']

    # Defensive rebounds
    data['DRB'] = data['TRB'] - data['ORB']
    data['DRB.1'] = data['TRB.1'] - data['ORB.1']

    # Safe possessions calculation
    team_possessions = (
        data['FGA']
        + 0.4 * data['FTA']
        - 1.07 * (data['ORB'] / (data['ORB'] + data['DRB.1']).replace(0, np.nan)) * (data['FGA'] - data['FG'])
        + data['TOV']
    )

    opp_possessions = (
        data['FGA.1']
        + 0.4 * data['FTA.1']
        - 1.07 * (data['ORB.1'] / (data['ORB.1'] + data['DRB']).replace(0, np.nan)) * (data['FGA.1'] - data['FG.1'])
        + data['TOV.1']
    )

    data['Pace'] = team_possessions
    data['Pace.1'] = opp_possessions

    return data


def clean_data(data=None) -> None:
    if not data:
        data = pd.read_csv('leagues/NBA/data/gamelogs2022_2024.csv', header=1)

    data = data.loc[:, ~data.columns.str.contains('Unnamed')]
    data = data.rename(columns={'Tm': 'PTS', data.columns[7]: 'PTS.1'})
    data = data[data['Rk'].apply(lambda x: str(x).isdigit())]

    data.reset_index(drop=True, inplace=True)

    data = calculate_team_pace(data)

    feature_columns = ['PTS.1',
                       'FG%', 'FG%.1',
                       'FGA', 'FGA.1',
                       '3P%', '3P%.1',
                       '3PA', '3PA.1',
                       'ORB', 'ORB.1',
                       'TRB', 'TRB.1',
                       'AST', 'AST.1',
                       'TOV', 'TOV.1',
                       'STL', 'STL.1',
                       'PF', 'PF.1',
                       'Pace']

    data.to_csv('leagues/NBA/data/cleaned_gamelogs.csv', index=False)

    data = data.dropna(axis=0)
    print(f'Null Data:', data.isna().any().sum())

    features = data[feature_columns]
    target = data['PTS']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.1)

    # Scale the features for better performandce
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # scaler_y = MinMaxScaler()  # Alternatively, StandardScaler()
    # y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    # y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    return X_train, X_test, y_train, y_test


def clean_adv(data=None):
    if data is None or data.empty:
        raise ValueError("clean_adv: expected a DataFrame with matchup rows")

    # Ensure required columns exist
    required_cols = ['Date', 'Team', 'Opp', 'PTS'] + feature_columns
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"clean_adv: missing required columns: {missing}")

    df = data.copy()

    # 1) Build Season label (October–April): e.g., Oct 2023–Apr 2024 => Season '2023-2024'
    df['Date'] = pd.to_datetime(df['Date'])
    month = df['Date'].dt.month
    year = df['Date'].dt.year
    season_start_year = np.where(month >= 10, year, year - 1).astype(int)
    season_end_year = season_start_year + 1
    start_s = pd.Series(season_start_year, index=df.index).astype(str)
    end_s = pd.Series(season_end_year, index=df.index).astype(str)
    df['Season'] = start_s + '-' + end_s

    # 2) 30-game rolling means per team within each season
    #    Home stats are the no-suffix columns. Away stats carry '.1'
    home_cols = [c for c in feature_columns if not c.endswith('.1')]
    away_cols = [c for c in feature_columns if c.endswith('.1')]

    # Sort by date so rolling windows are chronological
    df = df.sort_values('Date').reset_index(drop=True)

    # Count prior games within season (0-based); used to filter out tiny histories
    df['Team_games'] = df.groupby(['Team', 'Season']).cumcount()
    df['Opp_games']  = df.groupby(['Opp',  'Season']).cumcount()

    # Minimum prior games to trust a rolling average
    MINP = 10  # try 10; increase to 15 if you can afford fewer rows

    # Compute rolling means and shift by 1 to use only prior games
    def roll30(g, cols):
        g = g.sort_values('Date')
        r = g[cols].rolling(window=30, min_periods=MINP).mean().shift(1)
        return r

    # Home team 30-game means within season (now by 'Team')
    home_roll = df.groupby(['Team', 'Season'], group_keys=False).apply(lambda g: roll30(g, home_cols))
    # Away team 30-game means within season (now by 'Opp')
    away_roll = df.groupby(['Opp', 'Season'], group_keys=False).apply(lambda g: roll30(g, away_cols))

    # Inject rolled values back; align on index
    for c in home_cols:
        df[c] = home_roll[c]
    for c in away_cols:
        df[c] = away_roll[c]

    # Drop any rows that still have NaNs after shift (first game of a season/team etc.)
    features = df[feature_columns]
    target = df['PTS']
    mask = (
        (df['Team_games'] >= MINP) &
        (df['Opp_games']  >= MINP) &
        (~features.isnull().any(axis=1)) &
        (target.notnull())
    )

    # Materialize the averaged training rows for inspection/export
    cols_to_keep = ['Date', 'Team', 'Opp', 'Season', 'Team_games', 'Opp_games'] + feature_columns + ['PTS']
    averaged_training_df = df.loc[mask, cols_to_keep].copy()
    out_path = 'leagues/NBA/data/adv_training_30avg.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    averaged_training_df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote 30-game rolled training dataset → {out_path} ({len(averaged_training_df)} rows)")

    # Time-based split: last 20% by date is test
    dates_kept = df.loc[mask, 'Date']
    cutoff = dates_kept.quantile(0.8)
    train_idx = mask & (df['Date'] <= cutoff)
    test_idx  = mask & (df['Date'] >  cutoff)

    X_train = df.loc[train_idx, feature_columns]
    y_train = df.loc[train_idx, 'PTS']
    X_test  = df.loc[test_idx,  feature_columns]
    y_test  = df.loc[test_idx,  'PTS']

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[INFO] clean_adv: seasons={df['Season'].nunique()}, rows_kept={len(features)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train(X_train, y_train, input_size):
    model = Sequential([
        layers.Input(shape=(input_size,)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(64,  activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(16,  activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=Huber(delta=5.0), metrics=['mae'])

    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        shuffle=True,
        callbacks=[lr_scheduler, early_stopping]
    )
    if args.model == 'adv' or args.model == '30dw':
        model.save('leagues/NBA/models/adv_model.keras')
        print("Model saved as 'adv_model.keras'")
    else:
        model.save('leagues/NBA/models/raw_model.keras')
        print("Model saved as 'nba_model.keras'")

    return history, model


def create_matchup_feature_vector(home_team, away_team):
    """For a given matchup, fetch stats for the home and away teams and
    create a combined feature vector.

    We normalize both inputs to team abbreviations using
    `to_team_abbr`, so that lookups against TPG (which is keyed by
    abbreviations, e.g. CLE, PHI) succeed even if `get_today_games` returns
    full team names.
    """
    # Convert whatever we get (full name or abbr) into a standard abbreviation
    home_abbr = to_team_abbr(home_team)
    away_abbr = to_team_abbr(away_team)

    # For the advanced twin-vector model, pull per-team averages from the
    # TPG sheet. For other models, fall back to get_team_per_game_stats.
    if args.model in ("adv", "30dw"):
        home_stats = get_tpg_stats(home_abbr)
        away_stats = get_tpg_stats(away_abbr)
    else:
        home_stats = get_team_per_game_stats(home_abbr, args=args.model)
        away_stats = get_team_per_game_stats(away_abbr, args=args.model)

    # Rename away team keys with '.1' suffix
    away_stats = {f"{key}.1": value for key, value in away_stats.items()}

    # Compute Pace for both teams using helper
    try:
        home_stats['Pace'] = compute_pace(home_stats, away_stats, suffix="")
        away_stats['Pace.1'] = compute_pace(away_stats, home_stats, suffix=".1")
    except Exception as e:
        print(f"[WARN] Failed to compute pace: {e}")
        home_stats['Pace'] = 0
        away_stats['Pace.1'] = 0

    combined_stats = {**home_stats, **away_stats}
    return combined_stats


def get_home_vector(today_games):
    # Generate feature vectors for today's matchups
    feature_vectors = []
    for _, game in today_games.iterrows():
        features = create_matchup_feature_vector(
            game['home_team'], game['away_team']
        )

        feature_vectors.append(features)

    # Convert to a DataFrame for easy viewing
    home_feature_df = pd.DataFrame(feature_vectors)
    # print(home_feature_df.columns)

    # Filter the DataFrame to include only the columns your model expects
    home_feature_df = home_feature_df[feature_columns]

    return home_feature_df


def get_away_vector(today_games):
    feature_vectors = []
    for _, game in today_games.iterrows():
        features = create_matchup_feature_vector(
            game['away_team'], game['home_team']
        )

        feature_vectors.append(features)

    # Convert to a DataFrame for easy viewing
    away_feature_df = pd.DataFrame(feature_vectors)

    # Filter the DataFrame to include only the columns your model expects
    away_feature_df = away_feature_df[feature_columns]

    return away_feature_df


if __name__ == "__main__":

    print(f"Graph mode: {args.graph}\nTraining mode: {args.train}\nModel: {args.model}")
    print(f'Scraping data ⛏ ...')

    data = None
    X_train, X_test, y_train, y_test = None, None, None, None

    if args.model == 'adv' or args.model == '30dw':
        data = pd.read_excel('leagues/NBA/data/offsets/offset.xlsx',
                             sheet_name='Worksheet', header=0)
        
        data = data.dropna(axis=0)
        data = calculate_team_pace(data)
        X_train, X_test, y_train, y_test = clean_adv(data)

    else:
        X_train, X_test, y_train, y_test = clean_data(data)

    if args.train:
        history, model = train(X_train, y_train, X_train.shape[1])

        y_pred = model.predict(X_test)

        # Calculate performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f'Mean Absolute Error: {mae:.2f}')
        print(f'Root Mean Squared Error: {rmse:.2f}')

        if args.graph:
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.7)
            plt.scatter(range(len(y_test)), y_pred,
                        label='Predicted', alpha=0.7)
            plt.xlabel('Game Index')
            plt.ylabel('Points')
            plt.title('Actual vs Predicted Scores')
            plt.legend()
            plt.savefig('leagues/NBA/Data/images/scatterplot.png')
            plt.show()

    todays_games: pd.DataFrame = get_today_games()

    if todays_games.empty:
        print("No games today :( ")
        sys.exit(0)

    home = get_home_vector(todays_games)
    print("\n[DEBUG] Home feature head:")
    print(home.head())
    print("\n[DEBUG] Home feature describe:")
    print(home.describe())
    if not args.model:
        time.sleep(160)
    away = get_away_vector(todays_games)

    raw_model = None
    adv_model = None
    if args.model == 'adv' or args.model == '30dw':
        model_path = 'leagues/NBA/models/adv_model.keras'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}. Train with -m adv -t first.")
        adv_model = tf.keras.models.load_model(model_path)
    else:
        model_path = 'leagues/NBA/models/raw_model.keras'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}. Train without -m adv first to create it, or run with -m adv to use the advanced model.")
        raw_model = tf.keras.models.load_model(model_path)

    # loaded_model = model
    # Example: Make predictions using the loaded model

    scaler = StandardScaler()
    home_scaled_features = scaler.fit_transform(home)
    away_scaled_features = scaler.fit_transform(away)
    print('Scaled feature shape', home_scaled_features.shape)
    # Make predictions using the loaded model
    if args.model == 'adv':
        home_predicted_scores = adv_model.predict(home_scaled_features)
        away_predicted_scores = adv_model.predict(away_scaled_features)

        # Add the predicted scores to the original DataFrame
        todays_games['home_predicted_scores'] = home_predicted_scores.flatten()
        todays_games['away_predicted_scores'] = away_predicted_scores.flatten()


        # Replace full team names with abbreviations for clarity
        from utils.abbr_map import get_team_name_or_abbr

        todays_games['home_team'] = todays_games['home_team'].apply(lambda x: get_team_name_or_abbr(x))
        todays_games['away_team'] = todays_games['away_team'].apply(lambda x: get_team_name_or_abbr(x))

        # Fetch & merge DK totals into today's games
        try:
            _dk = get_nba_dk_lines()
            if _dk is not None and not _dk.empty:
                # Normalize expected column name
                if 'DK_Line' in _dk.columns and 'dk lines' not in _dk.columns:
                    _dk = _dk.rename(columns={'DK_Line': 'dk lines'})

                # Normalize names for reliable join
                tg = todays_games.copy()
                tg['__home'] = tg['home_team'].astype(str).str.strip().str.lower()
                tg['__away'] = tg['away_team'].astype(str).str.strip().str.lower()

                dk = _dk[['home_team', 'away_team', 'dk lines']].copy()
                dk['__home'] = dk['home_team'].astype(str).str.strip().str.lower()
                dk['__away'] = dk['away_team'].astype(str).str.strip().str.lower()

                tg = tg.merge(dk[['__home', '__away', 'dk lines']],
                              on=['__home', '__away'], how='left')
                tg = tg.drop(columns=['__home', '__away'])
                todays_games = tg
            else:
                print("[WARN] DK lines empty; continuing without dk lines.")
        except Exception as e:
            print(f"[WARN] Failed to fetch/merge DK lines: {e}")
        print("\nPredicted Scores for Today's Matchups:")
        print(todays_games)
        data_to_googlesheets(todays_games, 'Adv')

    elif args.model == '30dw':
        home_predicted_scores = adv_model.predict(home_scaled_features)
        away_predicted_scores = adv_model.predict(away_scaled_features)
        
        # Add the predicted scores to the original DataFrame
        todays_games['home_predicted_scores'] = home_predicted_scores.flatten()
        todays_games['away_predicted_scores'] = away_predicted_scores.flatten()
        
        print("\nPredicted Scores for Today's Matchups:")
        print(todays_games)
        data_to_googlesheets(todays_games, '30dw')

    else:
        home_predicted_scores = raw_model.predict(home_scaled_features)
        away_predicted_scores = raw_model.predict(away_scaled_features)

        # Add the predicted scores to the original DataFrame
        todays_games['home_predicted_scores'] = home_predicted_scores.flatten()
        todays_games['away_predicted_scores'] = away_predicted_scores.flatten()
        print("\nPredicted Scores for Today's Matchups:")
        print(todays_games)

        data_to_googlesheets(todays_games, 'Raw')
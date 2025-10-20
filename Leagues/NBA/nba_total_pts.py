import argparse
import sys
import time
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

from get_data import get_team_per_game_stats, get_today_games
from utils.abbr_map import TEAM_MAP, get_team_name_or_abbr

print("TensorFlow version:", tf.__version__)


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
                   ]

if args.model == 'adv' or args.model == '30dw':
    feature_columns = feature_columns + [
        'ORtg', 'ORtg.1',
        'DRtg', 'DRtg.1',
        'FT%', 'FT%.1',
        'FTA', 'FTA.1',
    ]


def clean_data(data=None) -> None:
    if not data:
        data = pd.read_csv('leagues/NBA/data/gamelogs2022_2024.csv', header=1)

    data = data.loc[:, ~data.columns.str.contains('Unnamed')]
    data = data.rename(columns={'Tm': 'PTS', data.columns[7]: 'PTS.1'})
    data = data[data['Rk'].apply(lambda x: str(x).isdigit())]

    data.reset_index(drop=True, inplace=True)

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
                       'PF', 'PF.1']

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
                       'ORtg', 'ORtg.1',
                       'DRtg', 'DRtg.1',
                       'FT%', 'FT%.1',
                       'FTA', 'FTA.1',
                       ]

    # data.to_csv('Data/cleaned_offset', index=False)
    features = data[feature_columns]
    target = data['PTS']
    print(features.isnull().sum())
    print(features[features.isnull().any(axis=1)])

    features = features.dropna(axis=0)  # Check for NaNs
    # print(np.isinf(features).sum())  # Check for infinite values

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.1)

    data[feature_columns] = data[feature_columns].apply(
        pd.to_numeric, errors='coerce')
    data['PTS'] = data['PTS'].apply(pd.to_numeric, errors='coerce')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train(X_train, y_train, input_size):

    model = Sequential(
        [
            layers.Input(shape=(input_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),

            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            # layers.Dropout(0.1),
            layers.Dense(1)  # Output layer for regression
        ]
    )

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='mse', metrics=['mae'])

    # Train the model
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5)
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=200, batch_size=8,
                        validation_split=0.2, shuffle=True, callbacks=[lr_scheduler, early_stopping])
    if args.model == 'adv' or args.model == '30dw':
        model.save('leagues/NBA/models/adv_model.keras')
        print("Model saved as 'adv_model.keras'")
    else:
        model.save('leagues/NBA/models/raw_model.keras')
        print("Model saved as 'nba_model.keras'")

    return history, model


def create_matchup_feature_vector(home_team, away_team, team_abbreviation_func):
    """
    For a given matchup, fetch stats for the home and away teams,
    and create a combined feature vector.
    """
    home_abbr = team_abbreviation_func(home_team)
    away_abbr = team_abbreviation_func(away_team)

    # Get stats for both teams
    home_stats = get_team_per_game_stats(home_abbr, args=args.model)
    away_stats = get_team_per_game_stats(away_abbr, args=args.model)

    # Rename away team columns with '.1' suffix
    away_stats = {f"{key}.1": value for key, value in away_stats.items()}

    # Combine home and away team stats
    combined_stats = {**home_stats, **away_stats}

    return combined_stats


def get_home_vector(today_games):
    # Generate feature vectors for today's matchups
    feature_vectors = []
    for _, game in today_games.iterrows():
        features = create_matchup_feature_vector(
            game['home_team'], game['away_team'], get_team_name_or_abbr
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
            game['away_team'], game['home_team'], get_team_name_or_abbr
        )

        feature_vectors.append(features)

    # Convert to a DataFrame for easy viewing
    away_feature_df = pd.DataFrame(feature_vectors)

    # Filter the DataFrame to include only the columns your model expects
    away_feature_df = away_feature_df[feature_columns]

    return away_feature_df


if __name__ == "__main__":
    args = parser.parse_args()

    print(f"Graph mode: {args.graph}\nTraining mode: {args.train}\nModel: {args.model}")
    print(f'Scraping data ⛏ ...')

    data = None
    X_train, X_test, y_train, y_test = None, None, None, None

    if args.model == 'adv' or args.model == '30dw':
        data = pd.read_excel('leagues/NBA/data/offsets/offset.xlsx',
                             sheet_name='Worksheet', header=0)
        
        data = data.dropna(axis=0)
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
    if not args.model:
        time.sleep(160)
    away = get_away_vector(todays_games)

    raw_model = tf.keras.models.load_model(
        'leagues/NBA/models/raw_model.keras')
    adv_model = tf.keras.models.load_model(
        'leagues/NBA/models/adv_model.keras')

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

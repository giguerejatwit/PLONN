import os
import sys
import time
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
import tensorflow as tf
from bs4 import BeautifulSoup

print("TensorFlow version:", tf.__version__)

import argparse

import gspread
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from google.oauth2.service_account import Credentials
from keras import callbacks, layers
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

parser = argparse.ArgumentParser(
    prog='PLONN-NBA1.0',
    description='This model takes basic data nba data, and computes TeamA score when matched up with TeamB'
)
parser.add_argument('-g', '--graph', action='store_true', help="Enable graph mode")
parser.add_argument('-t', '--train', action='store_true', help="Enable training mode")
parser.add_argument('-m', '--model', type=str, help="select model 'adv' | '30dw'")
args = parser.parse_args()

feature_columns = []
TEAM_MAP = None
if args.model == 'adv' or args.model == '30dw':
    TEAM_MAP = {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BRK",
        "Charlotte Hornets": "CHO",
        "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL",
        "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",
        "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHO",
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS"
    }
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
else:
    TEAM_MAP = {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BRK",
        "Charlotte Hornets": "CHO",
        "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL",
        "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",
        "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHO",
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS"
        }
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
ABBR_MAP = {abbr: team for team, abbr in TEAM_MAP.items()}

def get_team_name_or_abbr(input_str):
    """Convert a team name to its abbreviation or vice versa."""
    input_str = input_str.strip()
    
    # Check if the input is an abbreviation
    if input_str.upper() in ABBR_MAP:
        return ABBR_MAP[input_str.upper()]
    # Check if the input is a full team name
    elif input_str in TEAM_MAP:
        return TEAM_MAP[input_str]
    else:
        return "Team not found."

def get_train_data() -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    abreviations = list(TEAM_MAP.values()) #list of all abreviations
    years = range(2023, 2026)
    teams_data = pd.DataFrame()
    for year in years:
        for abrv in abreviations:
            url = f'https://www.basketball-reference.com/teams/{abrv}/{year}/gamelog/'

            try:
                r = requests.get(url, headers=headers)
                r.raise_for_status()
            except Exception as e:
                print(f'Could not fetch {abrv}/{year}')
                r.raise_for_status()

            soup = BeautifulSoup(r.text, 'html.parser')
            table = soup.find('table', {'id': 'tgl_basic'})


            if table:
                table = StringIO(str(table))
                team = pd.read_html(table)[0]

                teams_data = pd.concat([teams_data, team], ignore_index=True)

        time.sleep(180)
    
    return teams_data

def clean_data(data=None) -> None:
    if not data:
        data = pd.read_csv('data/gamelogs2022_2024.csv', header=1)

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

    data.to_csv('Data/cleaned_gamelogs.csv', index=False)

    data = data.dropna(axis=0)
    print(f'Null Data:', data.isna().any().sum())

    features = data[feature_columns]
    target = data['PTS']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1)

    # Scale the features for better performandce
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    scaler_y = MinMaxScaler()  # Alternatively, StandardScaler()
    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    

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

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1)

    data[feature_columns] = data[feature_columns].apply(pd.to_numeric, errors='coerce')
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
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

    # Train the model
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=200, batch_size=8, validation_split=0.2, shuffle=True, callbacks=[lr_scheduler, early_stopping])
    if args.model == 'adv' or '30dw':
        model.save('models/adv_model.keras')
    else:
        model.save('models/raw_model.keras')
    print("Model saved as 'PLONN1-0.keras'")
    
    return history, model

def get_today_games():
    
    month = datetime.now().strftime('%B').lower()
    year = datetime.now().strftime('%Y')

    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        print(e)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the schedule table
    schedule_table = soup.find('table', {'id': 'schedule'})
    if not schedule_table:
        print("Schedule table not found.")
        return pd.DataFrame()
    
    # Extract rows from the table
    rows = schedule_table.find_all('tr')
    
    # Get today's date in the format used on the website
    today = datetime.now().strftime('%a, %b %-d, %Y')
    print(today)
    tmrw = "Thu, Feb 20, 2025"
    # Initialize a list to hold today's games
    games_today = []
    
    for row in rows:
        # Extract the date cell
        date_cell = row.find('th', {'data-stat': 'date_game'})
        if date_cell and date_cell.text.strip() == today:
            # Extract team names
            away_team = row.find('td', {'data-stat': 'visitor_team_name'}).text.strip()
            home_team = row.find('td', {'data-stat': 'home_team_name'}).text.strip()
            games_today.append({'home_team': home_team, 'away_team': away_team})

        if date_cell and date_cell.text.strip() == tmrw:
            # Extract team names
            away_team = row.find('td', {'data-stat': 'visitor_team_name'}).text.strip()
            home_team = row.find('td', {'data-stat': 'home_team_name'}).text.strip()
            games_today.append({'home_team': home_team, 'away_team': away_team})
    
    
    # Convert the list to a DataFrame
    return pd.DataFrame(games_today)

def get_team_per_game_stats(team_abbr):
    
    """
    Scrapes the 'per_game' table for a given team using its abbreviation.
    """
    if args.model == 'adv':
        
        try:
            data = pd.read_excel('data/tpg.xlsx', sheet_name='Worksheet', header=0)
            data = data[data['Team'] == team_abbr]

            # Extract only the required columns
            feature_columns = ['PTS', 'FG%', 'FGA', '3P%', '3PA', 'ORB', 'TRB', 
                               'AST', 'TOV', 'STL', 'PF', 'ORtg', 'DRtg', 'FTA', 'FT%']

            # If there is no matching data, return zeros
            if data.empty:
                return {col: 0 for col in feature_columns}

            # Extract scalar values
            team_stats = data.iloc[0]  # Assume the first matching row is correct
            return {col: team_stats.get(col, 0) for col in feature_columns}
        except Exception as e:
            print('Error in get_team_per_game_stats(): Adv model:', e)
            pass
    
    elif args.model == '30dw':
        data = pd.read_csv('data/window/nbaAvgWindow.csv', header=0)
        data = data[data['Team_Name'] == team_abbr]

        # Extract only the required columns
        feature_columns = ['PTS', 'FG%', 'FGA', '3P%', '3PA', 'ORB', 'TRB', 
                        'AST', 'TOV', 'STL', 'PF', 'ORtg', 'DRtg', 'FTA', 'FT%']
        
        # If there is no matching data, return zeros
        if data.empty:
            return {col: 0 for col in feature_columns}
        
        # Extract scalar values
        team_stats = data.iloc[0]  # Assume the first matching row is correct
        return {col: team_stats.get(col, 0) for col in feature_columns}
    else:
        url = f"https://www.basketball-reference.com/teams/{team_abbr}/2025.html"

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Locate the per-game table
        per_game_table = soup.find('table', {'id': 'per_game_stats'})
        if not per_game_table:
            print(f"Per-game table not found for {team_abbr}.")
            return {}
        if per_game_table:
            table = StringIO(str(per_game_table))
            df = pd.read_html(table)[0]

            # Read the table using pandas


            # Extract the row corresponding to 'Team Totals'
            team_totals_row = df[df['Player'] == 'Team Totals']
            if team_totals_row.empty:
                print(f"No 'Team Totals' row found for {team_abbr}.")
                return {}

            # Convert the row to a dictionary
            team_stats = team_totals_row.iloc[0].to_dict()

            # Extract only the required columns
            feature_columns = ['PTS', 'FG%', 'FGA', '3P%', '3PA', 'ORB', 'TRB', 
                               'AST', 'TOV', 'STL', 'PF']

            # Map to desired format
            relevant_stats = {col: team_stats.get(col, 0) for col in feature_columns}

            return relevant_stats

def create_matchup_feature_vector(home_team, away_team, team_abbreviation_func):
    """
    For a given matchup, fetch stats for the home and away teams,
    and create a combined feature vector.
    """
    home_abbr = team_abbreviation_func(home_team)
    away_abbr = team_abbreviation_func(away_team)
    
    # Get stats for both teams
    home_stats = get_team_per_game_stats(home_abbr)
    away_stats = get_team_per_game_stats(away_abbr)
    
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
            game['home_team'],game['away_team'],get_team_name_or_abbr 
        )


        feature_vectors.append(features)

    # Convert to a DataFrame for easy viewing
    home_feature_df = pd.DataFrame(feature_vectors)
    # print("Feature Vectors for Today's Matchups:")
    # print(home_feature_df.columns)

    # Filter the DataFrame to include only the columns your model expects
    home_feature_df = home_feature_df[feature_columns]
    
    return home_feature_df
    
def get_away_vector(today_games):
    feature_vectors = []
    for _, game in today_games.iterrows():
        features = create_matchup_feature_vector(
            game['away_team'],game['home_team'],get_team_name_or_abbr 
        )


        feature_vectors.append(features)

    # Convert to a DataFrame for easy viewing
    away_feature_df = pd.DataFrame(feature_vectors)

    # Filter the DataFrame to include only the columns your model expects
    away_feature_df = away_feature_df[feature_columns]

    return away_feature_df
    
def data_to_googlesheets(data, sheet_name='Raw') -> None:
    
    """
    Add data into the NBA google spread sheet
    """

    scopes = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    
    creds = Credentials.from_service_account_file("credentials.json", scopes=scopes)
    client = gspread.authorize(creds)
    
    #Raw Sheet ID
    sheet_id = '1ClAB4iwIF-C12cty1DYZiH0o8G6kYTV1b1GvpaS2hWk'
    sheet = client.open_by_key(sheet_id)
    raw_sheet = sheet.worksheet(sheet_name)
    
    last_row = len(raw_sheet.get_all_values())  # Count columns based on the first row
    raw_sheet.insert_row([""] * len(raw_sheet.row_values(1)), last_row + 1)
    
    todays_games_formatted = pd.DataFrame({
        "date": [datetime.today().strftime("%m/%d/%Y")] * len(data),  # Add today's date
        "home": data["home_team"],
        "away": data["away_team"],
        "home pred": data["home_predicted_scores"],  # Round scores
        "away pred": data["away_predicted_scores"],
        # "pred total": (todays_games["home_predicted_scores"] + todays_games["away_predicted_scores"]),
    })
    data_to_upload = todays_games_formatted.values.tolist()
    raw_sheet.insert_rows(data_to_upload, row=last_row + 2)
    print(f'✅ Successfully appended `todays_games` predictions to the {sheet_name} sheet!')

if __name__ == "__main__":
    args = parser.parse_args()

    print(f"Graph mode: {args.graph}\nTraining mode: {args.train}\nModel: {args.model}")
    print(f'Scraping data ⛏ ...')
    
    data = None
    X_train, X_test, y_train, y_test = None, None, None, None

    if args.model == 'adv' or args.model == '30dw':
        data = pd.read_excel('data/offsets/offset.xlsx',sheet_name='Worksheet', header=0)
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
            plt.scatter(range(len(y_test)), y_pred, label='Predicted', alpha=0.7)
            plt.xlabel('Game Index')
            plt.ylabel('Points')
            plt.title('Actual vs Predicted Scores')
            plt.legend()
            plt.savefig('Data/images/scatterplot.png')
            plt.show()

    
    todays_games = get_today_games()
    
    home = get_home_vector(todays_games)
    if not args.model:
        time.sleep(160)
    away = get_away_vector(todays_games)

    
    raw_model = tf.keras.models.load_model('models/nba_model.keras')
    adv_model = tf.keras.models.load_model('models/adv_model.keras')
    
    
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

        # TODO: Add distribution
        data_to_googlesheets(todays_games, 'Raw')

    
    
    



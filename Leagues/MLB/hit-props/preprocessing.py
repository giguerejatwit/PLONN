import argparse
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
from MLB.StatsApi import *
from MLB.team_names import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(
    prog="preprocessing.py", usage='Fetches data for selected test_df')
parser.add_argument('-t', '--team', required=False,
                    type=str, help='team abriviation')
args = parser.parse_args()
team_name = args.team


# Features for training. TODO: check for overfitting
features = ['SO', 'AB', 'SLG', 'OPS', 'GEOM.BA', 'BA', 'H']

target = ['H/G']  # Target Variable

def get_geom_prob(batters: pd.DataFrame):
    """returns geometric probability"""
    geom_p = stats.geom.cdf(k=4, p=batters['BA'])
    print(geom_p)
    return geom_p

def get_training_data():
    """Get training data using all active MLB players in-season *uses season data*."""
    # Get all players data currently in MLB
    df = get_all_player_batting()
    
    
    if df.empty: # Check if df is empty
        print("No training data available")
        return None, None

    # Calculate Geom probability, handling zero or invalid BA values
    if 'BA' in df.columns:
        # Replace zero or negative BA values with a small positive value to avoid issues
        df['BA'] = df['BA'].apply(lambda x: max(x, 1e-6))
        
        def safe_geom_cdf(p): # Safe computation of GEOM.BA
            if p <= 0 or p >= 1:
                return np.nan  # Return NaN for invalid probabilities
            return stats.geom.cdf(4, p)
        
        df['GEOM.BA'] = df['BA'].apply(safe_geom_cdf)
        
    # Calculate hits per game - Target Variable
    df['H/G'] = df['H'] / df['G']

    # Clean rows of NaN values
    df.dropna(inplace=True)

    # Check if all required features are present
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        print(f"Warning: Missing features in training data: {missing_features}")
        return None, None

    X_train = df[features]
    y_train = df[target]
    
    # Scale stats
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train, y_train

def load_last_3_games() -> pd.DataFrame:
    file_path = "Leagues/MLB/Data/last3.csv"
    if os.path.exists(file_path):
        df = load_data(file_path)
    else: #fetch the data and save it
        df = pd.DataFrame(get_all_last_3_games())
                
        save_data(data=df,file_path=file_path) #save data after use
        return df

def get_test_last_3_games(team: str):
    df = pd.DataFrame(get_team_last_3_games(team=team))
    df.dropna(subset='BA', inplace=True)
    
    
    Names = df['Name']

    if 'BA' in df.columns:
        # Replace zero or negative BA values with a small positive value to avoid issues
        df['BA'] = df['BA'].apply(lambda x: max(x, 1e-6))
        
        # Safe computation of GEOM.BA using the geometric CDF function, applied row by row
        def compute_geom_ba(row):
            BA = row['BA']
            G = row['G']
            AB_per_game = np.ceil(row['BA'] / row['G'])  # Adjust as necessary based on your logic
            if 0 < BA < 1:
                return stats.geom.cdf(AB_per_game, BA)
            return 0
        
        # Apply the function row by row to compute 'GEOM.BA'
        df['GEOM.BA'] = df.apply(compute_geom_ba, axis=1)
        
        # Drop rows where 'GEOM.BA' couldn't be computed
        df.dropna(subset=['GEOM.BA'], inplace=True)
    
    df.dropna(inplace=True)
    df['H/G'] = df['H'] / df['G']

    X = df[features]
    y = df['H/G']

    # Scale stats
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X)
    y_test = y
    
    return X_test, y_test, Names

def get_train_last_3_games(excluded_team: str):
    
    X_train_list = []
    y_train_list = []
    Names_list = []
    
    df = pd.DataFrame(get_all_last_3_games())
    # Exclue test team
    df = df[df['Tm'] != excluded_team]
    
    df.dropna(inplace=True)
    
    Names = df['Name']
    
    if 'BA' in df.columns:
        # Convert 'BA' to numeric and handle non-numeric values by setting errors='coerce' to replace them with NaN
        df['BA'] = pd.to_numeric(df['BA'], errors='coerce')
        
        # Drop rows where 'BA' is NaN
        df.dropna(subset=['BA'], inplace=True)

        # Replace zero or negative BA values with a small positive value to avoid issues
        df['BA'] = df['BA'].apply(lambda x: max(x, 1e-6))
        
        # Safe computation of GEOM.BA using the geometric CDF function
        df['GEOM.BA'] = df['BA'].apply(lambda p: stats.geom.cdf(4, p) if 0 < p < 1 else np.nan)
        df.dropna(subset=['GEOM.BA'], inplace=True)
        
        # Calculate hits per game - Target Variable
        df['H/G'] = df['H'] / df['G']
        X = df[features]
        y = df['H/G']

        # Append data to lists
        X_train_list.append(X)
        y_train_list.append(y)
        Names_list.extend(Names)

    # Combine all teams data
    X_train = pd.concat(X_train_list, axis=0)
    y_train = pd.concat(y_train_list, axis=0)

    # Check for columns with sequences
    for col in X_train.columns:
        if X_train[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
            print(f"Column '{col}' contains sequences.")

    # Scale stats
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    
    return X_train, y_train
    
def get_test_yesterday_data(team: str):
    df = pd.DataFrame(get_last_game(team=team))
    # Names 
    Names = df['Name']
    # create batting average 
    df['BA'] = df['H'] / df['AB']
    # create geometric ba 
    df['GEOM.BA'] = stats.geom.cdf(k=4, p=df['BA'])
    
    # need to make sure the data is assignetd to the right payer 
    df['1B'] = df['H'] - df['2B'] + df['3B'] +df['HR']
    
    # Calculate total bases
    df['Total Bases'] = df['1B'] + (2 * df['2B']) + (3 * df['3B']) + (4 * df['HR'])

    # Calculate SLG
    df['SLG'] = df['Total Bases'] / df['AB']
    
    # Calculate OBP
    df['OBP'] = (df['H'] + df['BB'] + df['HBP']) / (df['AB'] + df['BB'] + df['HBP'] + df['SF'])

    # Calculate OPS
    df['OPS'] = df['OBP'] + df['SLG']
    
    X = df[features]
    y = df['H']

    # Scale stats
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X)
    y_test = y


    return X_test, y_test, Names

def get_test_data(team: str):
    """this method gets team data, creates new features, cleans data"""
    team.upper()
    global test_df

    # get data for last 7 games played.
    test_df = pd.DataFrame(get_last_7_games(team=team))
    # Clean rows of NaN values
    test_df.dropna(inplace=True)
    
    test_df = test_df[test_df['Name'] != 'Team Total']
    if test_df.empty:
        print(f"No data available for team: {team}")
        return None, None, None

    # Calculate Geom probability, handling zero or invalid BA values
    if 'BA' in test_df.columns:
        test_df['BA'] = test_df['BA'].apply(lambda x: max(x, 1e-6))  # Avoid zero values
        test_df['GEOM.BA'] = stats.geom.cdf(k=4, p=test_df['BA'])
    else:
        print("Warning: 'BA' column not found in test data")
        test_df['GEOM.BA'] = 0  # Assign default value or handle as needed


    # Calculate hits per game - Target Variable
    if 'H' in test_df.columns and 'G' in test_df.columns:
        test_df['H/G'] = test_df['H'] / test_df['G']
    else:
        print("Warning: 'H' or 'G' column not found in test data")
        test_df['H/G'] = 0  # Assign default value or handle as needed        

    # Check if test_df is empty after dropping NaNs
    if test_df.empty:
        print("No valid data available after cleaning")
        return None, None, None

    # List of batters names to better display results
    Names = test_df['Name']

    # Check if all required features are present
    missing_features = [feature for feature in features if feature not in test_df.columns]
    if missing_features:
        print(f"Warning: Missing features in test data: {missing_features}")
        return None, None, None

    X = test_df[features]
    y = test_df['H/G']

    # Scale stats
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X)
    y_test = y

    return X_test, y_test, Names

def save_data(data: pd.DataFrame, file_path: str):
    data.to_csv(file_path)
    
def load_data(file_path: str) -> pd.DataFrame:
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
        print(df)
    else:
        raise FileNotFoundError
    return df

if __name__ == '__main__':  
    pass
    # get_test_last_3_games("BOS")

import pandas as pd
import numpy as np

import tensorflow as tf
import keras


"""
import 5game span encoded data
import game logs
"""

def clean_dates(df, column):
    df[column] = df[column].astype(str).str.replace(r"\s*\(.*\)", "", regex=True)
    df[column] = pd.to_datetime(df[column],errors='coerce')
    return df

def map_opponent_encoding(gamelog, encoded_opponents):
    # Make a copy to avoid modifying original
    gamelog = gamelog.copy()
    
    # Create an empty list to collect encodings
    matched_encodings = []
    # Try this right before the for-loop return to debug:
    sample = gamelog.iloc[0]
    opp = sample['Opp']
    gdate = sample['Date']
    
    print("DEBUG -- Opponent:", opp)
    print("DEBUG -- Game Date:", gdate)
    
    # Check if anything in batting_encoded matches
    eligible = batting_encoded[
        (batting_encoded['Team'] == opp) &
        (batting_encoded['Span Ended'] <= gdate)
    ]
    print("DEBUG -- Eligible matches found:", len(eligible))
    print(eligible[['Team', 'Span Ended']].tail())

    for _, row in gamelog.iterrows():
        opp_team = row['Opp']
        game_date = row['Date']

        # Filter encoded spans for the same opponent team and before the game date
        eligible_spans = encoded_opponents[
            (encoded_opponents['Team'] == opp_team) &
            (encoded_opponents['Span Ended'] <= game_date + pd.Timedelta(days=5))
        ]

        # If a match is found, take the most recent one
        if not eligible_spans.empty:
            latest = eligible_spans.sort_values(by='Span Ended').iloc[-1]
            
            matched_encodings.append(latest.filter(like='encoded_').values)
        else:
            matched_encodings.append([None] * 8)  # Or np.nan if you prefer

    # Convert to DataFrame and merge
    encoding_df = pd.DataFrame(matched_encodings, columns=[f'enc_{i}' for i in range(8)])
    return pd.concat([gamelog.reset_index(drop=True), encoding_df], axis=1)

def add_pitcher_rolling_features(df):
    df = df.copy()

    df['IP'] = pd.to_numeric(df['IP'], errors='coerce')  # just in case
    df['SO'] = pd.to_numeric(df['SO'], errors='coerce')

    # Sort by pitcher + date
    df = df.sort_values(by=['Player', 'Date'])

    # Rolling features
    df['K_prev'] = df.groupby('Player')['SO'].shift(1)
    df['K_avg_3'] = df.groupby('Player')['SO'].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    df['IP_avg_3'] = df.groupby('Player')['IP'].shift(1).rolling(3).mean().reset_index(level=0, drop=True)

    return df



# Load the data
batting_encoded = pd.read_csv(
    'Leagues/MLB/data/encoded_5game_spans.csv', header=0)

game_logs = pd.read_excel(
    'Leagues/MLB/data/2024/gamelog_2024.xlsx', header=0)


game_logs_ps = add_pitcher_rolling_features(game_logs)

batting_encoded = clean_dates(batting_encoded, 'Span Started')
batting_encoded = clean_dates(batting_encoded, 'Span Ended')
            

game_logs_ps = clean_dates(game_logs_ps, 'Date')
game_logs_ps = game_logs_ps[game_logs_ps['Date'] >= '2025-04-01']
gamelog_with_encodings = map_opponent_encoding(game_logs_ps, batting_encoded)

gamelog_with_encodings.to_csv(
    'Leagues/MLB/data/gamelog_final.csv', index=False)
    
    




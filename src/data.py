import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# print Bruins 2023 data
# print(nhl23_data.loc["Boston Bruins"])
nhl23_data = pd.read_excel('Data/NHL2023_Data.xlsx')
nhl23_data.set_index('Team', inplace=True)
nhl22_data = pd.read_excel('Data/NHL2022_Data.xlsx')
nhl21_data = pd.read_excel('Data/NHL2022_Data.xlsx')
matches2023 = pd.read_excel('Data/Matches2023.xlsx')
# nhl23_data.set_index('Team', inplace = True)

# Spread Data
spread_data = pd.read_excel('Data/SpreadOct4.xlsx')

num_goals = 6.5
TeamA = spread_data['TeamA']
TeamB = spread_data['TeamB']
over = spread_data['O']
under = spread_data['U']
TeamA_spread = spread_data['TeamA Spread']
TeamB_spread = spread_data['TeamB Spread']

# Data for each game played each game is independent of another.

GameA = pd.Series([TeamA[0], TeamB[0]])
GameB = pd.Series([TeamA[1], TeamB[1]])
GameC = pd.Series([TeamA[2], TeamB[2]])
GameD = pd.Series([TeamA[3], TeamB[3]])
# print(GameA, GameB)

# GF/G GA/G W/L%, SV%
# Data for 2023 season

# teams = nhl23_data['Team']
gfpg = nhl23_data['GF/G']
gapg = nhl23_data['GA/G']
w = nhl23_data['W']
l = nhl23_data['L']
svp = nhl23_data['SV%']

data23 = pd.concat([w, l, gapg, gfpg, svp], axis=1)
# print(data23)


# Data for 2022 season

teams22 = nhl22_data['Team']
gfpg22 = nhl22_data['GF/G']
gapg22 = nhl22_data['GA/G']
w22 = nhl22_data['W']
l22 = nhl22_data['L']
svp22 = nhl22_data['SV%']

data22 = pd.concat([teams22, w22, l22, gapg22, gfpg22, svp22], axis=1)
# print(data22)

# Data for 2021 season

teams21 = nhl21_data['Team']
gfpg21 = nhl21_data['GF/G']
gapg21 = nhl21_data['GA/G']
w21 = nhl21_data['W']
l21 = nhl21_data['L']
svp21 = nhl21_data['SV%']

data21 = pd.concat([teams21, w21, l21, gapg21, gfpg21, svp21], axis=1)
# print(data21)

# Get team data for each of the games
matchupA = data23.loc[GameA]
matchupB = data23.loc[GameB]
matchupC = data23.loc[GameC]
matchupD = data23.loc[GameD]

matches = pd.DataFrame([GameA,
                        GameB,
                        GameC,
                        GameD])
column_names = ["TeamA", "TeamB"]
matches.columns = column_names
# print(matches)
# print(matchupB)
# print(matchupC)
# print(matchupD)

# Get test and train data with 20:80 test:train


def flatten_row(row, team_data):
    teamA_features = team_data.loc[row['TeamA']].values

    teamB_features = team_data.loc[row['TeamB']].values
    return pd.Series(list(teamA_features) + list(teamB_features))

column_names = ["TeamA_Wins", "TeamA_Losses", "TeamA_GA/G", "TeamA_GF/G", "TeamA_SV%",
                "TeamB_Wins", "TeamB_Losses", "TeamB_GA/G", "TeamB_GF/G", "TeamB_SV%"]

flattened_matches = matches.apply(
    lambda row: flatten_row(row, data23), axis=1)

flattened_matches.columns = column_names


flattened_matches2023 = matches2023.apply(
    lambda row: flatten_row(row, data23), axis=1)

flattened_matches2023.columns = column_names
flattened_data = pd.concat(
    [flattened_matches2023, matches2023['Total Goals']], axis=1)


combined_data = pd.concat([flattened_data, flattened_matches], axis=0, ignore_index=True)

predict_data = combined_data.iloc[:4].drop("Total Goals", axis=1) #there are no total goals yet
historical_data = combined_data.iloc[4:1311]

X = historical_data.drop("Total Goals", axis=1)
y = historical_data["Total Goals"]

# Split data into train and test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Normalize training data
normal = StandardScaler().fit(X_train)
X_train_normal = normal.transform(X_train)
X_test_normal =  normal.transform(X_test)
print(y_test[y_test.isna()])
predict_data_normal = normal.transform(predict_data)
# Testing data: (263, 10) (263,)
print("Training data:", X_train_normal.shape, y_train.shape)
print("Testing data:", X_test_normal.shape, y_test.shape)
# Training data: (1049, 10) (1049,)

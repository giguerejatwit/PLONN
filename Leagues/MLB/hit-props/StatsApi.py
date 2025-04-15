from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
import datetime
from datetime import datetime, time

numeric_columns = ['Age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'BA', 'OBP', 'SLG', 'OPS', 'HBP', 'SH', 'SF', 'IBB']

def get_team_last_3_games(team):
    """Returns data for each player of their previous game"""
    team = team.upper()
    
    today_date = datetime.now().date()
    month = datetime.now().month
    year = str(datetime.now().year)
    if month < 10: 
        month = str(month)
        month = '0' + month
    else: month = str(month)
    day = datetime.now().day
    

    url = f'https://www.baseball-reference.com/leagues/daily.fcgi?request=1&type=b&dates=lastndays&lastndays=3&since={year}-{month}-01&fromandto={year}-{month}-01.{year}-{month}-31&level=mlb&franch={team}#daily'
    try:
        r = requests.get(url)
        r.raise_for_status()
    except Exception as e:  # Check if the request was successful
        print("could not fetch yesterdays games", e)
        r.raise_for_status()
    
    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.find('table', {'id': 'daily'})
    # Find the relevant table by its class or id
    table = StringIO(str(table))
    # Read the table into a pandas DataFrames
    df = pd.read_html(table)[0]
    
    df_cleaned = df[df.iloc[:, 1] != 'Rk'].reset_index(drop=True)
    
    df_cleaned[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    return df_cleaned
    
def get_all_last_3_games():
    """Returns data for all players of their previous 3 games"""
    today_date = datetime.now().date()
    month = datetime.now().month
    year = str(datetime.now().year)
    if month < 10: 
        month = str(month)
        month = '0' + month
    else: month = str(month)
    day = datetime.now().day
    

    url = f'https://www.baseball-reference.com/leagues/daily.fcgi?request=1&type=b&dates=lastndays&lastndays=3&since={year}-{month}-01&fromandto={year}-{month}-01.{year}-{month}-{day}&level=mlb&franch=ANY#daily'
    try:
        r = requests.get(url)
        
        
        r.raise_for_status()
    except Exception as e:  # Check if the request was successful
        print("could not fetch yesterdays games", e)
        if r.status_code == 429:
            print("Retry after:", int(r.headers['Retry-After']))
            time.sleep(int(r.headers['Retry-After']))
    
    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.find('table', {'id': 'daily'})
    # Find the relevant table by its class or id
    table = StringIO(str(table))
    # Read the table into a pandas DataFrames
    df = pd.read_html(table)[0]
    df_cleaned = df[df.iloc[:, 1] != 'Rk'].reset_index(drop=True)
    
    df_cleaned[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    return df_cleaned
    
def get_last_7_games(team):
    url = f'https://www.baseball-reference.com/tools/split_stats_team.cgi?full=1&params=total%7CLast%207%20days%7C{team}%7C2024%7Cbat%7CAB%7C#team_split1'
    try:
        r = requests.get(url)
        r.raise_for_status()
    except Exception as e:  # Check if the request was successful
        print("could not fetch MLB-Last7Games", e)
        r.raise_for_status()
    
    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.find('table', {'id': 'team_split1'})
    # Find the relevant table by its class or id
    table = StringIO(str(table))
    # Read the table into a pandas DataFrame
    stats = pd.read_html(table)[0]
    
    return stats

def get_standings():
    url = 'https://www.baseball-reference.com/leagues/MLB-standings.shtml#all_expanded_standings_overall'
    r = requests.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, 'html.parser')
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    for comment in comments:
        if 'id="expanded_standings_overall"' in comment:
            table_soup = BeautifulSoup(comment, 'html.parser')
            table = table_soup.find('table', {'id' : 'expanded_standings_overall'})

            df = pd.read_html(StringIO(str(table)))[0]
            df = df[df['Tm']!= 'Average']
            break
    return df
# https://sdwww.baseball-reference.com/leagues/daily.fcgi?request=1&type=b&dates=yesterday&lastndays=3&since=2024-06-01&fromandto=2024-06-01.2024-06-30&level=mlb&franch=BOS#daily::18
def get_all_player_batting():
    url = 'https://www.baseball-reference.com/leagues/majors/2024-standard-batting.shtml#players_standard_batting'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    tables = soup.find('table', {'id': 'players_standard_batting'})
    table = StringIO(str(tables))
    df = pd.read_html(table)[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    
    # Drop rows with NaN in any column
    df = df.dropna()


    numeric_columns = ['Age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'BA', 'OBP', 'SLG', 'OPS', 'OPS+', 'TB', 'GIDP', 'HBP', 'SH', 'SF', 'IBB']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    return df

def get_last_game(team:str):
    """Returns data for each player of their previous game"""
    team = team.upper()
    
    
    today_date = datetime.now().date()
    month = datetime.now().month
    year = str(datetime.now().year)
    if month < 10: 
        month = str(month)
        month = '0' + month
    else: month = str(month)

    url = f'https://www.baseball-reference.com/leagues/daily.fcgi?request=1&type=b&dates=yesterday&lastndays=3&since={year}-{month}-01&fromandto={year}-{month}-01.{year}-{month}-30&level=mlb&franch={team}#daily::18'
    try:
        r = requests.get(url)
        r.raise_for_status()
    except Exception as e:  # Check if the request was successful
        print("could not fetch yesterdays games", e)
        r.raise_for_status()
    
    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.find('table', {'id': 'daily'})
    # Find the relevant table by its class or id
    table = StringIO(str(table))
    # Read the table into a pandas DataFrames
    stats = pd.read_html(table)[0]
    
    return stats

    
def get_team_streaks():
    """Returns each teams W/L streak"""
    standings = get_standings()
    return standings[['Tm', 'Strk']]
    

def get_season_stats(team):
    """TODO: Implement"""
    pass


if __name__ == '__main__':
    #get_team_last_3_games("BOS")
    pass
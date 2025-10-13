import pandas as pd
import requests
from io import StringIO
import time
from datetime import datetime
from bs4 import BeautifulSoup
from leagues.NBA.utils.abbr_map import TEAM_MAP

def get_train_data() -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    abreviations = list(TEAM_MAP.values())  # list of all abreviations
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


def get_today_games() -> pd.DataFrame:

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
    # today = "Sun, Apr 13, 2025"  # Manual date format
    # tmrw = "Wed, Apr 23, 2025"  # Manual date format
    # Initialize a list to hold today's games
    games_today = []

    for row in rows:
        # Extract the date cell
        date_cell = row.find('th', {'data-stat': 'date_game'})
        if date_cell and date_cell.text.strip() == today:
            # Extract team names
            away_team = row.find(
                'td', {'data-stat': 'visitor_team_name'}).text.strip()
            home_team = row.find(
                'td', {'data-stat': 'home_team_name'}).text.strip()
            games_today.append(
                {'home_team': home_team, 'away_team': away_team})

        # if date_cell and date_cell.text.strip() == tmrw:
            Extract team names
            # away_team = row.find(
                # 'td', {'data-stat': 'visitor_team_name'}).text.strip()
            # home_team = row.find(
                # 'td', {'data-stat': 'home_team_name'}).text.strip()
            # games_today.append(
                # {'home_team': home_team, 'away_team': away_team})

    # Convert the list to a DataFrame
    return pd.DataFrame(games_today)

def get_team_per_game_stats(team_abbr, args='adv'):
    """
    Scrapes the 'per_game' table for a given team using its abbreviation.
    """
    if args == 'adv':

        try:
            data = pd.read_excel('leagues/NBA/data/tpgApr.xlsx',
                                 sheet_name='Worksheet', header=0)
            data = data[data['Team'] == team_abbr]

            # Extract only the required columns
            feature_columns = ['PTS', 'FG%', 'FGA', '3P%', '3PA', 'ORB', 'TRB',
                               'AST', 'TOV', 'STL', 'PF', 'ORtg', 'DRtg', 'FTA', 'FT%']

            # If there is no matching data, return zeros
            if data.empty:
                return {col: 0 for col in feature_columns}

            # Extract scalar values
            # Assume the first matching row is correct
            team_stats = data.iloc[0]
            return {col: team_stats.get(col, 0) for col in feature_columns}
        except Exception as e:
            print('Error in get_team_per_game_stats(): Adv model:', e)
            pass

    elif args == '30dw':
        data = pd.read_csv('leagues/NBA/data/window/30dwFeb.csv', header=0)
        data = data[data['Team'] == team_abbr]

        # Ensure 'PTS' is numeric
        # data['PTS'] = pd.to_numeric(data['PTS'], errors='coerce')  # Convert non-numeric values to NaN if any

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
            relevant_stats = {col: team_stats.get(
                col, 0) for col in feature_columns}

            return relevant_stats
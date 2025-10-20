import pandas as pd
import requests
from io import StringIO
import time
from datetime import datetime
from bs4 import BeautifulSoup
from utils.abbr_map import TEAM_MAP

def get_train_data() -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.basketball-reference.com/",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
    }
    from datetime import datetime as _dt
    _cur_year = _dt.now().year
    years = range(_cur_year - 2, _cur_year + 1)  # last 2 seasons up to current year page
    abreviations = list(TEAM_MAP.values())  # list of all abreviations
    teams_data = pd.DataFrame()
    for year in years:
        for abrv in abreviations:
            url = f'https://www.basketball-reference.com/teams/{abrv}/{year}/gamelog/'

            try:
                r = requests.get(url, headers=headers)
                r.raise_for_status()
            except Exception as e:
                print(f'[warn] {abrv}/{year} fetch failed: {e}')
                time.sleep(2)
                continue

            soup = BeautifulSoup(r.text, 'html.parser')
            table = soup.find('table', {'id': 'tgl_basic'})

            if table:
                table = StringIO(str(table))
                team = pd.read_html(table)[0]

                teams_data = pd.concat([teams_data, team], ignore_index=True)

        time.sleep(180)

    return teams_data


def get_today_games() -> pd.DataFrame:
    """
    Scrape today's schedule from Basketball-Reference with polite headers and robust date handling.
    Returns an empty DataFrame if the schedule page isn't available yet.
    """
    import os
    import random
    import requests
    from datetime import datetime as _dt
    from bs4 import BeautifulSoup

    # Build polite headers to reduce 403/anti-bot responses
    ua_pool = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    ]
    headers = {
        "User-Agent": random.choice(ua_pool),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.basketball-reference.com/",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
    }

    # Respect current calendar date
    now = _dt.now()
    month_name = now.strftime('%B').lower()  # e.g. 'october'
    year = now.year

    # BBR uses league year pages like NBA_2025_games-october.html (calendar year)
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month_name}.html"

    # Build the "today" string as it appears in the schedule table, e.g. "Mon, Oct 20, 2025"
    today_label = now.strftime(f'%a, %b {now.day}, %Y')

    sess = requests.Session()
    sess.headers.update(headers)

    try:
        resp = sess.get(url, timeout=20)
        # If we got a 403, retry once with a different UA and short backoff
        if resp.status_code == 403:
            time.sleep(1.5)
            sess.headers["User-Agent"] = random.choice(ua_pool)
            resp = sess.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"[warn] schedule fetch failed for {url}: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, 'html.parser')
    schedule_table = soup.find('table', {'id': 'schedule'})
    if not schedule_table:
        print("[info] Schedule table not available yet.")
        return pd.DataFrame()

    rows = schedule_table.find_all('tr')
    games_today = []
    for row in rows:
        date_cell = row.find('th', {'data-stat': 'date_game'})
        if not date_cell:
            continue
        if date_cell.text.strip() == today_label:
            away = row.find('td', {'data-stat': 'visitor_team_name'})
            home = row.find('td', {'data-stat': 'home_team_name'})
            if away and home:
                games_today.append({'home_team': home.text.strip(), 'away_team': away.text.strip()})

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

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.basketball-reference.com/",
        }
        response = requests.get(url, headers=headers, timeout=20)
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
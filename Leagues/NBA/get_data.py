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
    Load today's NBA games from a local Excel schedule instead of scraping.
    File: leagues/NBA/Data/schedule/october_schedule.xlsx
    Sheet: "October Schedule"
    Returns DataFrame with columns: home_team, away_team
    """
    import os
    from datetime import datetime as _dt

    # Try both lowercase and capitalized 'Data' folders
    sheet_name = "October Schedule"
    # Try both lowercase and capitalized 'Data' folders
    path_candidates = [
        "leagues/NBA/data/schedule/october_schedule.xlsx",
        "leagues/NBA/Data/schedule/october_schedule.xlsx",
    ]
    xlsx_path = next((p for p in path_candidates if os.path.exists(p)), path_candidates[0])

    # Determine "today" (local) in Excel string format, e.g. "Tue Oct 21 2025"
    today_str = _dt.now().strftime("%a %b %d %Y")
    # For quick testing, include a small whitelist of known dates
    date_whitelist = {
        # today_str,
        "Tue Oct 28 2025",
        "Wed Oct 29 2025",
        "Thu Oct 30 2025",
    }
    # Read the Excel sheet
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"[error] Could not read schedule: {xlsx_path} sheet '{sheet_name}': {e}")
        return pd.DataFrame(columns=["home_team", "away_team"])

    # Normalize / detect key columns
    norm = {str(c).strip().lower(): c for c in df.columns}
    # Common header variants
    date_col = next((norm[k] for k in ("date", "game date", "day")), None)
    away_col = next((norm[k] for k in ("visitor/neutral", "away", "visitor team", "visitor")), None)
    home_col = next((norm[k] for k in ("home/neutral", "home", "home team")), None)

    if not date_col or not away_col or not home_col:
        print(f"[error] Missing expected columns. Found: {list(df.columns)}")
        return pd.DataFrame(columns=["home_team", "away_team"])

    # Parse dates and filter to allowed dates, matching Excel's string format
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%a %b %d %Y")
    todays = df[df["_date"].isin(date_whitelist)]
    

    if todays.empty:
        # Nothing for these dates (or sheet doesn't include them yet)
        print(f"[info] No schedule rows for dates: {sorted(list(date_whitelist))}")
        return pd.DataFrame(columns=["home_team", "away_team"])

    # Build output
    out = todays.rename(columns={home_col: "home_team", away_col: "away_team"})[["home_team", "away_team"]].copy()
    for c in ("home_team", "away_team"):
        out[c] = out[c].astype(str).str.strip()

    return out.reset_index(drop=True)

def get_team_per_game_stats(team_abbr, args='adv'):
    """
    Scrapes the 'per_game' table for a given team using its abbreviation.
    """
    if args == 'adv':

        try:
            # data = pd.read_excel('leagues/NBA/data/tpgApr.xlsx',
            #                      sheet_name='Worksheet', header=0)
            data = pd.read_excel('leagues/NBA/data/tpgOct26.xlsx',
                     sheet_name='TPG', header=0)
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
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
    Return all (home_team, away_team) matchups for a chosen slate of dates.

    Important:
    - This is NOT strictly "today". We intentionally allow multiple dates so
      we can batch more games into the model at once (bigger inference matrix).
    - You control the slate by editing DATE_WHITELIST below.

    Expected output columns:
      home_team, away_team

    How it works:
    1. Load the local schedule workbook (no scraping).
    2. Normalize the date column in the sheet to strings like "Fri Oct 31 2025".
       NOTE: that's strftime("%a %b %d %Y"), which zero-pads the day.
    3. Filter rows whose normalized date is in DATE_WHITELIST.
    4. Return just home/away team names.

    If the sheet can't be read, or the columns don't match, or no rows match
    the whitelist, this returns an empty DataFrame with the correct columns.
    """
    import pandas as pd
    from datetime import datetime as _dt

    # ----------------------------
    # 1. Control which dates we pull
    # ----------------------------
    # Use only today's date by default so we always pull the current slate.
    today_str = _dt.now().strftime("%a %b %d %Y")
    DATE_WHITELIST = {today_str}

    # ----------------------------
    # 2. Load the schedule Excel
    # ----------------------------
    # We'll try a primary path, and if that fails we bail gracefully
    SCHEDULE_PATHS = [
        "leagues/NBA/data/tpgOct26.xlsx",
        # "leagues/NBA/data/schedule.xlsx",  # future fallback if you make a generic file
    ]
    SHEET_NAME = "October Schedule"

    df = None
    last_err = None
    for path in SCHEDULE_PATHS:
        try:
            df = pd.read_excel(path, sheet_name=SHEET_NAME, header=0)
            break
        except Exception as e:
            last_err = e
            continue

    if df is None:
        print(f"[error] Could not read any schedule file from {SCHEDULE_PATHS} (last err: {last_err})")
        return pd.DataFrame(columns=["home_team", "away_team"])

    # ----------------------------
    # 3. Identify the important columns in whatever header naming the sheet uses
    # ----------------------------
    norm = {str(c).strip().lower(): c for c in df.columns}

    date_col = next((norm[k] for k in ("date", "game date", "day")), None)
    away_col = next((norm[k] for k in ("visitor/neutral", "away", "visitor team", "visitor")), None)
    home_col = next((norm[k] for k in ("home/neutral", "home", "home team")), None)

    if not date_col or not away_col or not home_col:
        print(f"[error] Missing expected columns for date/home/away. Found: {list(df.columns)}")
        return pd.DataFrame(columns=["home_team", "away_team"])

    # ----------------------------
    # 4. Normalize dates in the sheet to "%a %b %d %Y" strings
    # ----------------------------
    df["_date_norm"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%a %b %d %Y")

    # Some sheets might not zero-pad day numbers originally (e.g. 'Sat Nov 1 2025').
    # We'll also build a non-zero-padded version to be permissive:
    df["_date_alt"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%a %b %-d %Y" if hasattr(pd.Timestamp.now(), "day") else "%a %b %d %Y")
    # NOTE: the %-d trick is POSIX-y; on Windows it may not apply. Fallback above keeps it safe.

    # ----------------------------
    # 5. Filter rows that match ANY of the whitelisted dates
    # ----------------------------
    slate = df[(df["_date_norm"].isin(DATE_WHITELIST)) | (df["_date_alt"].isin(DATE_WHITELIST))]

    if slate.empty:
        print(f"[info] No schedule rows for dates: {sorted(list(DATE_WHITELIST))}")
        print("[info] This usually means today's games are not in the schedule workbook yet.")
        return pd.DataFrame(columns=["home_team", "away_team"])

    # ----------------------------
    # 6. Build clean output with abbreviations
    # ----------------------------
    out = slate.rename(
        columns={
            home_col: "home_team",
            away_col: "away_team",
        }
    ).copy()

    # Normalize spacing and map to abbreviations via TEAM_MAP so that
    # schedule uses the same representation (e.g., "CLE") as the Odds API.
    def _to_abbrev(name: str) -> str:
        name = str(name).strip()
        return TEAM_MAP.get(name, name)

    for c in ("home_team", "away_team"):
        out[c] = out[c].apply(_to_abbrev)

    return out[["home_team", "away_team"]].reset_index(drop=True)

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
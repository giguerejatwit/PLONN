'''
Odds API

Pulling pitcher K O/U
Pulling DK lines for MLB totals
'''
import time
from datetime import date
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import json
import requests
import datetime
import pandas as pd
import os
from bs4 import BeautifulSoup
from leagues.MLB.utils.abbr_map import get_team_name_or_abbr as mlb_name_or_abbr
from leagues.NBA.utils.abbr_map import get_team_name_or_abbr as nba_name_or_abbr

API_KEY = os.getenv('ODDS_API_KEY')
print(f'API_KEY:', API_KEY)
SPORT = 'baseball_mlb'
REGIONS = 'us'
ODDS_FORMAT = 'decimal'
DATE_FORMAT = 'iso'
MARKETS = 'totals'  # 'pitcher_strikeouts'


def get_sports():
    sports_response = requests.get(
        f'https://api.the-odds-api.com/v4/sports',
        params={
            'api_key': API_KEY,
            'regions': REGIONS,
            'markets': MARKETS,
            'oddsFormat': ODDS_FORMAT,
            'dateFormat': DATE_FORMAT,
        })
    
    if sports_response.status_code != 200:
        print(f'Failed to get odds: status_code {sports_response.status_code}')

    else:
        sports_json = sports_response.json()

        print(sports_json)


def get_event_ids(sport=SPORT) -> list:
    odds_response = requests.get(
        f'https://api.the-odds-api.com/v4/sports/{sport}/events',
        params={
            'api_key': API_KEY,
            'regions': REGIONS,
            'markets': MARKETS,
            'oddsFormat': ODDS_FORMAT,
            'dateFormat': DATE_FORMAT,
        })

    if odds_response.status_code != 200:
        print(
            f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')
        return [], odds_response
    else:
        odds_json = odds_response.json()
        eventIDs = [event['id'] for event in odds_json]
        print(f'Number of {sport} games today:', len(odds_json))
        return eventIDs, odds_response




def get_dk_lines(sportbook: str = 'draftkings', sport: str = SPORT) -> pd.DataFrame:
    """Fetch DraftKings totals lines for the given sport.
    Use MLB or NBA team name↔abbr mapping based on the sport key.
    - sport examples: 'baseball_mlb', 'basketball_nba'
    """
    # Choose the proper team-mapping function
    if sport.startswith('baseball'):
        name_or_abbr = mlb_name_or_abbr
    elif sport.startswith('basketball'):
        name_or_abbr = nba_name_or_abbr
    else:
        # Default to returning the original value unchanged
        name_or_abbr = lambda x: x

    lines = []
    eventIDs, odds_response = get_event_ids(sport=sport)
    for id in eventIDs:
        totals_response = requests.get(
            f'https://api.the-odds-api.com/v4/sports/{sport}/events/{id}/odds',
            params={
                'api_key': API_KEY,
                'regions': REGIONS,
                'markets': MARKETS,
                'oddsFormat': ODDS_FORMAT,
                'dateFormat': DATE_FORMAT,
            })
        if totals_response.status_code != 200:
            print(
                f'Failed to get odds: status_code {totals_response.status_code}, response body {odds_response.text}')
            return

        game_info = totals_response.json()
        if not game_info:
            print(f'No game info found for game ID {type(id)}: {id}')
            return

        dk_market = None
        for bookmaker in game_info.get('bookmakers', []):
            if bookmaker.get('key') == sportbook:
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'totals':
                        dk_market = market
                        break
        if dk_market is None:
            print(f'No {sportbook} market found for game ID {id}')
            continue

        try:
            point_val = dk_market['outcomes'][0]['point']
        except (KeyError, IndexError):
            point_val = None

        game_totals = {
            'home_team': name_or_abbr(game_info.get('home_team', '')),
            'away_team': name_or_abbr(game_info.get('away_team', '')),
            'DK_Line': point_val,
            'Last_Updated': dk_market.get('last_update'),
            'sport': sport,
        }

        lines.append(game_totals)

    return pd.DataFrame(lines)


def get_nba_dk_lines(sportbook: str = 'draftkings') -> pd.DataFrame:
    """Convenience wrapper for NBA totals lines."""
    return get_dk_lines(sportbook=sportbook, sport='basketball_nba')

    
def get_historic_odds(api_key,
                  sport='baseball_mlb',
                  markets='h2h',
                  odds_format='decimal',
                  start_date=datetime.date(2025, 4, 1)):
    """
    Fetch historical odds from Opening Day 2025 through yesterday,
    saving each day's JSON to 'historical_odds_2025/YYYY-MM-DD.json'.
    """
    # Determine end date (yesterday)
    end_date = datetime.date.today() - datetime.timedelta(days=1)
    
    # Prepare output directory
    output_dir = 'historical_odds_2025'
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop over each date
    total_days = (end_date - start_date).days + 1
    for offset in range(total_days):
        single_date = start_date + datetime.timedelta(days=offset)
        date_str = single_date.strftime('%Y-%m-%d')
        
        url = f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds/?apiKey={API_KEY}&regions=us&markets={MARKETS}&oddsFormat={ODDS_FORMAT}&date={date_str}T12:00:00Z'
        
        response = requests.get(url,
            params={
                'api_key': API_KEY,
                'regions': REGIONS,
                'markets': MARKETS,
                'oddsFormat': ODDS_FORMAT,
                'dateFormat': DATE_FORMAT,
            })
        if response.ok:
            data = response.json()
            out_path = os.path.join(output_dir, f'{date_str}.json')
            with open(out_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved odds for {date_str}")
        else:
            print(f"Error {response.status_code} fetching {date_str}")
        
        time.sleep(2)




def scrape_totals_sbr_selenium(single_date):
    date_str = single_date.strftime('%Y-%m-%d')
    url = f"https://www.sportsbookreview.com/betting-odds/mlb-baseball/totals/full-game/?date={date_str}"

    opts = Options()
    opts.add_argument("--headless")
    driver = webdriver.Chrome(options=opts)
    driver.get(url)

    # wait a couple seconds for JS to render
    driver.implicitly_wait(2)

    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, "html.parser")
    records = []
    for game_div in soup.find_all("div", id=lambda v: v and v.startswith("game-")):
        # description block
        desc = game_div.select_one("div[class*='OddsTableMobile_gameDescription']")
        teams_div = desc.select_one("div.h5")
        away, home = teams_div.get_text(strip=True).split(" vs ")
        
        # find opener slide
        odds_container = game_div.select_one("div[class*='OddsTableMobile_oddsNumberContainer']")
        opener = odds_container.select_one("div[data-index='1']")
        spans = opener.select("div[class*='OddsTableMobile_opener__'] span")
        total = spans[0].get_text(strip=True)
        over_odds  = spans[1].get_text(strip=True)
        under_odds = spans[3].get_text(strip=True)  # 0:line,1:over,2:line,3:under

        records.append({
            "date":      date_str,
            "away":      away,
            "home":      home,
            "total":     total,
            "over_odds": over_odds,
            "under_odds":under_odds
        })

    return pd.DataFrame(records)

# Test it locally:
if __name__ == "__main__":
    from datetime import date
    df = scrape_totals_sbr_selenium(date(2025,4,1))
    print(df)
    df.to_csv("mlb_totals_2025-04-01.csv", index=False)

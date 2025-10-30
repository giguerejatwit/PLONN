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
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

API_KEY = os.getenv('ODDS_API_KEY')
if API_KEY:
    print(f'API_KEY: {API_KEY[:4]}***{API_KEY[-4:]}')
else:
    print('API_KEY: None')
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
                f'Failed to get odds: status_code {totals_response.status_code}, response body {totals_response.text}')
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
            'dk lines': point_val,
            'Last_Updated': dk_market.get('last_update'),
            'sport': sport,
        }

        lines.append(game_totals)

    return pd.DataFrame(lines)


def get_nba_dk_lines(sportbook: str = 'draftkings') -> pd.DataFrame:
    """Convenience wrapper for NBA totals lines."""
    return get_dk_lines(sportbook=sportbook, sport='basketball_nba')


# --- NEW: Fetch NBA totals today to CSV via The Odds API ---
def fetch_nba_totals_today_csv(api_key: str = API_KEY,
                               regions: str = 'us',
                               odds_format: str = 'american',
                               date_format: str = 'iso',
                               bookmaker: str = 'draftkings',
                               out_dir: str = 'leagues/NBA/data') -> str:
    """Fetch today's NBA totals odds via The Odds API and write a CSV.

    Columns: date (YYYY-MM-DD), commence_time (ISO), event_id, home_team, away_team,
             bookmaker_key, bookmaker_title, point, over_price, under_price, market_last_update
    """
    if not api_key:
        raise ValueError('fetch_nba_totals_today_csv: missing api_key (env ODDS_API_KEY).')

    url = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
    params = {
        'apiKey': api_key,
        'regions': regions,
        'markets': 'totals',
        'oddsFormat': odds_format,
        'dateFormat': date_format,
    }

    resp = requests.get(url, params=params, timeout=45)
    resp.raise_for_status()
    events = resp.json() or []

    rows = []
    for ev in events:
        event_id   = ev.get('id')
        commence   = ev.get('commence_time')  # ISO
        home_team  = ev.get('home_team')
        away_team  = ev.get('away_team')
        # Derive date (UTC) from commence_time
        try:
            date_str = pd.to_datetime(commence, utc=True).strftime('%Y-%m-%d') if commence else None
        except Exception:
            date_str = None

        for bk in ev.get('bookmakers', []):
            bk_key   = bk.get('key')
            bk_title = bk.get('title')
            if bookmaker and bk_key != bookmaker:
                continue

            for m in bk.get('markets', []):
                if m.get('key') != 'totals':
                    continue
                m_last = m.get('last_update')
                over_price = None
                under_price = None
                point = None
                for out in m.get('outcomes', []):
                    name = (out.get('name') or '').lower()
                    if point is None:
                        point = out.get('point')
                    if name == 'over':
                        over_price = out.get('price')
                    elif name == 'under':
                        under_price = out.get('price')
                rows.append({
                    'date': date_str,
                    'commence_time': commence,
                    'event_id': event_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'bookmaker_key': bk_key,
                    'bookmaker_title': bk_title,
                    'point': point,
                    'over_price': over_price,
                    'under_price': under_price,
                    'market_last_update': m_last,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print('[INFO] No totals odds returned for NBA today.')
        return ''

    df.sort_values(['date','commence_time','bookmaker_key'], inplace=True)
    # Use first (min) date present for filename, fallback to today
    file_date = df['date'].dropna().min() or datetime.date.today().strftime('%Y-%m-%d')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'nba_totals_{file_date}.csv')
    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote NBA totals → {out_path} (rows={len(df)})")
    return out_path

def _iso_end_of_day(d: pd.Timestamp) -> str:
    return pd.Timestamp(d.date()) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

def _to_iso(ts) -> str:
    # Robustly convert any timestamp-like value to ISO UTC (Z) without tz errors
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None or ts.tz is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    return ts.strftime('%Y-%m-%dT%H:%M:%SZ')
     




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


def fetch_historical_totals_csv(api_key: str,
                                start_date: str,
                                end_date: str,
                                regions: str = 'us',
                                odds_format: str = 'american',
                                date_format: str = 'iso',
                                out_path: str = 'leagues/NBA/data/historical_totals.csv',
                                sleep_secs: float = 0.15,
                                show_progress: bool = True,
                                bookmaker_only: str | None = 'draftkings',
                                resume: bool = True,
                                max_workers: int = 8) -> str:
    """
    Pull historical NBA totals odds snapshots from The Odds API and save to CSV.
    - Iterates calendar days in [start_date, end_date]
    - For each day, fetch historical events at ~end-of-day
    - For each event, fetch totals odds near tip (commence-5m, fallback to commence)
    - Writes rows PER-DAY to CSV (append-mode) so you can resume with --resume
    """
    if not api_key:
        raise ValueError('fetch_historical_totals_csv: missing api_key. Set ODDS_API_KEY or pass api_key.')

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    total_days = (end - start).days + 1
    session = requests.Session()
    base = 'https://api.the-odds-api.com/v4/historical/sports/basketball_nba'

    # Resume support: dates already present in out_path
    written_dates = set()
    header_exists = os.path.exists(out_path)
    if resume and header_exists:
        try:
            prev = pd.read_csv(out_path, usecols=['date'], dtype=str)
            written_dates = set(prev['date'].dropna().unique().tolist())
            print(f"[INFO] Resuming; {len(written_dates)} dates already in {out_path}")
        except Exception as ex:
            print(f"[WARN] Could not read {out_path} for resume ({ex}); continuing without resume index")

    rng = range(total_days)
    if show_progress and tqdm is not None:
        rng = tqdm(rng, desc='Fetching historical totals', unit='day')

    def fetch_event_rows(ev, eod_iso):
        rows = []
        try:
            event_id = ev['id'] if isinstance(ev, dict) and 'id' in ev else (ev if isinstance(ev, str) else None)
            if not event_id:
                return rows
            commence = ev.get('commence_time') if isinstance(ev, dict) else None
            home_team = ev.get('home_team') if isinstance(ev, dict) else None
            away_team = ev.get('away_team') if isinstance(ev, dict) else None
            sport_key = (ev.get('sport_key') if isinstance(ev, dict) else None) or 'basketball_nba'

            # Preferred snapshot: 5 minutes before tip; fallback to EOD
            if commence:
                ct = pd.to_datetime(commence)
                if ct.tzinfo is None or ct.tz is None:
                    ct = ct.tz_localize('UTC')
                else:
                    ct = ct.tz_convert('UTC')
                pre_ts = ct - pd.Timedelta(minutes=5)
                snap_iso = _to_iso(pre_ts)
            else:
                snap_iso = eod_iso

            odds_url = f"{base}/events/{event_id}/odds"
            odds_params = {
                'apiKey': api_key,
                'date': snap_iso,
                'regions': regions,
                'markets': 'totals',
                'oddsFormat': odds_format,
                'dateFormat': date_format,
            }
            if bookmaker_only:
                odds_params['bookmakers'] = bookmaker_only

            # Retry/backoff
            for attempt in range(3):
                od_resp = session.get(odds_url, params=odds_params, timeout=30)
                if od_resp.status_code not in (429, 500, 502, 503, 504):
                    break
                time.sleep(2 ** attempt)
            if od_resp.status_code == 404 and commence:
                # Fallback to commence timestamp
                od_params2 = dict(odds_params)
                od_params2['date'] = _to_iso(ct)
                od_resp = session.get(odds_url, params=od_params2, timeout=30)
            od_resp.raise_for_status()
            odds_json = od_resp.json() or {}

            for bk in odds_json.get('bookmakers', []):
                bk_key = bk.get('key')
                if bookmaker_only and bk_key != bookmaker_only:
                    continue
                bk_title = bk.get('title')
                bk_last = bk.get('last_update')
                for m in bk.get('markets', []):
                    if m.get('key') != 'totals':
                        continue
                    m_last = m.get('last_update')
                    for out in m.get('outcomes', []):
                        name = out.get('name')
                        price = out.get('price')
                        point = out.get('point')
                        rows.append({
                            'date': pd.to_datetime(ev.get('commence_time')).strftime('%Y-%m-%d') if isinstance(ev, dict) and ev.get('commence_time') else None,
                            'commence_time': ev.get('commence_time') if isinstance(ev, dict) else None,
                            'event_id': event_id,
                            'home_team': home_team,
                            'away_team': away_team,
                            'sport_key': sport_key,
                            'bookmaker_key': bk_key,
                            'bookmaker_title': bk_title,
                            'bookmaker_last_update': bk_last,
                            'market_key': 'totals',
                            'market_last_update': m_last,
                            'outcome_name': name,
                            'price': price,
                            'point': point,
                            'date_snapshot': odds_params['date'],
                        })
        except Exception as ex:
            ev_id_dbg = ev['id'] if isinstance(ev, dict) and 'id' in ev else (ev if isinstance(ev, str) else 'UNKNOWN')
            print(f"[WARN] odds fetch failed for event {ev_id_dbg} → {ex}")
        return rows

    for i in rng:
        day = start + pd.Timedelta(days=i)
        day_str = day.strftime('%Y-%m-%d')
        if resume and day_str in written_dates:
            # Skip dates already present
            if show_progress and tqdm is None:
                print(f"[SKIP] {day_str} already in CSV")
            continue

        eod_iso = _to_iso(_iso_end_of_day(day))
        try:
            ev_url = f"{base}/events"
            ev_params = {'apiKey': api_key, 'date': eod_iso}
            for attempt in range(3):
                ev_resp = session.get(ev_url, params=ev_params, timeout=30)
                if ev_resp.status_code not in (429, 500, 502, 503, 504):
                    break
                time.sleep(2 ** attempt)
            ev_resp.raise_for_status()

            # Parse shape: dict with metadata + 'data' list OR raw list
            ev_json = ev_resp.json() or {}
            if isinstance(ev_json, dict):
                raw_events = ev_json.get('data') or ev_json.get('events') or []
            else:
                raw_events = ev_json

            # Normalize to list of dicts with at least 'id'
            events = []
            for ev in raw_events:
                if isinstance(ev, dict) and 'id' in ev:
                    events.append(ev)
                elif isinstance(ev, str):
                    events.append({'id': ev})

        except Exception as ex:
            print(f"[WARN] events fetch failed for {day_str} → {ex}")
            events = []

        rows_today = []
        if events:
            # Fetch per-event concurrently
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(fetch_event_rows, ev, eod_iso) for ev in events]
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        rows_today.extend(res)

        # Flush to disk per day (append)
        if rows_today:
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            day_df = pd.DataFrame(rows_today)
            # Optional server-side filter already applied, but keep client-side too
            if bookmaker_only:
                day_df = day_df[day_df['bookmaker_key'] == bookmaker_only]
            day_df.sort_values(['commence_time','bookmaker_key','outcome_name'], inplace=True)
            write_header = not header_exists and not os.path.exists(out_path)
            day_df.to_csv(out_path, mode='a', index=False, header=write_header)
            header_exists = True
            written_dates.add(day_str)
            if tqdm is None:
                print(f"[INFO] {day_str}: wrote {len(day_df)} rows → {out_path}")

        if show_progress and tqdm is not None:
            try:
                rng.set_postfix({'events': len(events), 'rows': len(rows_today)})
            except Exception:
                pass

        time.sleep(sleep_secs)

    print(f"[INFO] Historical totals fetch complete → {out_path}")
    return out_path

# Historical pull convenience entrypoint
if __name__ == "__main__":
    import argparse
    start_default = '2022-10-22'
    end_default = datetime.date.today().strftime('%Y-%m-%d')
    out_default = f"leagues/NBA/data/historical_totals_{start_default}_to_{end_default}.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default=start_default)
    parser.add_argument('--end-date', type=str, default=end_default)
    parser.add_argument('--out-path', type=str, default=out_default)
    parser.add_argument('--regions', type=str, default='us')
    parser.add_argument('--odds-format', type=str, default='american')
    parser.add_argument('--date-format', type=str, default='iso')
    parser.add_argument('--bookmaker', type=str, default='draftkings', help='Only keep rows for this bookmaker key; blank = keep all')
    parser.add_argument('--resume', action='store_true', help='Resume from existing CSV and append new days')
    parser.add_argument('--max-workers', type=int, default=8, help='Concurrency for per-event odds requests')
    args = parser.parse_args()

    fetch_historical_totals_csv(
        api_key=API_KEY,
        start_date=args.start_date,
        end_date=args.end_date,
        regions=args.regions,
        odds_format=args.odds_format,
        date_format=args.date_format,
        out_path=args.out_path,
        show_progress=True,
        bookmaker_only=(args.bookmaker or None),
        resume=args.resume,
        max_workers=args.max_workers,
    )

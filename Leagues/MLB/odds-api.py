'''
Odds API

Pulling pitcher K O/U
'''

import requests
API_KEY='e4d198375b199b3ee5215e87dcefb69c'
SPORT='baseball_mlb'
REGIONS='us'
ODDS_FORMAT='decimal'
DATE_FORMAT='iso'
MARKETS='pitcher_strikeouts'

def get_sports():
    sports_response = requests.get(
        f'https://api.the-odds-api.com/v4/sports',
        params={
            'api_key': API_KEY,
            'regions': REGIONS,
            'markets': MARKETS,
            'oddsFormat': ODDS_FORMAT,
            'dateFormat': DATE_FORMAT,
        }
    )
    if sports_response.status_code != 200:
        print(f'Failed to get odds: status_code {sports_response.status_code}, response body {odds_response.text}')

    else:
        sports_json = sports_response.json()
        print('Number of events:', len(sports_json))
        print(sports_json)

odds_response = requests.get(
    f'https://api.the-odds-api.com/v4/sports/{SPORT}/events',
    params={
        'api_key': API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': DATE_FORMAT,
    }
)

if odds_response.status_code != 200:
    print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

else:
    odds_json = odds_response.json()
    eventIDs = [event['id'] for event in odds_json]
    print('Number of events:', len(odds_json))
    print(odds_json)
    print(eventIDs)

for id in eventIDs:
    pitcher_response = requests.get(
    f'https://api.the-odds-api.com/v4/sports/{SPORT}/events/{id}/odds',
    params={
        'api_key': API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': DATE_FORMAT,
    }
)
    if pitcher_response.status_code != 200:
        print(f'Failed to get odds: status_code {pitcher_response.status_code}, response body {odds_response.text}')

    else:
        pitcher_json = pitcher_response.json()
        print(pitcher_json)
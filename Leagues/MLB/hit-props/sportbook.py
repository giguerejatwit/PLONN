import requests
import os
from bs4 import BeautifulSoup, Comment

API_KEY = os.getenv("ODDS_API_KEY")

url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={API_KEY}&regions=us&markets=hits"

try:
    r = requests.get(url)
    print(r.json())
    r.raise_for_status()
except Exception as e:
    r.raise_for_status()



        




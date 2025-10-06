import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from Leagues.MLB.utils.logger import setup_logger





def scrape_today_pitchers() -> pd.DataFrame:
    url = "https://www.baseball-reference.com/previews/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    games = []

    # Each preview table lists both teams and their pitchers
    for table in soup.select("div#content table"):
        try:
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue

            team1_td = rows[0].find_all("td")[1]
            team2_td = rows[1].find_all("td")[1]

            team1_pitcher = team1_td.find("a").text.strip() if team1_td.find("a") else "TBD"
            team2_pitcher = team2_td.find("a").text.strip() if team2_td.find("a") else "TBD"

            team1_abbr = rows[0].find("strong").text.strip() if rows[0].find("strong") else rows[0].text.strip().split()[0]
            team2_abbr = rows[1].find("strong").text.strip() if rows[1].find("strong") else rows[1].text.strip().split()[0]

            # Only save if pitcher looks like a real name
            if " " in team1_pitcher and team1_pitcher.lower() not in [team1_abbr.lower(), team2_abbr.lower()]:
                games.append({
                    "Date": datetime.now().date(),
                    "Team": team1_abbr,
                    "Opponent": team2_abbr,
                    "Player": team1_pitcher
                })

            if " " in team2_pitcher and team2_pitcher.lower() not in [team1_abbr.lower(), team2_abbr.lower()]:
                games.append({
                    "Date": datetime.now().date(),
                    "Team": team2_abbr,
                    "Opponent": team1_abbr,
                    "Player": team2_pitcher
                })

        except Exception as e:
            print(f"Error parsing table: {e}")
            print("Row 1:", rows[0].text.strip())
            print("Row 2:", rows[1].text.strip())
            print("-" * 50)
            continue


    games = pd.DataFrame(games)
    # games.to_csv("Leagues/MLB/data/today_pitchers.csv", index=False)
    return games

if __name__ == "__main__":
    df_today_pitchers = scrape_today_pitchers()
    
    # Logging
    logger = setup_logger(name="mlb_today_pitchers", log_dir="Leagues/MLB/logs/StartingPitchers")
    logger.info(df_today_pitchers)
    df_today_pitchers.to_csv("Leagues/MLB/data/today_pitchers.csv", index=False)
    print(df_today_pitchers)
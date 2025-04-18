import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime


def scrape_today_pitchers():
    url = "https://www.baseball-reference.com/previews/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    games = []

    # Each preview table lists both teams and their pitchers
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) != 2:
            continue

        try:
            team1_abbr = rows[0].find("strong").text.strip()
            team1_pitcher = rows[0].find("a").text.strip()

            team2_abbr = rows[1].find("strong").text.strip()
            team2_pitcher = rows[1].find("a").text.strip()

            games.append({
                "Date": datetime.now().date(),
                "Team": team1_abbr,
                "Opponent": team2_abbr,
                "Player": team1_pitcher
            })

            games.append({
                "Date": datetime.now().date(),
                "Team": team2_abbr,
                "Opponent": team1_abbr,
                "Player": team2_pitcher
            })
        except Exception as e:
            print("Error parsing table:", e)
            continue

    return pd.DataFrame(games)


if __name__ == "__main__":
    df_today_pitchers = scrape_today_pitchers()
    df_today_pitchers.to_csv("Leagues/MLB/data/today_pitchers.csv", index=False)
    print(df_today_pitchers)
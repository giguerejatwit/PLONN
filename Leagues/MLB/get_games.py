import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime


def scrape_today_pitchers() -> pd.DataFrame:
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
            team1_strong = rows[0].find("strong")
            team1_link = rows[0].find("a")

            team2_strong = rows[1].find("strong")
            team2_link = rows[1].find("a")

            team1_abbr = team1_strong.text.strip() if team1_strong else rows[0].text.split()[0]
            team1_pitcher = team1_link.text.strip() if team1_link else "TBD"

            team2_abbr = team2_strong.text.strip() if team2_strong else rows[1].text.split()[0]
            team2_pitcher = team2_link.text.strip() if team2_link else "TBD"

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
            print(f"Error parsing table: {e}")
            print("Row 1:", rows[0].text.strip())
            print("Row 2:", rows[1].text.strip())
            print("-" * 50)
            continue


    games = pd.DataFrame(games)
    games.to_csv("Leagues/MLB/data/today_pitchers.csv", index=False)
    return games


if __name__ == "__main__":
    # pass
    df_today_pitchers = scrape_today_pitchers()
    df_today_pitchers.to_csv("Leagues/MLB/data/today_pitchers.csv", index=False)
    print(df_today_pitchers)
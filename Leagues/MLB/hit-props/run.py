import argparse
from MLB.Models.nn import train
from MLB.StatsApi import get_team_streaks
from MLB.team_names import team_name_to_abbr

parser = argparse.ArgumentParser(
    prog="preprocessing.py", usage='Fetches data for selected Batting')
parser.add_argument('-t', '--team', required=False,
                    type=str, help='team abbreviation')
args = parser.parse_args()

# Get team W streaks
standings = get_team_streaks()
w_strk = standings[standings['Strk'].str.startswith('W')]
strk = standings[['Tm', 'Strk']]

if __name__ == '__main__':
    print(strk)
    if args.team is None:
        hot_teams = strk['Tm'].map(team_name_to_abbr)
        for team_name, streak in zip(strk['Tm'], strk['Strk']): # Iterate over both team names and streaks
            team_abbr = team_name_to_abbr.get(team_name, None)
            if team_abbr:
                print(f'------ | Training... | Team: {team_abbr} | Streak: {streak} ------')
                _, predictions = train(team_name=team_abbr)
                print(predictions)
                print ('\n')
            else:
                print(f'Error: Abbreviation for team "{team_name}" not found')
    else:
        print(f'------ | Attempting Training | Team: {args.team} ------')
        _, predictions = train(team_name=args.team)
        print(predictions)

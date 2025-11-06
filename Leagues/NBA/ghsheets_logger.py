import os
from datetime import datetime
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
def data_to_googlesheets(data, model, sheet_name='PLONN V2') -> None:
    """
    Add data into the NBA google spread sheet
    """
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    scopes = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

    creds = Credentials.from_service_account_file(
        creds_path, scopes=scopes)
    client = gspread.authorize(creds)

    # Raw Sheet ID
    sheet_id = '1X23ZAHpCAt1ksR4-vLkj8TjdUwpaxpPh2iuNqp6JXHs'
    sheet = client.open_by_key(sheet_id)
    raw_sheet = sheet.worksheet(sheet_name)

    # Count columns based on the first row
    last_row = len(raw_sheet.get_all_values())
    raw_sheet.insert_row([""] * len(raw_sheet.row_values(1)), last_row + 1)
    if model == 'adv':
        todays_games_formatted = pd.DataFrame({
            # Add today's date
            "date": [datetime.today().strftime("%m/%d/%Y")] * len(data),
            "home": data["home_team"],
            "away": data["away_team"],
            "home pred": data["home_predicted_scores"],  # Round scores
            "away pred": data["away_predicted_scores"],
            'pred total': (data["home_predicted_scores"] + data["away_predicted_scores"]),
            "dk lines": data["dk lines"],
        }).sort_values(by='pred total', ascending=False)
    
    elif model == 'V2':
        todays_games_formatted = pd.DataFrame({
            # Add today's date
            "date": [datetime.today().strftime("%m/%d/%Y")] * len(data),
            "home": data["home_team"],
            "away": data["away_team"],
            'pred total': (data["predicted_total"]),
            "edge": data["edge"],  # Round scores
            "dk lines": data["dk lines"],
            "O/U": data["O/U"],
        }).sort_values(by='pred total', ascending=False)

    todays_games_formatted = todays_games_formatted.replace([np.nan, np.inf, -np.inf], "").fillna(0)
    data_to_upload = todays_games_formatted.values.tolist()
    raw_sheet.insert_rows(data_to_upload, row=last_row + 2)
    print(
        f'✅ Successfully appended `todays_games` predictions to the {sheet_name} sheet!')

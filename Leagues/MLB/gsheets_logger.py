from google.oauth2.service_account import Credentials
from datetime import datetime
import gspread
import numpy as np
import os
import pandas as pd
from predict_lstm import pred_df as lstm_pred_df
from predict_mlp import pred_df as mlp_pred_df
from get_games import scrape_today_pitchers

def data_to_googlesheets(data, sheet_name='MLP') -> None:
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
    sheet_id = '1VsQqLyLtnQcgL4Jp02ydbGdLGvO71xATtlxc30H3v4c'
    sheet = client.open_by_key(sheet_id)
    raw_sheet = sheet.worksheet(sheet_name)

    # Count columns based on the first row
    last_row = len(raw_sheet.get_all_values())
    raw_sheet.insert_row([""] * len(raw_sheet.row_values(1)), last_row + 1)

    todays_games_formatted = pd.DataFrame({
        # Add today's date
        "Date": [datetime.today().strftime("%m/%d/%Y")] * len(data),
        "Pitcher": data["Pitcher"],
        "Team": data["Team"],  # Round scores
        "Opp": data["Opponent"],
        "Pred K": data["Predicted_Ks"].round(2),
        
    })

    todays_games_formatted = todays_games_formatted.replace([np.nan, np.inf, -np.inf], "")

    data_to_upload = todays_games_formatted.values.tolist()
    raw_sheet.insert_rows(data_to_upload, row=last_row + 2)
    print(
        f'✅ Strikeout data successfully appended predictions to the {sheet_name} sheet!')
    

if __name__ == "__main__":
    scrape_today_pitchers()
    
    
    data_to_googlesheets(lstm_pred_df, sheet_name='LSTM')
    data_to_googlesheets(mlp_pred_df, sheet_name='MLP')
    
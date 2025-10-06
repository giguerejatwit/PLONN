from google.oauth2.service_account import Credentials
from datetime import datetime
import gspread
import numpy as np
import os
import pandas as pd
# from Leagues.MLB.predict_lstm import pred_df as lstm_pred_df
# from Leagues.MLB.predict_mlp import pred_df as mlp_pred_df
# from Leagues.MLB.get_games import scrape_today_pitchers


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

    todays_games_formatted = todays_games_formatted.replace(
        [np.nan, np.inf, -np.inf], "")

    data_to_upload = todays_games_formatted.values.tolist()
    raw_sheet.insert_rows(data_to_upload, row=last_row + 2)
    print(
        f'✅ Strikeout data successfully appended predictions to the {sheet_name} sheet!')


def runs_to_gsheets(data, sheetname="Runs") -> None:
    """
    Add data into the MLB google spread sheet
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
    raw_sheet = sheet.worksheet(sheetname)

    # Count columns based on the first row
    # Insert an empty row (your existing logic)
    last_row = len(raw_sheet.get_all_values())
    raw_sheet.insert_row([""] * len(raw_sheet.row_values(1)), last_row + 1)

    # Build base DataFrame from columns we know should exist
    todays_games_formatted = pd.DataFrame({
        "Date": [datetime.today().strftime("%Y-%m-%d")] * len(data),
        "Team_1": data["Team_1"],
        "Runs_1": data["Runs_1"],
        "Team_2": data["Team_2"],
        "Runs_2": data["Runs_2"],
        "Total_Runs": data["Total_Runs"],
    })

    # SAFELY add DK_Line if present; coerce to numeric and replace NaNs with empty string for display
    if 'DK_Line' in data.columns:
        # coerce to numeric so downstream consumers know it's numeric
        todays_games_formatted['DK_Line'] = pd.to_numeric(
            data['DK_Line'], errors='coerce')
        # Keep a display-friendly string column (optional)
        todays_games_formatted['DK_Line_str'] = todays_games_formatted['DK_Line'].astype(
            'Float64').astype(str).replace('<NA>', '')
    else:
        # no DK lines: create empty DK_Line and DK_Line_str so the sheet columns exist
        todays_games_formatted['DK_Line'] = pd.NA
        todays_games_formatted['DK_Line_str'] = ""

    # If you had a DK_Diff earlier, compute only if DK_Line numeric exists in the source
    # todays_games_formatted['DK_Diff'] = (todays_games_formatted['DK_Line'] - todays_games_formatted['Total_Runs'])

    # Clean up infinities / NaNs for upload (use empty strings for display fields)
    todays_games_formatted = todays_games_formatted.replace(
        [np.inf, -np.inf], pd.NA)
    # Convert NA to empty string for uploading (optional)
    todays_games_formatted = todays_games_formatted.fillna("")

    # Sort
    todays_games_formatted = todays_games_formatted.sort_values(
        by='Total_Runs', ascending=False)

    data_to_upload = todays_games_formatted.values.tolist()
    raw_sheet.insert_rows(data_to_upload, row=last_row + 2)
    print(
        f'✅ Runs data successfully appended predictions to the {sheetname} sheet!')


if __name__ == "__main__":
    # scrape_today_pitchers()

    pass
    # data_to_googlesheets(lstm_pred_df, sheet_name='LSTM')
    # data_to_googlesheets(mlp_pred_df, sheet_name='MLP')

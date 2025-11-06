
import argparse
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from ghsheets_logger import data_to_googlesheets
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, callbacks, layers, regularizers
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

from utils.odds_api import get_nba_dk_lines
from get_data import get_team_per_game_stats, get_today_games
from utils.abbr_map import get_team_name_or_abbr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = "leagues/NBA/data/offsets/offset_with_dk_final.csv"

MODEL_PATH = "leagues/NBA/models/total_model.keras"
SCALER_PATH = "leagues/NBA/models/total_scaler.pkl"
FEATURES_PATH = "leagues/NBA/models/total_features.json"
CALIBRATION_PATH = "leagues/NBA/models/total_calibration.json"
# Default ADV/TPG workbook path for date-based predictions
BACKTEST_XLSX_DEFAULT = "leagues/NBA/data/tpgOct26.xlsx"

# Base stat features (home = no suffix, away = ".1")
STATS_FEATURE_COLUMNS = [
    # Offensive / defensive ratings
    "ORtg", "ORtg.1",
    "DRtg", "DRtg.1",
    # Shooting / efficiency
    "FTr", "FTr.1",
    "3PAr", "3PAr.1",
    "TS%", "TS%.1",
    "eFG%", "eFG%.1",
    "FT/FGA", "FT/FGA.1",
    # Tempo
    "Pace", "Pace.1",
]

DK_COL_CANDIDATES = [
    "dk_total_point",
    "DK_Line",
    "dk lines",
    "dk_line",
    "total_line",
]

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def compute_pace(stats: dict, opp_stats: dict, suffix: str = "") -> float:
    """Single-game pace estimate from a stats dict and opponent stats.

    This is used for building matchup feature vectors at prediction time.
    """
    fga = stats.get(f"FGA{suffix}", 0)
    fta = stats.get(f"FTA{suffix}", 0)
    fg_pct = stats.get(f"FG%{suffix}", 0)
    fg = fg_pct * fga

    orb = stats.get(f"ORB{suffix}", 0)
    trb = stats.get(f"TRB{suffix}", 0)
    tov = stats.get(f"TOV{suffix}", 0)

    # Defensive rebounds for the opponent
    drb_opp = opp_stats.get(f"TRB{'.1' if suffix == '' else ''}", 0) - opp_stats.get(
        f"ORB{'.1' if suffix == '' else ''}", 0
    )
    orb_denom = orb + drb_opp
    orb_factor = orb / orb_denom if orb_denom else 0

    pace = fga + 0.4 * fta - 1.07 * orb_factor * (fga - fg) + tov
    return pace


def calculate_team_pace(data: pd.DataFrame) -> pd.DataFrame:
    """Vectorized pace calculation for the historical training data.

    Adds `Pace` and `Pace.1` columns in-place.
    """
    data = data.copy()

    # Estimate FG if missing
    if "FG" not in data.columns:
        data["FG"] = data["FG%"] * data["FGA"]
    if "FG.1" not in data.columns:
        data["FG.1"] = data["FG%.1"] * data["FGA.1"]

    # Defensive rebounds
    data["DRB"] = data["TRB"] - data["ORB"]
    data["DRB.1"] = data["TRB.1"] - data["ORB.1"]

    # Safe possessions calculation
    team_possessions = (
        data["FGA"]
        + 0.4 * data["FTA"]
        - 1.07
        * (data["ORB"] / (data["ORB"] + data["DRB.1"]).replace(0, np.nan))
        * (data["FGA"] - data["FG"])
        + data["TOV"]
    )

    opp_possessions = (
        data["FGA.1"]
        + 0.4 * data["FTA.1"]
        - 1.07
        * (data["ORB.1"] / (data["ORB.1"] + data["DRB"]).replace(0, np.nan))
        * (data["FGA.1"] - data["FG.1"])
        + data["TOV.1"]
    )

    data["Pace"] = team_possessions
    data["Pace.1"] = opp_possessions

    return data


def detect_and_normalize_dk_col(df: pd.DataFrame) -> str | None:
    """Detect any DK total column, normalize its name to `dk_total_point`.

    Returns the normalized column name or None if not found.
    """
    for col in DK_COL_CANDIDATES:
        if col in df.columns:
            if col != "dk_total_point":
                df.rename(columns={col: "dk_total_point"}, inplace=True)
            return "dk_total_point"
    return None


def parse_total_from_result(df: pd.DataFrame) -> pd.Series:
    """Parse the `Result` column (e.g., "W, 117-103") into total points."""
    if "Result" not in df.columns:
        raise ValueError("Expected a 'Result' column like 'W, 117-103'.")

    score_match = df["Result"].str.extract(r"(\d+)-(\d+)")
    if score_match.isnull().any().any():
        raise ValueError("Failed to parse some scores from 'Result' column.")

    team_pts = score_match[0].astype(float)
    opp_pts = score_match[1].astype(float)
    total_pts = team_pts + opp_pts
    return total_pts


def save_scaler(scaler: StandardScaler, path: str = SCALER_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str = SCALER_PATH) -> StandardScaler:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing scaler file: {path}")
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def save_feature_list(feature_list: list[str], path: str = FEATURES_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(feature_list, f)


def load_feature_list(path: str = FEATURES_PATH) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing feature list file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
def save_calibration(slope: float, intercept: float, path: str = CALIBRATION_PATH) -> None:
    """Persist linear calibration parameters (y ≈ slope * y_hat + intercept)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"slope": float(slope), "intercept": float(intercept)}, f)


def load_calibration(path: str = CALIBRATION_PATH) -> tuple[float, float] | None:
    """Load calibration parameters if available; return (slope, intercept) or None."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "slope" in data and "intercept" in data:
        return float(data["slope"]), float(data["intercept"])
    return None


def apply_calibration(preds: np.ndarray, path: str = CALIBRATION_PATH) -> np.ndarray:
    """Apply saved linear calibration to raw predictions if calibration exists."""
    params = load_calibration(path)
    if params is None:
        return preds
    slope, intercept = params
    return slope * preds + intercept


# ---------------------------------------------------------------------------
# Helper: resolve column names case-insensitively
# ---------------------------------------------------------------------------
def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first column name in df that matches any candidate (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    raise KeyError(f"None of the candidate columns {candidates} found in DataFrame: {list(df.columns)}")
# ---------------------------------------------------------------------------
# Backtest utilities: ADV/TPG Excel evaluation
# ---------------------------------------------------------------------------


def load_adv_tpg_sheets(xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ADV and TPG sheets from a backtest workbook."""
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Backtest file not found at {xlsx_path}")
    xls = pd.ExcelFile(xlsx_path)
    sheet_map = {name.lower(): name for name in xls.sheet_names}
    adv_name = sheet_map.get("adv")
    tpg_name = sheet_map.get("tpg")
    if adv_name is None or tpg_name is None:
        raise ValueError(f"Expected sheets named 'ADV' and 'TPG' (case-insensitive). Found: {xls.sheet_names}")
    adv_df = xls.parse(adv_name)
    tpg_df = xls.parse(tpg_name)
    return adv_df, tpg_df


# ---------------------------------------------------------------------------
# Helper: enrich TPG with derived advanced stats for backtests
# ---------------------------------------------------------------------------
def enrich_tpg_with_advanced(tpg: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure TPG has the advanced stats needed for STATS_FEATURE_COLUMNS.

    If columns like FTr, 3PAr, TS%, FT/FGA are missing but their base stats
    (FGA, FTA, 3PA, PTS, FT) exist, compute and add them.
    """
    tpg = tpg.copy()

    def col_or_none(candidates: list[str]) -> str | None:
        try:
            return resolve_column(tpg, candidates)
        except KeyError:
            return None

    fga_col = col_or_none(["FGA", "fga"])
    fta_col = col_or_none(["FTA", "fta"])
    three_pa_col = col_or_none(["3PA", "3PA.2", "3PA per G", "3PA/G"])
    pts_col = col_or_none(["PTS", "pts"])
    ftm_col = col_or_none(["FT", "ft", "FTM", "ftm"])

    # Free throw rate: FTr = FTA / FGA
    if "FTr" not in tpg.columns and fga_col and fta_col:
        with np.errstate(divide="ignore", invalid="ignore"):
            tpg["FTr"] = tpg[fta_col] / tpg[fga_col]

    # Three-point attempt rate: 3PAr = 3PA / FGA
    if "3PAr" not in tpg.columns and fga_col and three_pa_col:
        with np.errstate(divide="ignore", invalid="ignore"):
            tpg["3PAr"] = tpg[three_pa_col] / tpg[fga_col]

    # True shooting percentage: TS% = PTS / (2 * (FGA + 0.44 * FTA))
    if "TS%" not in tpg.columns and pts_col and fga_col and fta_col:
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = 2.0 * (tpg[fga_col] + 0.44 * tpg[fta_col])
            tpg["TS%"] = tpg[pts_col] / denom

    # FT/FGA
    if "FT/FGA" not in tpg.columns and ftm_col and fga_col:
        with np.errstate(divide="ignore", invalid="ignore"):
            tpg["FT/FGA"] = tpg[ftm_col] / tpg[fga_col]

    # Approximate Pace (possessions per game) if we have enough stats
    # Using a simplified version of the same formula as calculate_team_pace:
    # possessions ≈ FGA + 0.4 * FTA - 1.07 * ORB_factor * (FGA - FG) + TOV
    # where ORB_factor ≈ ORB / (ORB + DRB) and DRB = TRB - ORB.
    if "Pace" not in tpg.columns:
        fg_col = col_or_none(["FG", "fg"])
        fg_pct_col = col_or_none(["FG%", "fg%"])
        orb_col = col_or_none(["ORB", "orb"])
        trb_col = col_or_none(["TRB", "trb"])
        tov_col = col_or_none(["TOV", "tov"])

        if fga_col and fta_col and tov_col and orb_col and trb_col and (fg_col or fg_pct_col):
            with np.errstate(divide="ignore", invalid="ignore"):
                if fg_col:
                    fg_vals = tpg[fg_col]
                else:
                    # Estimate made shots from FG% * FGA if FG is missing
                    fg_vals = tpg[fg_pct_col] * tpg[fga_col]

                drb_vals = tpg[trb_col] - tpg[orb_col]
                orb_denom = tpg[orb_col] + drb_vals
                orb_factor = np.where(orb_denom != 0, tpg[orb_col] / orb_denom, 0.0)

                pace_vals = (
                    tpg[fga_col]
                    + 0.4 * tpg[fta_col]
                    - 1.07 * orb_factor * (tpg[fga_col] - fg_vals)
                    + tpg[tov_col]
                )
                tpg["Pace"] = pace_vals

    return tpg


def build_backtest_feature_matrix_from_tpg(
    adv_df: pd.DataFrame,
    tpg_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build feature matrix for backtesting using ADV (matchups) and TPG (team averages).

    - ADV must have: home, away, dk lines, Total Actual
    - TPG must have team-level stats for the base feature columns (no .1 suffix)
    """
    # Use enriched advanced stats for TPG to fill in derived columns
    adv = adv_df.copy()
    tpg = tpg_df.copy()
    tpg = enrich_tpg_with_advanced(tpg)

    # Resolve critical columns in ADV
    home_col = resolve_column(adv, ["home"])
    away_col = resolve_column(adv, ["away"])
    dk_col_adv = resolve_column(adv, ["dk lines", "dk_total_point", "dk_line", "DK_Line"])
    actual_col = resolve_column(adv, ["Total Actual", "total_actual", "Actual Total"])

    # Normalize DK column name
    adv["dk_total_point"] = adv[dk_col_adv].astype(float)
    adv["Total_Actual"] = adv[actual_col].astype(float)

    # Resolve team column in TPG
    team_col = resolve_column(tpg, ["Team", "team", "Team_full", "team_full", "Tm"])
    tpg["__team_key"] = tpg[team_col].astype(str).str.strip().str.lower()

    # Prepare lookups
    adv["__home_key"] = adv[home_col].astype(str).str.strip().str.lower()
    adv["__away_key"] = adv[away_col].astype(str).str.strip().str.lower()

    # Base feature names without .1; exclude dk_total_point (comes from ADV)
    base_cols = [c for c in feature_cols if not c.endswith(".1") and c != "dk_total_point"]
    away_cols = [c for c in feature_cols if c.endswith(".1")]

    rows = []
    kept_indices = []

    for idx, row in adv.iterrows():
        hk = row["__home_key"]
        ak = row["__away_key"]

        home_row = tpg[tpg["__team_key"] == hk]
        away_row = tpg[tpg["__team_key"] == ak]

        if home_row.empty or away_row.empty:
            # Skip games we can't map cleanly
            continue

        home_row = home_row.iloc[0]
        away_row = away_row.iloc[0]

        feat = {}

        # Fill home/away stats
        for col in base_cols:
            # base column name in TPG is assumed to be identical
            if col in tpg.columns:
                feat[col] = float(home_row[col])
            else:
                # if missing, fill 0 and warn once (optional)
                feat.setdefault("_missing_warned", set())
                if col not in feat["_missing_warned"]:
                    print(f"[WARN] TPG missing column '{col}', filling with 0.0")
                    feat["_missing_warned"].add(col)
                feat[col] = 0.0

        for col in away_cols:
            base = col.replace(".1", "")
            if base in tpg.columns:
                feat[col] = float(away_row[base])
            else:
                feat.setdefault("_missing_warned", set())
                if base not in feat["_missing_warned"]:
                    print(f"[WARN] TPG missing column '{base}' for away, filling with 0.0")
                    feat["_missing_warned"].add(base)
                feat[col] = 0.0

        # DK feature if used in training
        if "dk_total_point" in feature_cols:
            feat["dk_total_point"] = float(row["dk_total_point"])

        # Drop internal helper
        if "_missing_warned" in feat:
            feat.pop("_missing_warned", None)

        rows.append(feat)
        kept_indices.append(idx)

    if not rows:
        raise ValueError("No backtest rows could be built from ADV/TPG; team name mismatch?")

    features_df = pd.DataFrame(rows)
    # Ensure all expected feature columns exist
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        print(f"[WARN] Missing columns in backtest feature set, filling with 0: {missing}")
        for c in missing:
            features_df[c] = 0.0

    # Order columns to match training
    features_df = features_df[feature_cols]

    adv_kept = adv.loc[kept_indices].reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    return features_df, adv_kept


def backtest_from_excel(xlsx_path: str) -> None:
    """Run a backtest on an Excel workbook with ADV + TPG sheets."""
    try:
        feature_cols = load_feature_list()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}. Train the model first with -t.")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Missing model file: {MODEL_PATH}. Train the model first with -t.")
        return

    try:
        scaler = load_scaler()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}. Train the model first with -t.")
        return

    adv_df, tpg_df = load_adv_tpg_sheets(xlsx_path)
    features_df, adv_kept = build_backtest_feature_matrix_from_tpg(adv_df, tpg_df, feature_cols)

    X = features_df.values.astype(float)
    X_scaled = scaler.transform(X)

    # IMPORTANT: we must scale with the same scaler used in training
    preds = model.predict(X_scaled).flatten()

    total_actual = adv_kept["Total_Actual"].values.astype(float)
    dk_lines = adv_kept["dk_total_point"].values.astype(float)

    # Overall metrics (raw model predictions)
    model_mae = mean_absolute_error(total_actual, preds)
    model_rmse = np.sqrt(mean_squared_error(total_actual, preds))
    dk_mae = mean_absolute_error(total_actual, dk_lines)
    dk_rmse = np.sqrt(mean_squared_error(total_actual, dk_lines))

    print("\n[BACKTEST] Overall performance on ADV sheet:")
    print(f"Model (raw) MAE: {model_mae:.2f}, RMSE: {model_rmse:.2f}")
    print(f"DK         MAE: {dk_mae:.2f}, RMSE: {dk_rmse:.2f}")

    # Fit a simple linear calibration y ≈ a * y_hat + b if we have enough data
    calibrated = None
    if len(total_actual) >= 20 and np.var(preds) > 1e-6:
        p_mean = preds.mean()
        y_mean = total_actual.mean()
        cov = np.mean((preds - p_mean) * (total_actual - y_mean))
        var_p = np.var(preds)
        a = cov / var_p if var_p > 0 else 1.0
        b = y_mean - a * p_mean
        calibrated = a * preds + b

        cal_mae = mean_absolute_error(total_actual, calibrated)
        cal_rmse = np.sqrt(mean_squared_error(total_actual, calibrated))
        print(f"Model (cal) MAE: {cal_mae:.2f}, RMSE: {cal_rmse:.2f}")
        print(f"[BACKTEST] Learned calibration: y ≈ {a:.4f} * y_hat + {b:.2f}")
        save_calibration(a, b)
    else:
        print("[BACKTEST] Not enough data to fit calibration; skipping calibration fit.")

    # Use calibrated predictions for edge analysis if available
    preds_for_edge = calibrated if calibrated is not None else preds

    # Edge-based analysis
    edge = preds_for_edge - dk_lines
    abs_edge = np.abs(edge)

    bins = [(0, 2), (2, 4), (4, 6), (6, np.inf)]
    print("\n[BACKTEST] Performance by absolute edge size (|model - DK|):")
    for low, high in bins:
        mask = (abs_edge >= low) & (abs_edge < high)
        if not mask.any():
            continue
        m_mae = mean_absolute_error(total_actual[mask], preds_for_edge[mask])
        d_mae = mean_absolute_error(total_actual[mask], dk_lines[mask])

        # O/U hit rate for model vs DK
        # Model bets: Over if edge > 0, Under if edge < 0
        model_side = np.where(edge[mask] > 0, "O", "U")
        actual_side = np.where(
            total_actual[mask] > dk_lines[mask],
            "O",
            np.where(total_actual[mask] < dk_lines[mask], "U", "P"),
        )
        valid = actual_side != "P"
        hits = (model_side[valid] == actual_side[valid]).sum()
        total_bets = valid.sum()
        win_rate = hits / total_bets if total_bets > 0 else 0.0

        label = f"{low}-{high if high != np.inf else '+'}"
        print(
            f"  |edge| in [{label}] → rows={mask.sum()}, "
            f"Model MAE={m_mae:.2f}, DK MAE={d_mae:.2f}, "
            f"Model win-rate={win_rate*100:.1f}%"
        )


# ---------------------------------------------------------------------------
# Calibration scatter graph helper
# ---------------------------------------------------------------------------
def calibration_graph_from_excel(xlsx_path: str = BACKTEST_XLSX_DEFAULT) -> None:
    """Plot a calibration scatter graph (raw preds vs actual) from ADV/TPG workbook.

    This mirrors the backtest data pipeline but only produces a scatter plot of
    raw model predictions vs actual totals, along with the fitted calibration
    line parameters.
    """
    # Load feature list, model, and scaler
    try:
        feature_cols = load_feature_list()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}. Train the model first with -t.")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Missing model file: {MODEL_PATH}. Train the model first with -t.")
        return

    try:
        scaler = load_scaler()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}. Train the model first with -t.")
        return

    # Load ADV/TPG sheets
    try:
        adv_df, tpg_df = load_adv_tpg_sheets(xlsx_path)
    except Exception as e:
        print(f"[ERROR] Failed to load ADV/TPG workbook '{xlsx_path}': {e}")
        return

    # Build feature matrix as in backtest
    try:
        features_df, adv_kept = build_backtest_feature_matrix_from_tpg(adv_df, tpg_df, feature_cols)
    except Exception as e:
        print(f"[ERROR] Failed to build feature matrix for calibration graph: {e}")
        return

    X = features_df.values.astype(float)
    X_scaled = scaler.transform(X)

    # Raw model predictions (before calibration)
    preds = model.predict(X_scaled).flatten()
    total_actual = adv_kept["Total_Actual"].values.astype(float)

    if len(total_actual) == 0:
        print("[ERROR] No rows available to plot calibration graph.")
        return

    # Fit linear calibration y ≈ a * y_hat + b (same as backtest)
    p_mean = preds.mean()
    y_mean = total_actual.mean()
    cov = np.mean((preds - p_mean) * (total_actual - y_mean))
    var_p = np.var(preds)
    if var_p <= 0:
        print("[ERROR] Zero variance in predictions; cannot fit calibration line.")
        return

    a = cov / var_p
    b = y_mean - a * p_mean

    # Compute and print diagnostics for raw vs calibrated predictions
    calibrated = a * preds + b
    raw_rmse = np.sqrt(mean_squared_error(total_actual, preds))
    cal_rmse = np.sqrt(mean_squared_error(total_actual, calibrated))
    raw_r2 = r2_score(total_actual, preds)
    cal_r2 = r2_score(total_actual, calibrated)

    print(f"[CG] Raw:   RMSE={raw_rmse:.2f}, R^2={raw_r2:.3f}")
    print(f"[CG] Calib: RMSE={cal_rmse:.2f}, R^2={cal_r2:.3f}")

    # Scatter plot: raw preds vs actual totals, with calibration and identity lines
    plt.figure(figsize=(7, 7))
    plt.scatter(preds, total_actual, alpha=0.6, label="Games (raw vs actual)")

    # Plot identity and calibration lines
    x_line = np.linspace(preds.min(), preds.max(), 100)
    y_identity = x_line
    y_calibrated = a * x_line + b
    plt.plot(x_line, y_identity, linestyle="--", color="gray", label="y = x (no calibration)")
    plt.plot(x_line, y_calibrated, color="red", label=f"y = {a:.2f}x + {b:.1f}")

    plt.xlabel("Raw model prediction (y_hat)")
    plt.ylabel("Actual total points (y)")
    plt.title(f"Calibration scatter: y ≈ {a:.2f} * y_hat + {b:.1f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Data preparation for total-points model
# ---------------------------------------------------------------------------


def prepare_training_data(
    data_path: str = DATA_PATH,
    # sheet_name: str = SHEET_NAME,
    window: int = 30,
    min_games: int = 10,
):
    """Load historical data and build X/y for total-points regression.

    - Computes PACE for both teams
    - Parses total points from `Result`
    - Builds season labels
    - Computes rolling 30-game means for each team & opponent within a season
    - Uses DK total as a feature when available
    """

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}")

    df = pd.read_csv(data_path, header=0)

    # Drop unnamed columns and non-game rows
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    if "Rk" in df.columns:
        df = df[df["Rk"].apply(lambda x: str(x).isdigit())]

    df = df.copy().reset_index(drop=True)

    required_base_cols = {"Date", "Team", "Opp", "Result"}
    missing = required_base_cols - set(df.columns)
    if missing:
        raise ValueError(f"prepare_training_data: missing required columns: {missing}")

    # Ensure datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Pace for both teams
    df = calculate_team_pace(df)

    # Normalize DK column name (if present)
    dk_col = detect_and_normalize_dk_col(df)
    if dk_col:
        print(f"[INFO] Using DK column: {dk_col}")

    # Total points target
    df["total_points"] = parse_total_from_result(df)

    # Season label: Oct–Apr is one NBA season
    month = df["Date"].dt.month
    year = df["Date"].dt.year

    season_start_year = np.where(month >= 10, year, year - 1).astype(int)
    season_end_year = season_start_year + 1

    # Build as pandas Series to avoid NumPy string ufunc issues
    start_s = pd.Series(season_start_year, index=df.index).astype(str)
    end_s = pd.Series(season_end_year, index=df.index).astype(str)
    df["Season"] = start_s + "-" + end_s

    # Prepare rolling 30-game histories
    home_cols = [c for c in STATS_FEATURE_COLUMNS if not c.endswith(".1")]
    away_cols = [c for c in STATS_FEATURE_COLUMNS if c.endswith(".1")]

    df = df.sort_values("Date").reset_index(drop=True)
    df["Team_games"] = df.groupby(["Team", "Season"]).cumcount()
    df["Opp_games"] = df.groupby(["Opp", "Season"]).cumcount()

    def roll30(g: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        g = g.sort_values("Date")
        return g[cols].rolling(window=window, min_periods=min_games).mean().shift(1)

    home_roll = df.groupby(["Team", "Season"], group_keys=False).apply(
        lambda g: roll30(g, home_cols)
    )
    away_roll = df.groupby(["Opp", "Season"], group_keys=False).apply(
        lambda g: roll30(g, away_cols)
    )

    for c in home_cols:
        df[c] = home_roll[c]
    for c in away_cols:
        df[c] = away_roll[c]

    feature_cols = list(STATS_FEATURE_COLUMNS)
    if dk_col:
        feature_cols.append("dk_total_point")

    # Filter rows with enough history and no NaNs
    features = df[feature_cols]
    target = df["total_points"]

    mask = (
        (df["Team_games"] >= min_games)
        & (df["Opp_games"] >= min_games)
        & (~features.isnull().any(axis=1))
        & target.notnull()
    )

    df_kept = df.loc[mask].copy()
    if df_kept.empty:
        raise ValueError("No rows left after applying history / NaN filters.")

    print(
        f"[INFO] prepare_training_data: seasons={df_kept['Season'].nunique()}, "
        f"rows_kept={len(df_kept)}"
    )

    # Time-based split: last 20% of dates is test
    dates_kept = df_kept["Date"]
    cutoff = dates_kept.quantile(0.8)
    train_mask = dates_kept <= cutoff

    X_train = df_kept.loc[train_mask, feature_cols].values.astype(float)
    y_train = df_kept.loc[train_mask, "total_points"].values.astype(float)
    X_test = df_kept.loc[~train_mask, feature_cols].values.astype(float)
    y_test = df_kept.loc[~train_mask, "total_points"].values.astype(float)

    print(f"[INFO] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Persist the exact feature column order for inference
    save_feature_list(feature_cols)

    return X_train, X_test, y_train, y_test, feature_cols


# ---------------------------------------------------------------------------
# Model definition and training
# ---------------------------------------------------------------------------


def build_total_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="linear"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=Huber(delta=5.0),
        metrics=["mae"],
    )
    return model


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    graph: bool = False,
):
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_total_model(X_train_scaled.shape[1])

    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=8, verbose=1
    )
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        shuffle=True,
        callbacks=[lr_scheduler, early_stopping],
        verbose=1,
    )

    # Evaluation
    y_pred = model.predict(X_test_scaled).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")

    # Baseline: mean-only
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    print(f"Baseline (mean-only) MAE: {baseline_mae:.2f}")
    print(f"Baseline (mean-only) RMSE: {baseline_rmse:.2f}")

    # Baseline: use DK total line directly if available as a feature
    if "dk_total_point" in feature_cols:
        dk_idx = feature_cols.index("dk_total_point")
        dk_test = X_test[:, dk_idx]
        dk_mae = mean_absolute_error(y_test, dk_test)
        dk_rmse = np.sqrt(mean_squared_error(y_test, dk_test))
        print(f"Baseline (DK line) MAE: {dk_mae:.2f}")
        print(f"Baseline (DK line) RMSE: {dk_rmse:.2f}")

    # Persist model and scaler
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[INFO] Saved total model to {MODEL_PATH}")

    save_scaler(scaler)
    print(f"[INFO] Saved scaler to {SCALER_PATH}")

    if graph:
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test, label="Actual", alpha=0.7)
        plt.scatter(range(len(y_test)), y_pred, label="Predicted", alpha=0.7)
        plt.xlabel("Game Index")
        plt.ylabel("Total Points")
        plt.title("Actual vs Predicted Totals (Test Set)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return history, model


# ---------------------------------------------------------------------------
# Prediction for today's games
# ---------------------------------------------------------------------------


def create_matchup_feature_vector(home_team: str, away_team: str) -> dict:
    """Build a single feature dict for a matchup (home vs away).

    Uses per-game stats from `get_team_per_game_stats` and derives Pace and
    advanced shooting stats so that all STATS_FEATURE_COLUMNS are populated.
    """
    home_abbr = get_team_name_or_abbr(home_team)
    away_abbr = get_team_name_or_abbr(away_team)

    # Base per-game stats from the local TPG workbook (or scraper)
    home_stats = get_team_per_game_stats(home_abbr, args="adv")
    away_base = get_team_per_game_stats(away_abbr, args="adv")

    def _add_advanced_inplace(s: dict) -> None:
        """Derive FTr, 3PAr, TS%, eFG%, FT/FGA from basic box score stats.

        Expected keys in `s`: PTS, FGA, FTA, 3PA, FG%, 3P%, FT% (as rates).
        Missing values are treated as 0.
        """
        pts = float(s.get("PTS", 0) or 0)
        fga = float(s.get("FGA", 0) or 0)
        fta = float(s.get("FTA", 0) or 0)
        three_pa = float(s.get("3PA", 0) or 0)
        fg_pct = float(s.get("FG%", 0) or 0)
        three_pct = float(s.get("3P%", 0) or 0)
        ft_pct = float(s.get("FT%", 0) or 0)

        # Free-throw rate and 3P attempt rate
        if fga > 0:
            s["FTr"] = fta / fga
            s["3PAr"] = three_pa / fga
        else:
            s["FTr"] = 0.0
            s["3PAr"] = 0.0

        # True shooting percentage: TS% = PTS / (2 * (FGA + 0.44 * FTA))
        denom_ts = 2.0 * (fga + 0.44 * fta)
        s["TS%"] = pts / denom_ts if denom_ts > 0 else 0.0

        # Approximate FT/FGA using FT made ≈ FT% * FTA
        if fga > 0:
            ft_made = ft_pct * fta
            s["FT/FGA"] = ft_made / fga
        else:
            s["FT/FGA"] = 0.0

        # Approximate eFG% using FG% and 3P volume & accuracy
        # eFG% = (FG + 0.5 * 3P) / FGA, with FG ≈ FG% * FGA, 3P ≈ 3PA * 3P%
        if fga > 0:
            fg_made = fg_pct * fga
            three_made = three_pa * three_pct
            s["eFG%"] = (fg_made + 0.5 * three_made) / fga
        else:
            s["eFG%"] = 0.0

    # Derive advanced stats for both home and away before suffixing
    _add_advanced_inplace(home_stats)
    _add_advanced_inplace(away_base)

    # Create suffixed away-team dict (".1" suffix for all keys)
    away_stats = {f"{k}.1": v for k, v in away_base.items()}

    # Compute Pace for both sides using the combined dictionaries
    try:
        home_stats["Pace"] = compute_pace(home_stats, away_stats, suffix="")
        away_stats["Pace.1"] = compute_pace(away_stats, home_stats, suffix=".1")
    except Exception as e:  # pragma: no cover - purely defensive
        print(f"[WARN] Failed to compute pace for {home_team} vs {away_team}: {e}")
        home_stats["Pace"] = 0.0
        away_stats["Pace.1"] = 0.0

    combined = {**home_stats, **away_stats}
    return combined


def merge_dk_totals(todays_games: pd.DataFrame) -> pd.DataFrame:
    """Fetch DK totals and merge into `todays_games` as `dk_total_point`."""
    try:
        dk = get_nba_dk_lines()
        if dk is None or dk.empty:
            print("[WARN] DK lines empty; continuing without dk_total_point.")
            return todays_games

        # Normalize column names and team names for join
        if "DK_Line" in dk.columns and "dk_total_point" not in dk.columns:
            dk = dk.rename(columns={"DK_Line": "dk_total_point"})

        # Some older helpers may still use 'dk lines'
        if "dk lines" in dk.columns and "dk_total_point" not in dk.columns:
            dk = dk.rename(columns={"dk lines": "dk_total_point"})

        for col in ["home_team", "away_team"]:
            if col not in dk.columns:
                raise ValueError("DK lines must contain 'home_team' and 'away_team' columns.")

        tg = todays_games.copy()

        def _norm_team(val: str) -> str:
            # Canonicalize to a common representation (e.g., 3-letter abbr)
            try:
                return get_team_name_or_abbr(str(val)).strip().lower()
            except Exception:
                return str(val).strip().lower()

        tg["__home"] = tg["home_team"].apply(_norm_team)
        tg["__away"] = tg["away_team"].apply(_norm_team)

        dk_local = dk[["home_team", "away_team", "dk_total_point"]].copy()
        dk_local["__home"] = dk_local["home_team"].apply(_norm_team)
        dk_local["__away"] = dk_local["away_team"].apply(_norm_team)

        tg = tg.merge(
            dk_local[["__home", "__away", "dk_total_point"]],
            on=["__home", "__away"],
            how="left",
        )
        tg = tg.drop(columns=["__home", "__away"])

        matched = tg["dk_total_point"].notna().sum()
        print(f"[INFO] Merged DK totals for {matched}/{len(tg)} games.")
        return tg

    except Exception as e:  # pragma: no cover - defensive
        print(f"[WARN] Failed to fetch/merge DK lines: {e}")
        return todays_games


def build_today_feature_matrix(feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features_df, games_df) for today's games.

    This mirrors the ADV/TPG backtest pipeline used by predict_date_totals:
    - Use October Schedule in the tpgOct26.xlsx workbook to get today's matchups
    - Merge in DraftKings totals via merge_dk_totals
    - Use the TPG sheet to supply team stats
    - Build the feature matrix via build_backtest_feature_matrix_from_tpg
    """
    # 1) Get today's games from the local schedule workbook
    todays_games = get_today_games()
    if todays_games.empty:
        print("No games today :(")
        return pd.DataFrame(), todays_games

    # 2) Attach DK totals using the Odds API
    todays_games = merge_dk_totals(todays_games)

    # 3) Load the TPG sheet from the same workbook used for backtests
    try:
        _, tpg_df = load_adv_tpg_sheets(BACKTEST_XLSX_DEFAULT)
    except Exception as e:
        print(f"[ERROR] Failed to load TPG sheet '{BACKTEST_XLSX_DEFAULT}': {e}")
        return pd.DataFrame(), todays_games

    # 4) Build an ADV-like DataFrame for today with home/away + DK lines.
    #    We do NOT have actual totals for today, so we supply a dummy
    #    'Total Actual' column that build_backtest_feature_matrix_from_tpg
    #    will simply carry through but not use for inference.
    adv_like = pd.DataFrame({
        "home": todays_games["home_team"],
        "away": todays_games["away_team"],
        "Total Actual": np.nan,  # placeholder; not used for prediction
        "dk lines": todays_games["dk_total_point"],
    })

    # 5) Reuse the same feature-building pipeline as backtests
    try:
        features_df, adv_kept = build_backtest_feature_matrix_from_tpg(adv_like, tpg_df, feature_cols)
    except Exception as e:
        print(f"[ERROR] Failed to build feature matrix for today's games: {e}")
        return pd.DataFrame(), todays_games

    # 6) Handle any missing DK totals in the feature matrix to avoid NaNs
    if "dk_total_point" in features_df.columns:
        if features_df["dk_total_point"].isna().all():
            print("[WARN] DK totals missing for all games; filling dk_total_point with 230.0 for model input.")
            features_df["dk_total_point"] = 230.0
        else:
            dk_mean = features_df["dk_total_point"].dropna().mean()
            features_df["dk_total_point"] = features_df["dk_total_point"].fillna(dk_mean)

    # The games_df we return should have home_team/away_team and dk_total_point.
    # adv_kept has 'home', 'away', and 'dk_total_point' columns by construction.
    games_df = adv_kept.rename(columns={"home": "home_team", "away": "away_team"})

    return features_df, games_df


def predict_today_totals() -> None:
    """Load model + scaler and print predictions for today's games."""
    try:
        feature_cols = load_feature_list()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}. Train the model first with -t.")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Missing model file: {MODEL_PATH}. Train the model first with -t.")
        return

    try:
        scaler = load_scaler()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}. Train the model first with -t.")
        return

    features_df, games_df = build_today_feature_matrix(feature_cols)
    if features_df.empty or games_df.empty:
        return

    X_today = features_df.values.astype(float)
    X_today_scaled = scaler.transform(X_today)

    preds = model.predict(X_today_scaled).flatten()
    preds = apply_calibration(preds)

    # Attach predictions and DK lines for inspection
    output = games_df.copy()
    output["predicted_total"] = preds

    if "dk_total_point" in output.columns:
        output["edge"] = output["predicted_total"] - output["dk_total_point"]
        output["O/U"] = np.where(output["edge"] > 0, "O", "U")

    print("\nPredicted Totals for Today's Matchups:")
    print(output)

    try:
        data_to_googlesheets(output, model="V2", sheet_name="PLONN V2")
    except Exception as e:  # pragma: no cover - logging is best-effort
        print(f"[WARN] Failed to push totals to Google Sheets: {e}")


# ---------------------------------------------------------------------------
# Predict for a specific date
# ---------------------------------------------------------------------------
def predict_date_totals(target_date: str, xlsx_path: str = BACKTEST_XLSX_DEFAULT) -> None:
    """
    Predict totals for a specific date (YYYY-MM-DD) using the ADV/TPG workbook.

    This filters the ADV sheet to the given date, builds the same feature matrix
    used in backtests, and runs the trained total model.
    """
    try:
        feature_cols = load_feature_list()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}. Train the model first with -t.")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Missing model file: {MODEL_PATH}. Train the model first with -t.")
        return

    try:
        scaler = load_scaler()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}. Train the model first with -t.")
        return

    # Load ADV/TPG sheets
    try:
        adv_df, tpg_df = load_adv_tpg_sheets(xlsx_path)
    except Exception as e:
        print(f"[ERROR] Failed to load ADV/TPG workbook '{xlsx_path}': {e}")
        return

    # Resolve and parse date column in ADV
    try:
        date_col = resolve_column(adv_df, ["Date", "date", "GAME_DATE", "game_date"])
    except KeyError as e:
        print(f"[ERROR] ADV sheet has no recognizable date column: {e}")
        return

    adv_df = adv_df.copy()
    adv_df[date_col] = pd.to_datetime(adv_df[date_col], errors="coerce")
    target_dt = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target_dt):
        print(f"[ERROR] Could not parse target_date '{target_date}' as YYYY-MM-DD.")
        return

    # First try exact date match (year-month-day)
    mask = adv_df[date_col].dt.date == target_dt.date()
    adv_day = adv_df.loc[mask]

    # If no rows, fall back to matching by month/day only (ignore year),
    # which is useful if ADV stores dates for a different season year or
    # without a proper year component.
    if adv_day.empty:
        month_series = adv_df[date_col].dt.month
        day_series = adv_df[date_col].dt.day
        fallback_mask = (month_series == target_dt.month) & (day_series == target_dt.day)
        adv_day = adv_df.loc[fallback_mask]
        if adv_day.empty:
            print(f"No ADV rows found for date {target_date} (even ignoring year).")
            return

    # Build feature matrix from this subset and run predictions
    try:
        features_df, adv_kept = build_backtest_feature_matrix_from_tpg(adv_day, tpg_df, feature_cols)
    except Exception as e:
        print(f"[ERROR] Failed to build feature matrix for {target_date}: {e}")
        return

    X = features_df.values.astype(float)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled).flatten()
    preds = apply_calibration(preds)

    output = adv_kept.copy()
    output["predicted_total"] = preds
    if "dk_total_point" in output.columns:
        output["edge"] = output["predicted_total"] - output["dk_total_point"]
        output["O/U"] = np.where(output["edge"] > 0, "O", "U")

    print(f"\nPredicted Totals for {target_date}:")
    print(output)

    try:
        data_to_googlesheets(output, model='V2', sheet_name=f"PLONN V2")
    except Exception as e:
        print(f"[WARN] Failed to push totals to Google Sheets: {e}")

parser = argparse.ArgumentParser(
    prog="PLONN-NBA-Total-v2",
    description=(
        "PLONN total-points model: predicts a single total score per game "
        "using team/opp stats and DraftKings lines."
    ),
)
parser.add_argument("-t", "--train", action="store_true", help="Train the total-points model")
parser.add_argument("-g", "--graph", action="store_true", help="Plot test-set predictions vs actual totals")
parser.add_argument("-p", "--predict-today", action="store_true", help="Run predictions for today's games")
parser.add_argument("--data-path", type=str, default=DATA_PATH, help="Path to historical data CSV file")
parser.add_argument("--backtest", type=str, help="Run backtest on an Excel file with ADV and TPG sheets")
# NEW: Add predict-date flag
parser.add_argument("--predict-date", type=str, help="Run predictions for a specific date (YYYY-MM-DD)")
# NEW: Add calibration graph flag
parser.add_argument("--cg", action="store_true", help="Show calibration scatter graph from ADV/TPG workbook")

if __name__ == "__main__":
    args = parser.parse_args()

    # If user requests only the calibration graph, do that and exit.
    if args.cg:
        xlsx_path = args.backtest if args.backtest else BACKTEST_XLSX_DEFAULT
        calibration_graph_from_excel(xlsx_path)
        sys.exit(0)

    if args.train:
        X_train, X_test, y_train, y_test, feature_cols = prepare_training_data(
            data_path=args.data_path,
        )
        train_and_evaluate(X_train, X_test, y_train, y_test, feature_cols, graph=args.graph)

    # If the user passes --backtest without --predict-date, run a full backtest on that workbook.
    if args.backtest and not args.predict_date:
        backtest_from_excel(args.backtest)

    if args.predict_today:
        predict_today_totals()

    # For predict-date, allow the user to override the ADV/TPG workbook via --backtest.
    if args.predict_date:
        xlsx_path = args.backtest if args.backtest else BACKTEST_XLSX_DEFAULT
        predict_date_totals(args.predict_date, xlsx_path=xlsx_path)

    if not args.train and not args.predict_today and not args.backtest and not args.predict_date:
        # Default behavior: just predict today if there is a trained model.
        predict_today_totals()

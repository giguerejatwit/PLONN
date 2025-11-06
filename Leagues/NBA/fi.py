# fi.py
import sys
import types

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

# --- Patch external deps so nba_total_pts imports cleanly ---
sys.modules['ghsheets_logger'] = types.SimpleNamespace(
    data_to_googlesheets=lambda *a, **kw: None
)
sys.modules['get_data'] = types.SimpleNamespace(
    get_team_per_game_stats=lambda *a, **kw: {},
    get_today_games=lambda *a, **kw: pd.DataFrame()
)

from leagues.NBA import nba_total_pts  # uses clean_adv, calculate_team_pace, feature_columns


def main():
    # ---- 1) Load DK-enriched training data ----
    data_path = "leagues/NBA/data/offsets/offset_with_dk_final.csv"
    print(f"[INFO] Loading data from {data_path}")
    data = pd.read_csv(data_path)

    # Drop obvious garbage rows (you can relax this later if you want)
    data = data.dropna(axis=0)

    # Recompute Pace / Pace.1 the same way as in training
    data = nba_total_pts.calculate_team_pace(data)

    # Build rolling 30-game features + DK line using your existing logic
    X_train, X_test, y_train, y_test = nba_total_pts.clean_adv(data)

    print(f"[INFO] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"[INFO] y_train: {y_train.shape}, y_test: {y_test.shape}")

    # ---- 2) Load the adv model ----
    model_path = "leagues/NBA/models/adv_model.keras"
    print(f"[INFO] Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    expected_features = model.input_shape[-1]
    current_features = X_test.shape[1]
    print(f"[INFO] Model expects {expected_features} features; X_test has {current_features}")
    if expected_features != current_features:
        raise ValueError(
            f"Feature dimension mismatch: model expects {expected_features}, "
            f"but X_test has {current_features}. Retrain adv on offset_with_dk_final.csv first."
        )

    # ---- 3) Baseline performance ----
    y_pred_base = model.predict(X_test, verbose=0).flatten()
    baseline_mae = mean_absolute_error(y_test, y_pred_base)
    print(f"\n[BASELINE] MAE = {baseline_mae:.3f}\n")

    # ---- 4) Permutation importance ----
    rng = np.random.default_rng(42)
    results = []

    for col in X_test.columns:
        X_perm = X_test.copy()
        X_perm[col] = rng.permutation(X_perm[col].values)
        y_perm = model.predict(X_perm, verbose=0).flatten()
        mae = mean_absolute_error(y_test, y_perm)
        results.append((col, mae - baseline_mae))

    # Sort by impact
    results.sort(key=lambda x: x[1], reverse=True)

    print("All features by MAE increase (higher = more important):\n")
    for name, inc in results:
        print(f"{name:15s}  +{inc:.4f}")

    # Export all features and their MAE deltas to CSV
    df_importance = pd.DataFrame(results, columns=["feature", "mae_increase"])
    df_importance.to_csv("feature_importance_full.csv", index=False)

    # ---- 5) Plot bar chart ----
    top_n = 15
    top = results[:top_n]
    feat_names, incs = zip(*top)
    plt.figure(figsize=(10, 6))
    plt.barh(feat_names[::-1], incs[::-1])
    plt.xlabel("MAE Increase when Shuffled")
    plt.title("Feature Importance (Permutation, adv model)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
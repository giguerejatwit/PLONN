import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# === Load Data ===
df = pd.read_csv("Leagues/MLB/data/gamelog_final.csv")

# Drop NaNs in key features
feature_cols = ['K_prev', 'K_avg_3', 'IP_avg_3', 'BF', 'WHIP_avg_3', 'H_per9', 'BB_per9'] + ['enc_4', 'enc_5', 'enc_6','enc_0'] #[f'enc_{i}' for i in range(8)]
df = df.dropna(subset=feature_cols + ['SO'])

# === Features & Target ===
X = df[feature_cols]
y = df['SO']

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model ===
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    objective='reg:squarederror',
    seed=42
)

model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nTest MAE: {mae:.2f} strikeouts")

# === Feature Importance (optional) ===
import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import joblib

import keras

# === Load & Filter Data ===
data = pd.read_csv("Leagues/MLB/data/gamelog_final.csv")
feature_cols =  ['K_prev', 'K_avg_3', 'IP_avg_3', 'BF', 'WHIP_avg_3', 'H_per9', 'BB_per9'] + ['enc_4', 'enc_5', 'enc_6','enc_0']
data = data.dropna(subset=feature_cols + ['SO'])

X = data[feature_cols].values
y = data['SO'].values.reshape(-1, 1)

# === Scale Inputs and Target ===
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# Save scalers for inference
joblib.dump(x_scaler, "Leagues/MLB/models/mlp_x_scaler.pkl")
joblib.dump(y_scaler, "Leagues/MLB/models/mlp_y_scaler.pkl")

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# === Build MLP Model ===
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # regression output
])

model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=['mae'])

# === Train ===
early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=75, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# === Evaluate ===
loss, mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE (scaled): {mae:.4f}")

# === Save Model ===
model.save("Leagues/MLB/models/mlp_strikeout_model_scaled.keras")
print("Saved model and scalers.")
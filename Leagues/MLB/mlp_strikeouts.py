import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load final dataset
df = pd.read_csv("Leagues/MLB/data/gamelog_final.csv")

# Drop rows with NaNs in key features
df = df.dropna(subset=['K_prev', 'K_avg_3', 'IP_avg_3'] + [f'enc_{i}' for i in range(8)])

# Features and target
features = ['K_prev', 'K_avg_3', 'IP_avg_3'] + [f'enc_{i}' for i in range(8)]
X = df[features]
y = df['SO']  # strikeouts

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# MLP Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    
    Dense(8, activation='relu'),

    Dense(1)  # regression output
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE: {mae:.2f} strikeouts")

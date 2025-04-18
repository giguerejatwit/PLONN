import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==== Load Data ====
df = pd.read_csv("Leagues/MLB/data/2024/gamelog_final.csv")

# ==== Clean & Filter ====
df = df.dropna(subset=['K_prev', 'K_avg_3', 'IP_avg_3'] + [f'enc_{i}' for i in range(8)])

# Convert Date to datetime for sorting
df['Date'] = pd.to_datetime(df['Date'])

# ==== Feature Columns ====
feature_cols = ['K_prev', 'K_avg_3', 'BB', 'BF', 'IP_avg_3'] + [f'enc_{i}' for i in range(8)]

# ==== LSTM Dataset Builder ====
def build_lstm_dataset(df, feature_cols, sequence_length=3):
    sequences = []
    targets = []
    df = df.sort_values(by=['Player', 'Date'])

    for player in df['Player'].unique():
        player_df = df[df['Player'] == player]

        if len(player_df) < sequence_length + 1:
            continue

        for i in range(len(player_df) - sequence_length):
            seq = player_df.iloc[i:i+sequence_length][feature_cols].values
            target = player_df.iloc[i+sequence_length]['SO']
            sequences.append(seq)
            targets.append(target)

    return np.array(sequences), np.array(targets)


# ==== Build Sequences ====
X, y = build_lstm_dataset(df, feature_cols)

# ==== Train/Test Split ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== LSTM Model ====
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # regression output
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# ==== Train ====
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# ==== Evaluate ====
loss, mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE: {mae:.2f} strikeouts")

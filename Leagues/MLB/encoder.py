import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras




"""
TODO:
    1. Normalize Key Stats
        - K-rate = SO/PA
        - BB-rate = BB/PA
        - HR-rate = HR/AB
        - 1B-rate = 1B / AB (1B = H - 2B - 3B - HR)
    2. Composite metrics
        - IsoPower = SLG - BA
        - OBP+SLG = OPS
        - ContactRate = 1 - (SO / PA)
        - XBH = 2B + 3B + HR
    3. Drop redundancies
        - Check HR & HR.1
        - OPS and OPS.1
    4. Per Game Averages
        -SO_per_game
        -TB_per_game
        -R_per_game
"""
class Autoencoder(keras.Model):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.Dense(encoding_dim, activation='relu', input_shape=(input_dim,))
        ])
        self.decoder = keras.Sequential([
            keras.layers.Dense(input_dim, activation='linear')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def preprocess(data:pd.DataFrame) -> pd.DataFrame:
    meta_cols = ['Team', 'Span Started', 'Span Ended']
    meta = data[meta_cols]
    
    # Calculate key stats
    data['K-rate'] = data['SO'] / data['PA']
    data['BB-rate'] = data['BB'] / data['PA']
    data['HR-rate'] = data['HR'] / data['AB']
    data['1B-rate'] = (data['H'] - data['2B'] - data['3B'] - data['HR']) / data['AB']
    
    # Calculate composite metrics
    data['IsoPower'] = (data['H'] + (data['2B'] * 2) + (data['3B'] * 3) + (data['HR'] * 4)) / data['AB']
    data['OBP+SLG'] = ((data['H'] + data['BB']) / (data['AB'] + data['BB'])) + ((data['H'] + (data['2B'] * 2) + (data['3B'] * 3) + (data['HR'] * 4)) / (data['AB']))
    data['ContactRate'] = 1 - (data['SO'] / data['PA'])
    data['XBH'] = data['2B'] + data['3B'] + data['HR']
    
    data['SO_per_game'] = data['SO'] / data['G']
    data['TB_per_game'] = data['TB'] / data['G']
    data['R_per_game'] = data['R'] / data['G']
    
    features = data.drop(columns=meta_cols)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, meta.reset_index(drop=True)

if __name__ == "__main__":
    # Load the data
    data = pd.read_excel("data/5gw.xlsx", header=0)
    # Preprocess the data
    features_scaled, meta_data = preprocess(data)

    input_shape = features_scaled.shape[1]
    encoding_dim = 8


    # normalized_df = pd.concat([meta_data, preprocessed_data], axis=1)

    autoencoder = Autoencoder(input_dim=input_shape, encoding_dim=encoding_dim)
    autoencoder.compile(optimizer='adam', loss=tf.losses.MeanSquaredError())

    autoencoder.fit(features_scaled, features_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

    # --- Encode the data ---
    encoded_ouput = autoencoder.encoder.predict(features_scaled)
    encoded_df = pd.DataFrame(encoded_ouput, columns=[f'encoded_{i}' for i in range(encoded_ouput.shape[1])])

    # --- Combine with meta data ---
    final_encoded_df = pd.concat([meta_data, encoded_df], axis=1)

    # Save the data
    final_encoded_df.to_csv("data/encoded_5game_spans.csv", index=False)






    
    
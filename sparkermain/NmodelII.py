import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
import random
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

from binance.client import Client
from .config import API_KEY, API_SECRET

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


class NModelII:
    def __init__(self, symbol, interval='1m', lookback='2 day ago UTC'):
        self.client = Client(API_KEY, API_SECRET)
        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback
        self.n_past = 30
        self.n_future = 5
        self.model = None
        self.scaler_close = None

    def fetch_historical_data(self, symbol, interval, lookback):
        klines = self.client.get_historical_klines(symbol, interval, lookback)
        data = []
        for kline in klines:
            data.append([float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4])])
        return pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'])

    def create_sequences(self, close_prices):
        X, y = [], []
        for i in range(self.n_past, len(close_prices) - self.n_future + 1):
            X.append(close_prices[i - self.n_past:i])
            y.append(close_prices[i:i + self.n_future])
        return np.array(X), np.array(y)

    def train_model(self):
        df = self.fetch_historical_data(self.symbol, self.interval, self.lookback)
        df = df.astype(float).dropna()

        close_values = df[['Close']].values
        self.scaler_close = StandardScaler()
        scaled_close = self.scaler_close.fit_transform(close_values)

        X, y = self.create_sequences(scaled_close)

        print(f"Training data shape: X={X.shape}, y={y.shape}")

        model = Sequential([
            Input(shape=(X.shape[1], 1)),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(self.n_future)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=1)

        self.model = model
        self.save_model()

    def predict_next(self, df=None):
        if df is None or len(df) < self.n_past:
            print("Fetching fresh historical data for prediction...")
            df = self.fetch_historical_data(self.symbol, self.interval, self.lookback)

        df = df.astype(float).dropna()
        close_series = df[['Close']].values
        scaled_close = self.scaler_close.transform(close_series)

        recent_data = scaled_close[-self.n_past:]
        input_seq = np.expand_dims(recent_data, axis=0)  # Shape: (1, n_past, 1)

        scaled_preds = self.model.predict(input_seq, verbose=0)[0]
        inv_preds = self.scaler_close.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()

        return inv_preds.tolist()

    def save_model(self, model_path='model.keras', scaler_path='scaler_close.pkl'):
        self.model.save(model_path)
        joblib.dump(self.scaler_close, scaler_path)
        print("Model and scaler saved.")

    def load_model(self, model_path='model.keras', scaler_path='scaler_close.pkl'):
        self.model = load_model(model_path)
        self.scaler_close = joblib.load(scaler_path)

    def load_or_train_model(self, model_path='model.keras', scaler_path='scaler_close.pkl'):
        
        try:
            self.load_model(model_path=model_path, scaler_path=scaler_path)
            print("Loaded existing model and scaler.")
        except Exception as e:
            print(f"Error loading model/scaler: {e}")
            print("Training new model...")
            self.train_model()
    

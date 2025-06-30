# !/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import load_model
import joblib

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from binance.client import Client

from .config import API_KEY, API_SECRET

'''
Univariate Recursive LSTM Model for Time Series Forecasting
Forecasts future Close prices using recursive 1-step predictions.
Author: Dennis K. M.
Date: 2025-05-24 (updated)
'''


class NModelII:
    def __init__(self, symbol, interval='1m', lookback='7 day ago UTC'):
        self.client = Client(API_KEY, API_SECRET)
        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback
        self.n_past = 30
        self.n_future = 5  # we now forecast 5 steps recursively

    def fetch_historical_data(self, symbol, interval, lookback):
        klines = self.client.get_historical_klines(symbol, interval, lookback)
        data = []
        for kline in klines:
            data.append([float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4])])
        return pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'])

    def create_sequences(self, data, n_past, target_column):
        X, y = [], []
        for i in range(n_past, len(data)):
            X.append(data[i - n_past:i])
            y.append(data[i, target_column])  # single-step univariate target
        return np.array(X), np.array(y)

    def train_model(self):
        df = pd.DataFrame(self.fetch_historical_data(self.symbol, self.interval, self.lookback))
        df = df.astype(float).dropna()

        # Scaling
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close']])
        df[['Open', 'High', 'Low', 'Close']] = scaled_features

        feature_columns = ['Open', 'High', 'Low', 'Close']
        data = df[feature_columns].values

        # Create sequences
        X, y = self.create_sequences(data, self.n_past, target_column=3)  # 'Close' is the last column

        print(f"Input shape X: {X.shape}, y: {y.shape}")

        # Model
        model = Sequential([
            Input(shape=(X.shape[1], X.shape[2])),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)  # single-step univariate
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32)

        self.model = model
        self.scaler = scaler
        self.save_model()

    def predict_next(self, df=None, steps=10):
        if df is None or len(df) < self.n_past:
            print(f'len(df): {len(df)} - Fetching historical data')
            df = pd.DataFrame(self.fetch_historical_data(self.symbol, self.interval, self.lookback))
            df = df.astype(float).dropna()
        else:
            print(f'len(df): {len(df)} - Using provided data')
            df = df.astype(float).dropna()

        feature_columns = ['Open', 'High', 'Low', 'Close']
        df[feature_columns] = self.scaler.transform(df[feature_columns])

        recent_data = df[feature_columns].values[-self.n_past:]
        if recent_data.shape[0] < self.n_past:
            raise ValueError("Not enough data for prediction window.")

        predictions = []

        for _ in range(steps):
            input_sequence = np.expand_dims(recent_data, axis=0)  # (1, n_past, num_features)
            scaled_pred = self.model.predict(input_sequence, verbose=0)[0][0]

            # Inverse transform using dummy row
            dummy_row = np.zeros((1, len(feature_columns)))
            dummy_row[0, -1] = scaled_pred
            inverse_prediction = self.scaler.inverse_transform(dummy_row)[0, -1]
            predictions.append(inverse_prediction)

            # Append predicted 'Close' back to recent_data
            new_row = recent_data[-1].copy()
            new_row[-1] = scaled_pred
            recent_data = np.vstack([recent_data[1:], new_row])

        return predictions

    def save_model(self, model_path='model.h5', scaler_path='scaler.pkl'):
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

    def load_model(self, model_path='model.h5', scaler_path='scaler.pkl'):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        #print(f"Model loaded from {model_path}")
        #print(f"Scaler loaded from {scaler_path}")

    def load_or_train_model(self, model_path='model.h5', scaler_path='scaler.pkl'):
        try:
            self.load_model(model_path=model_path, scaler_path=scaler_path)
            print("Successfully loaded existing model and scaler.")
        except (FileNotFoundError, IOError, Exception) as e:
            print(f"Failed to load model or scaler due to: {e}")
            print("Training a new model instead...")
            self.train_model()

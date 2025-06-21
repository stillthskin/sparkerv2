# # !/usr/bin/env python
# # coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Input
import joblib
from tensorflow.keras.models import load_model
import os
from .config import API_KEY, API_SECRET
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from binance.client import Client
'''
Multivariate LSTM Model for Time Series Forecasting
This code implements a multivariate LSTM model for time series forecasting using historical data from Binance.
This model is designed to predict future prices based on past price data.
Author: Dennis K. M.
Date: 2025-05-24
'''


class NModelII:
    def __init__(self,symbol,interval='1m',lookback='7 day ago UTC'):
        self.client = Client(API_KEY, API_SECRET)
        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback
        self.n_past = 30  
        self.n_future = 10
        
    
    def fetch_historical_data(self, symbol, interval, lookback):
        klines = self.client.get_historical_klines(symbol, interval, lookback)
        data = []
        for kline in klines:
            data.append([float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4])])
        return pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'])

    def create_sequences(self, data, n_past, n_future,target_column):
        X, y = [], []
        self.n_past = n_past
        for i in range(n_past, len(data) - n_future + 1):
            X.append(data[i - n_past:i])
            y.append(data[i:i + n_future, target_column]) 
        return np.array(X), np.array(y)


    def train_model(self):
        df = pd.DataFrame(self.fetch_historical_data(self.symbol, self.interval, self.lookback))
        df = df.astype(float).dropna()

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close']].astype(float))
        df[['Open', 'High', 'Low', 'Close']] = scaled_features

        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        # Features to include (multivariate)
        feature_columns = ['Open', 'High', 'Low', 'Close']
        data = df[feature_columns].values
        past= 30 
        future = 10
        # Create sequences
        X, y = self.create_sequences(data,past,future,target_column=len(feature_columns)-1)  # Target is last column
        print(f"Input shape X: {X.shape}, y: {y.shape}")

        # Build the LSTM model
       # Build the LSTM model
        model = Sequential([
            Input(shape=(X.shape[1], X.shape[2])),  # input_shape = (timesteps, features)

            LSTM(units=50, return_sequences=True),
            Dropout(0.2),

            LSTM(units=50, return_sequences=True),
            Dropout(0.2),

            LSTM(units=50, return_sequences=True),
            Dropout(0.2),

            LSTM(units=50), 
            Dropout(0.2),

            Dense(units=future)  #prdict N-future steps
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32)

        self.model = model
        self.scaler = scaler
        self.save_model()


  


    def predict_next(self,df):
        # Fetch latest data to create the last input sequence
        if df is None or  len(df) < self.n_past:
            print(f'len(df): {len(df)} - Fetching historical data')
            df = pd.DataFrame(self.fetch_historical_data(self.symbol, self.interval, self.lookback))
            df = df.astype(float).dropna()
        else:
            print(f'len(df): {len(df)} - Using provided data')
            df = df.astype(float).dropna()
        feature_columns = ['Open', 'High', 'Low', 'Close']
        #print(f'Recent data: {df[feature_columns].tail(5)}')
        # Apply the same scaling
        df[feature_columns] = self.scaler.transform(df[feature_columns])
        

        # Extract the most recent window
        recent_data = df[feature_columns].values[-self.n_past:]  # Shape: (window_size, num_features)
        if recent_data.shape[0] < self.n_past:
            raise ValueError("Not enough data for prediction window.")

        # Reshape for model: (1, timesteps, features)
        input_sequence = np.expand_dims(recent_data, axis=0)

        # Predict the next 'Close' value (scaled)
        scaled_prediction = self.model.predict(input_sequence)
        print(f'Scaled prediction shape: {scaled_prediction.shape}')
        scaled_prediction = scaled_prediction.reshape(-1, 1)  # (10, 1)
        num_features = len(feature_columns)
        # Dummy array for inverse transform
        dummy_rows = np.zeros((10, num_features))  # num_features = 4
        dummy_rows[:, -1] = scaled_prediction[:, 0]

        # Inverse transform
        inverse_predictions = self.scaler.inverse_transform(dummy_rows)
        print(f'Inverse prediction shape: {inverse_predictions.shape}')
        #print(f'Inverse prediction: {inverse_predictions}')
        inverse_close_predictions = inverse_predictions[:, -1]  # (10,)

        return inverse_close_predictions
    def save_model(self, model_path='model.h5', scaler_path='scaler.pkl'):
        # Save the Keras model
        self.model.save(model_path)
        # Save the scaler using joblib
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

    def load_model(self, model_path='model.h5', scaler_path='scaler.pkl'):
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
     
    def load_or_train_model(self, model_path='model.h5', scaler_path='scaler.pkl'):
        try:
            self.load_model(model_path=model_path, scaler_path=scaler_path)
            print("Successfully loaded existing model and scaler.")
        except (FileNotFoundError, IOError, Exception) as e:
            print(f"Failed to load model or scaler due to: {e}")
            print("Training a new model instead...")
            self.train_model()


      



# model=NModelII('BTCUSDT')


# model.load_or_train_model(model_path='model.h5', scaler_path='scaler.pkl')
# print(f' SOme prdiction: {model.predict_next()}')
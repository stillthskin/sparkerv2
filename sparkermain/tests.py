#!/usr/bin/env python3

import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense,Input
from joblib import dump, load
import os


class CryptoPredictor:
    def __init__(self, api_key, api_secret):
        self.model = None
        self.scaler = MinMaxScaler()
        self.mae = None
        self.api_key = api_key
        self.api_secret = api_secret
        self.model_file = 'best_model.keras'
        self.scaler_file = 'scaler.joblib'
        self.mae_file = 'mae.npy'

    def train_model(self, symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1HOUR,
                    start_date='2025-01-01', iterations=10):
        """
        Main training method that either loads existing models or trains new ones
        """
        try:
            self._load_assets()
            print("Successfully loaded existing model")
        except Exception as e:
            print(f"Loading failed: {e}\nTraining new model...")
            self._full_training_pipeline(symbol, interval, start_date, iterations)
    def predict(self, input_data):
        """
        Make predictions using the trained model
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_model() first")

        # Ensure we have enough data points
        if len(input_data) < 60:
            raise ValueError(f"Need at least 60 data points, got {len(input_data)}")

        # Preprocess input
        scaled_data = self.scaler.transform(input_data.reshape(-1, 1))
        
        # Get last 60 points (works for both >60 and exactly 60 points)
        sequence = scaled_data[-60:] if len(scaled_data) > 60 else scaled_data
        
        # Reshape for model input
        sequence = sequence.reshape(1, 60, 1)
        
        # Make prediction
        scaled_pred = self.model.predict(sequence)
        return self.scaler.inverse_transform(scaled_pred).flatten()


    def _full_training_pipeline(self, symbol, interval, start_date, iterations):
        """
        Complete training pipeline from data fetching to model saving
        """
        # Data acquisition
        data = self._fetch_binance_data(symbol, interval, start_date)
        prices = data['close'].values.reshape(-1, 1)

        # Data preprocessing
        train_data, test_data = self._prepare_data(prices)
        X_train, y_train = self._create_sequences(train_data)
        X_test, y_test = self._create_sequences(test_data)

        # Model training
        best_model, best_mae = self._train_multiple_models(X_train, y_train, X_test, y_test, iterations)

        # Save assets
        self.model = best_model
        self.mae = best_mae
        self._save_assets()

    def _fetch_binance_data(self, symbol, interval, start_date):
        """Fetch historical data from Binance API"""
        client = Client(self.api_key, self.api_secret)
        klines = client.get_historical_klines(symbol, interval, start_date)
        print(f"Fetched {len(klines)} data points from Binance")
        
        return pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]).astype({'close': float})

    def _prepare_data(self, prices, split_ratio=0.8):
        """Prepare and split scaled data"""
        split_idx = int(len(prices) * split_ratio)
        self.scaler.fit(prices[:split_idx])
        return (
            self.scaler.transform(prices[:split_idx]),
            self.scaler.transform(prices[split_idx:])
        )

    @staticmethod
    def _create_sequences(data, lookback=60, future_steps=10):
        """Create input sequences and target outputs"""
        X, y = [], []
        for i in range(len(data) - lookback - future_steps + 1):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback:i+lookback+future_steps])
        return np.array(X), np.array(y)

    def _build_model(self):
        """Construct LSTM model architecture"""
        model =  Sequential([
    Input(shape=(60, 1)),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(10)
])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _train_multiple_models(self, X_train, y_train, X_test, y_test, iterations):
        """Train multiple models and keep the best performing one"""
        best_mae = float('inf')
        best_model = None

        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")
            model = self._build_model()
            model.fit(X_train, y_train, epochs=30, 
                     batch_size=64, validation_split=0.2, verbose=0)
            
            current_mae = model.evaluate(X_test, y_test, verbose=0)[1]
            print(f"Validation MAE: {current_mae:.4f}")
            
            if current_mae < best_mae:
                best_mae = current_mae
                best_model = model
                print("New best model found!")

        return best_model, best_mae

    def _save_assets(self):
        """Save model and preprocessing assets"""
        save_model(self.model, self.model_file)
        dump(self.scaler, self.scaler_file)
        np.save(self.mae_file, self.mae)

    def _load_assets(self):
        """Load existing model and preprocessing assets"""
        self.model = load_model(self.model_file)
        self.scaler = load(self.scaler_file)
        self.mae = np.load(self.mae_file)


if __name__ == "__main__":
    # Example usage
    # Note: Replace with actual API keys for real usage
    predictor = CryptoPredictor(
        api_key='null',
        api_secret='null'
    )
    
    # Train or load model
    predictor.train_model()
    
    # Minimum viable test case (60 points)
    minimal_test_data = np.linspace(40000, 41000, 60)
    print(f"Sample : {minimal_test_data}")

    prediction = predictor.predict(minimal_test_data)
    print(f"Predicted next 10 prices: {prediction}")
    print(f"Model MAE: {predictor.mae}")
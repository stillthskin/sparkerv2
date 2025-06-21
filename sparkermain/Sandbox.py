import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

class CryptoPrediction:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.url = f'https://api.binance.com/api/v1/klines'
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lstm_model = None
        self.rf_model = None

    def get_binance_data(self, interval="1d", lookback="30"):
        # Get the historical data for the symbol
        params = {
            'symbol': self.symbol,
            'interval': interval,
            'limit': lookback
        }
        response = requests.get(self.url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = pd.to_numeric(df['close'])
            df = df[['timestamp', 'close']]
            return df
        else:
            raise Exception(f"Failed to fetch data: {response.status_code}")

    def preprocess_data(self):
        '''
        Gets data scales and 80-20 slits them
        Returns train test for both Random forest and 
        Lstm...
        '''
        # Fetch the data
        self.data = self.get_binance_data()

        # Scale the 'close' prices
        scaled_data = self.scaler.fit_transform(self.data['close'].values.reshape(-1, 1))

        # Prepare data for LSTM and Random Forest models
        X_lstm, y_lstm = [], []
        X_rf, y_rf = [], []
        for i in range(30, len(scaled_data)):
            X_lstm.append(scaled_data[i-30:i, 0])  # Last 30 days' prices
            y_lstm.append(scaled_data[i, 0])  # Predict next day's price

            X_rf.append(scaled_data[i-30:i, 0])  # Last 30 days' prices for Random Forest  *****************************
            y_rf.append(scaled_data[i, 0])  # Predict next day's price  *****************************

        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
        X_rf, y_rf = np.array(X_rf), np.array(y_rf)

        # Reshape X_lstm for LSTM (3D input)
        X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

        # Train/test split for Random Forest
        split_idx = int(0.8 * len(X_rf))   #80-20 split
        self.X_train_rf, self.X_test_rf = X_rf[:split_idx], X_rf[split_idx:]
        self.y_train_rf, self.y_test_rf = y_rf[:split_idx], y_rf[split_idx:]

        return X_lstm, y_lstm, X_rf, y_rf

    def train_lstm(self, X_lstm, y_lstm):
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=False, input_shape=(X_lstm.shape[1], 1)))
        model.add(Dense(units=1))  # Output a single prediction
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train LSTM model
        model.fit(X_lstm, y_lstm, epochs=10, batch_size=16)
        self.lstm_model = model

    def train_rf(self, X_rf, y_rf):
        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_rf, y_rf)
        self.rf_model = rf_model

    def predict_lstm(self, X_input):
        # Predict using LSTM
        scaled_input = self.scaler.transform(X_input.reshape(-1, 1))
        scaled_input = np.reshape(scaled_input, (1, 30, 1))  # Reshaping for LSTM
        prediction = self.lstm_model.predict(scaled_input)
        return self.scaler.inverse_transform(prediction)[0][0]

    def predict_rf(self, X_input):
        # Predict using Random Forest
        scaled_input = self.scaler.transform(X_input.reshape(-1, 1))
        return self.rf_model.predict(scaled_input.reshape(1, -1))[0]

    def predict_next_day(self):
        # Prepare data
        X_lstm, y_lstm, X_rf, y_rf = self.preprocess_data()

        # Train both models
        self.train_lstm(X_lstm, y_lstm)
        self.train_rf(X_rf, y_rf)

        # Make predictions for the next day
        last_30_days_data = self.data['close'].values[-30:]
        
        lstm_prediction = self.predict_lstm(last_30_days_data)
        rf_prediction = self.predict_rf(last_30_days_data)

        # Print predictions
        print(f"LSTM Prediction for next day: {lstm_prediction}")
        print(f"Random Forest Prediction for next day: {rf_prediction}")

        return lstm_prediction, rf_prediction

# Example usage
crypto = CryptoPrediction(symbol="BTCUSDT")
lstm_prediction, rf_prediction = crypto.predict_next_day()

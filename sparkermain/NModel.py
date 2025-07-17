import numpy as np
import pandas as pd
import talib as ta
import joblib
from .config import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from binance.client import Client
from binance.enums import *

class AI_Model:
    def __init__(self, symbol, period='15m', lookback=30, steps_to_predict=10):
        """
        Initializes the model with the given symbol, period (time frame), lookback (number of periods),
        and steps_to_predict (number of future steps to predict).
        :param symbol: Trading pair symbol (e.g., 'BTCUSDT').
        :param period: Timeframe for historical data (e.g., '1m', '1h', '1d').
        :param lookback: Number of periods to consider for prediction (e.g., 30).
        :param steps_to_predict: Number of future closing prices to predict (e.g., 10).
        """
        self.model = None
        self.symbol = symbol
        self.interval = period
        self.lookback = lookback
        self.steps_to_predict = steps_to_predict
        self.client = Client(API_KEY, API_SECRET)

    def train_model(self):
        """
        Train the model using historical data with features like Open, High, Low, volume,
        and predict the closing prices for the next `steps_to_predict` periods.
        """
        # Fetch historical data from Binance
        print(self.symbol, self.lookback,self.steps_to_predict)
        #klines = self.client.get_historical_klines(self.symbol, self.period, f"{self.lookback + self.steps_to_predict} ago UTC")
        klines= self.client.get_historical_klines(self.symbol, self.interval, f"{self.lookback} day ago UTC")
        print(f'Length of klines {len(klines)}')

        # Create DataFrame
        df = pd.DataFrame(klines, columns=[  "timestamp", "Open", "High", "Low", "Close", "volume",
                "Close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Keep relevant columns for prediction
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'volume', 'number_of_trades']]

        # Convert the necessary columns to numeric
        df[['Open', 'High', 'Low', 'Close', 'volume', 'number_of_trades']] = df[['Open', 'High', 'Low', 'Close', 'volume', 'number_of_trades']].apply(pd.to_numeric)

        # Add technical indicators (RSI, Bollinger Bands, SMA)
        df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
        df['SMA'] = ta.SMA(df['Close'], timeperiod=14)
        df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        feature_columns = df[['Open', 'High', 'Low', 'Close', 'volume', 'number_of_trades']]
        print(f'length of feature columns: {len(feature_columns)}')

        X=[]
        y=[]
        for i in range(self.lookback, len(df) - self.steps_to_predict):
                # Features: past 'lookback' timesteps of OHLC data
                features = feature_columns.iloc[i-self.lookback:i].values.flatten()
                # Labels: next 'future_steps' Close prices
                labels = df['Close'].iloc[i:i+self.steps_to_predict].values
                X.append(features)
                y.append(labels)
        # Drop missing values (the last few rows will have no target values)
        df.dropna(inplace=True)

   
        # Split data into train and test sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model (using RandomForestRegressor for continuous values)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        joblib.dump(self.model,'sparker.joblib')

        # Predict on the test set
        y_pred = self.model.predict(X_test)

        # Evaluate the model using Mean Absolute Error, Mean Squared Error, and R^2 score
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R^2 Score: {r2}")

        return 'Train and evaluation successful'

    def prediction(self, current_data):

        if self.model is None:
            self.model=self.load_model()
             


        #print(f'model from within:  {self.model}')
   
        # Convert current_data to DataFrame if it's a dictionary
        if isinstance(current_data, dict):
            current_df = pd.DataFrame([current_data])
        else:
            current_df = current_data  # Assume it's already a DataFrame

        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'volume', 'number_of_trades']
        current_df = current_df[required_columns]

        #print(f'AT pred currentDF: {current_df}')

        # Define feature columns as a list (not a DataFrame)
        feature_columns = required_columns  # Use the list directly

        # Prepare features (no need to reindex)
        current_features = current_df[feature_columns].values.reshape(1, -1)
        #print(f'Lenght of fearures: {current_features}')

        # Predict using the model
        predicted_Closes = self.model.predict(current_features)

        return predicted_Closes

    def load_model(self):
        """
        Load the pre-trained model from a file.
        """
        thismodel = joblib.load('sparker.joblib')
        return thismodel

    
import pandas as pd
from .config import *
from binance.client import Client
from binance.enums import *

class SocketProce:
    def __init__(self, symbol, interval):
        self.max_length = 30
        self.client = Client(API_KEY, API_SECRET)
        self.symbol = symbol
        self.interval = interval

        # Load historical klines (only once)
        klines = self.client.get_historical_klines(
            self.symbol,
            self.interval,
            f"{self.max_length + 10} {self.interval} ago UTC"
        )

        # Create historical DataFrame
        self.df_hist = pd.DataFrame(
            klines,
            columns=["timestamp", "Open", "High", "Low", "Close", "volume",
                     "Close_time", "quote_asset_volume", "number_of_trades",
                     "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
        )

        # Convert numeric columns
        numeric_cols = ["Open", "High", "Low", "Close", "volume", "quote_asset_volume",
                        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
        self.df_hist[numeric_cols] = self.df_hist[numeric_cols].astype(float)

        # Keep last `max_length` rows
        self.df_hist = self.df_hist.tail(self.max_length).reset_index(drop=True)

        # Initialize empty stream DataFrame
        self.df_stream = pd.DataFrame(columns=self.df_hist.columns)

    def processDf(self, kline):
        # Parse incoming kline into dictionary
        processed = {
            'timestamp': kline['t'],
            'Open': float(kline['o']),
            'High': float(kline['h']),
            'Low': float(kline['l']),
            'Close': float(kline['c']),
            'volume': float(kline['v']),
            'Close_time': kline['T'],
            'quote_asset_volume': float(kline['q']),
            'number_of_trades': kline['n'],
            'taker_buy_base_asset_volume': float(kline['V']),
            'taker_buy_quote_asset_volume': float(kline['Q']),
            'ignore': kline['B']
        }

        # Wrap into single-row DataFrame
        new_row = pd.DataFrame([processed])
        new_row = new_row.reindex(columns=self.df_stream.columns)  # Avoid FutureWarning

        # Append to stream
        self.df_stream = pd.concat([self.df_stream, new_row], ignore_index=True)

        # Keep only last `max_length` rows
        if len(self.df_stream) > self.max_length:
            self.df_stream = self.df_stream.iloc[1:]

        # Debug: log length
        print(f"DataFrame length: {len(self.df_stream)}")

        # Return historical until stream is full
        if len(self.df_stream) < self.max_length:
            return self.df_hist
        else:
            return self.df_stream

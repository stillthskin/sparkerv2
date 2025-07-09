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

        # Load initial historical data
        self.df_hist = self.fetch_historical_data()
        print(f"Initial historical DataFrame length: {len(self.df_hist)}")

        # Initialize empty stream DataFrame
        self.df_stream = pd.DataFrame(columns=self.df_hist.columns)

    def fetch_historical_data(self):
        """Fetches latest historical candles from Binance"""
        klines = self.client.get_klines(
            symbol=self.symbol,
            interval=self.interval,
            limit=self.max_length + 10
        )

        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp", "Open", "High", "Low", "Close", "volume",
                "Close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ]
        )

        numeric_cols = [
            "Open", "High", "Low", "Close", "volume", "quote_asset_volume",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
        ]
        df[numeric_cols] = df[numeric_cols].astype(float)

        # Keep only the most recent max_length rows
        return df.tail(self.max_length).reset_index(drop=True)

    def processDf(self, kline):
        # Refresh historical data on closed candle
        if kline.get('x', False):  # 'x' is True only when the kline is closed
            self.df_hist = self.fetch_historical_data()
            print(f"Refreshed historical DataFrame length: {len(self.df_hist)}")

        # Process and format new kline
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

        new_row = pd.DataFrame([processed])
        new_row = new_row.reindex(columns=self.df_stream.columns)

        self.df_stream = pd.concat([self.df_stream, new_row], ignore_index=True)

        if len(self.df_stream) > self.max_length:
            self.df_stream = self.df_stream.iloc[1:]

        print(f"Live stream DataFrame length: {len(self.df_stream)}")

        return self.df_stream if len(self.df_stream) >= self.max_length else self.df_hist

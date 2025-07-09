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

        # ✅ Load historical klines ONCE during initialization
        klines = self.client.get_klines(
    symbol=self.symbol,
    interval=self.interval,
    limit=self.max_length + 10
)


        # ✅ Create DataFrame from historical data
        self.df_hist = pd.DataFrame(
            klines,
            columns=[
                "timestamp", "Open", "High", "Low", "Close", "volume",
                "Close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ]
        )

        # ✅ Convert numeric columns
        numeric_cols = [
            "Open", "High", "Low", "Close", "volume", "quote_asset_volume",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
        ]
        self.df_hist[numeric_cols] = self.df_hist[numeric_cols].astype(float)

        # ✅ Keep last `max_length` rows
        self.df_hist = self.df_hist.tail(self.max_length).reset_index(drop=True)
        print(f"Historical DataFrame length: {len(self.df_hist)}")

        # ✅ Initialize an empty DataFrame for streaming data
        self.df_stream = pd.DataFrame(columns=self.df_hist.columns)

    def processDf(self, kline):
        # ✅ Parse incoming kline dictionary to match DataFrame format
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

        # ✅ Wrap new row in a DataFrame
        new_row = pd.DataFrame([processed])
        new_row = new_row.reindex(columns=self.df_stream.columns)  # To prevent FutureWarning

        # ✅ Append to stream DataFrame
        self.df_stream = pd.concat([self.df_stream, new_row], ignore_index=True)

        # ✅ Keep only the last `max_length` entries
        if len(self.df_stream) > self.max_length:
            self.df_stream = self.df_stream.iloc[1:]

        # ✅ Debug print
        print(f"DataFrame length: {len(self.df_stream)}")

        # ✅ Return historical until live data is filled up
        if len(self.df_stream) < self.max_length:
            return self.df_hist
        else:
            return self.df_stream

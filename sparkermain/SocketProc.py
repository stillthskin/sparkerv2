
import pandas as pd
from .config import *
from binance.client import Client
from binance.enums import *
import pandas as pd

class SocketProce:
    def __init__(self, symbol, interval):
        self.max_length = 30
        self.client = Client(API_KEY, API_SECRET)
        self.symbol = symbol
        self.interval = interval
        
        # # Load historical data ONCE during initialization
        # klines = self.client.get_historical_klines(
        #     self.symbol, 
        #     self.interval, 
        #     f"40 {self.interval} ago UTC"
        # )
        
        # Initialize df2 with historical data
        self.df2 = pd.DataFrame(
            columns=["timestamp", "Open", "High", "Low", "Close", "volume",
                     "Close_time", "quote_asset_volume", "number_of_trades",
                     "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
        )
        # self.df = pd.DataFrame(
        #     klines,
        #     columns=["timestamp", "Open", "High", "Low", "Close", "volume",
        #              "Close_time", "quote_asset_volume", "number_of_trades",
        #              "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
        # )
        
        # Convert numerical columns to float
        numeric_cols = ["Open", "High", "Low", "Close", "volume", "quote_asset_volume",
                        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
        self.df2[numeric_cols] = self.df2[numeric_cols].astype(float)
        # self.df[numeric_cols] = self.df[numeric_cols].astype(float)

    def processDf(self, kline):
        # Process incoming kline data
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

        # Create DataFrame from new kline
        new_row = pd.DataFrame([processed])

        
        # Append new data using pd.concat
        self.df2 = pd.concat(
            [self.df2, new_row],
            ignore_index=True
        )
        # Maintain rolling window of max_length
        if len(self.df2) > self.max_length:
            #print('Greater removing Un')
            self.df2 = self.df2.iloc[1:]  # Remove oldest row
        if len(self.df2)<self.max_length:
            toreturn=self.df2

        else:
            toreturn=self.df2
        # print(f'Shape of DF {self.df2.shape}')
        return toreturn
            
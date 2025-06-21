
import statistics

import statistics
import numpy as np

class RSI_Calculator:
    def __init__(self, period: int = 14):
        self.period = period

    def calculate_rsi(self, prices: list) -> list:
        """
        Calculate the RSI for a given list of price data.
        :param prices: List of price data (e.g., closing prices).
        :return: List containing the RSI values.
        """
        if len(prices) != self.period + 1:
            raise ValueError(f"List must have exactly {self.period + 1} data points for a single RSI calculation.")

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]  # Price changes
        gains = [delta if delta > 0 else 0 for delta in deltas]  # Positive changes
        losses = [-delta if delta < 0 else 0 for delta in deltas]  # Negative changes

        # Initial average gain/loss over the first 'period' values
        avg_gain = statistics.mean(gains[:self.period])
        avg_loss = statistics.mean(losses[:self.period])

        # Calculate the RSI
        if avg_loss == 0:
            rs = float('inf')  # Avoid division by zero
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        return rsi
    def calculate_Bollinger(self,window:list)->list:
        deviation = 2
        middle_band = np.mean(window)
        std_dev = np.std(window)
        upper_band = middle_band + (deviation * std_dev)
        lower_band = middle_band - (deviation * std_dev)
        return middle_band,upper_band,lower_band


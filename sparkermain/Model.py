import json
import talib as ta
import websocket
import time
import logging
from .Client import Cliente
#from .NModel import AI_Model
from .NmodelII import NModelII
from .SocketProc import SocketProce
import numpy as np
from .config import *
import pandas as pd
from binance.client import Client
from .SocketProc import SocketProce
from binance.enums import *
from django.core.cache import cache
from celery import shared_task

class Model:
    def __init__(self, symbol, trade_quantity, interval='1m', starting_balance=300, stop_loss_pct=0.002, take_profit_pct=0.005):
        self.symbol = symbol
        self.trade_quantity = f"{float(trade_quantity):.8f}"
        self.interval = interval
        self.starting_balance = starting_balance
        self.current_balance = self.starting_balance
        self.client = Client(API_KEY, API_SECRET)
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.predicted_prices = []
        self.closes = []
        self.predictor = None
        self.nModel = None
        self.count = 0
        self.myprocessor = SocketProce(self.symbol, self.interval)
        self.open_positions = {}
        self.socketProce= SocketProce('BTCUSDT', '1m')

    def on_message(self, ws, message):
        try:
            data = json.loads(message)['data']
            if 'k' in data:
                kline = data['k']
                if kline['x']:  
                    self.count += 1
                    close_price = float(kline['c']) 
                    self.closes.append(close_price)
                    print(f"Close price: {close_price}")
                    
                    print(f'count: {self.count}')
                    self.df=self.socketProce.processDf(kline)
                    if self.count>=360:
                        print(f'count: {self.count} - Training model')
                        model=NModelII('BTCUSDT')
                        model.train_model()
                        self.count=0


                    if self.count % 20 == 0:
                         print(f'count: {self.count} - Fetching current assets')
                         theclient = Cliente(
                symbol=self.symbol,
                tradeside=None,
                order_type=ORDER_TYPE_MARKET,
                quantity=self.trade_quantity
            )
                         current_assets= theclient.get_assets()
                         cache.set('current_assets', current_assets, timeout=None)


                    # # Maintain only last 60 closes
                    # if len(self.closes) > 60:
                    # if len(self.df)>=30:
                    #     # self.closes.pop(0)

                    # # Initialize predictor if needed
                    # if self.predictor is None:
                    #     self.predictor = NModelII(API_KEY, API_SECRET)
                    #     self.predictor.train_model()
                    
                    # if self.nModel is None:
                    #     print('Model is None Training')
                    #     self.nModel = AI_Model(self.symbol, self.interval, 30, 10)
                    #     self.nModel.train_model()

                   

                    # Generate predictions
                    # if len(self.closes) >= 60:
                    if len(self.df)>=30:
                        closes_array = np.array(self.closes, dtype=np.float64)
                        print(f"Closes array Length: { len(closes_array)}")
                        if not np.isnan(closes_array).any():
                            rsi_values = ta.RSI(closes_array, timeperiod=14)
                            sma_values = ta.SMA(closes_array, timeperiod=14)
                            rsi = rsi_values[-1]
                            sma = sma_values[-1]
                        else:
                            print("NaN found in closes_array.")
                            rsi = None
                            sma = None
                            

                        # Remove the oldest close to manage memory
                        self.closes.pop(0)

                        if len(self.df)>40:
                            self.df = self.df.tail(40)

                        try:
                            model=NModelII('BTCUSDT')
                            model.load_or_train_model(model_path='model.keras', scaler_path='scaler_close.pkl')
                            print(f' SOme prdiction: {model.predict_next(self.df)}')
                            self.predicted_prices= model.predict_next(self.df)



                        except Exception as e:
                            print(f"Prediction error: {e}")
                            return
                        
                        # try:
                        #     print(f'len(self.df): {len(self.df)}')
                        #     self.predicted_prices = self.nModel.prediction(self.df)
                        #     self.predicted_prices=self.predicted_prices[0]
                        #     print(f"Predictions: {self.predicted_prices}")
                        # except Exception as e:
                        #     print(f"Prediction error: {e}")
                        #     return

                        # Trading logic
                        consecutive_bullish = sum(self.predicted_prices[i] > self.predicted_prices[i-1] for i in range(1,5)) >= 3
                        consecutive_bearish = sum(self.predicted_prices[i] < self.predicted_prices[i-1] for i in range(1,5)) >= 3
                        ma_confirmation_great = np.mean(self.predicted_prices[:3]) > close_price
                        ma_confirmation_less = np.mean(self.predicted_prices[:3]) < close_price

                        should_buy = consecutive_bullish  and rsi < 30
                        should_sell = consecutive_bearish  and rsi > 70
                        print(f"Should buy: {should_buy}, Should sell: {should_sell} RSI: {rsi} consecutive_bullish: {consecutive_bullish} consecutive_bearish: {consecutive_bearish} ma_confirmation_great: {ma_confirmation_great} ma_confirmation_less: {ma_confirmation_less}")

                        # Execute trades
                        if should_buy:
                            print(f"Buy signal detected at {close_price}")
                            self._execute_trade(SIDE_BUY, close_price)
                        elif should_sell:
                            print(f"Sell signal detected at {close_price}")
                            self._execute_trade(SIDE_SELL, close_price)

                        # Check position management
                        self._manage_positions(close_price)

        except Exception as e:
            logging.error(f"Message processing error: {e}")

    def _execute_trade(self, side, price):
        try:
            theclient = Cliente(
                symbol=self.symbol,
                tradeside=side,
                order_type=ORDER_TYPE_MARKET,
                quantity=self.trade_quantity
            )
            res = theclient.order()
            current_assets= theclient.get_assets()
            cache.set('current_assets', current_assets, timeout=None)
            
            if res:
                if side == SIDE_BUY:
                    self.open_positions[self.symbol] = {
                        'entry_price': price,
                        'quantity': self.trade_quantity,
                        'stop_loss': price * (1 - self.stop_loss_pct),
                        'take_profit': price * (1 + self.take_profit_pct)
                    }
                else:
                    if self.symbol in self.open_positions:
                        del self.open_positions[self.symbol]
                
                self.assets = theclient.get_assets()
                cache.set('current_assets', self.assets, timeout=None)
                cache.set('open_positions', self.open_positions, timeout=None)

        except Exception as e:
            print(f"Trade execution failed: {e}")

    def _manage_positions(self, current_price):
        for symbol, position in list(self.open_positions.items()):
            print(f'Managing position for {symbol}: Entry Price: {position["entry_price"]}, Current Price: {current_price}, Stop Loss: {position["stop_loss"]}, Take Profit: {position["take_profit"]}')
            # Check stop loss first
            if current_price <= position['stop_loss']:
                print(f"Stop loss triggered at {current_price}")
                self._close_position(symbol, position, "stop_loss", current_price)
            
            # Check take profit
            elif current_price >= position['take_profit']:
                print(f"Take profit triggered at {current_price}")
                self._close_position(symbol, position, "take_profit", current_price)

    def _close_position(self, symbol, position, reason, current_price):
        try:
            theclient = Cliente(
                symbol=symbol,
                tradeside=SIDE_SELL,
                order_type=ORDER_TYPE_MARKET,
                quantity=position['quantity']
            )
            if theclient.order():
                del self.open_positions[symbol]
                cache.set('open_positions', self.open_positions, timeout=None)
                print(f"Closed position ({reason}) at {current_price}")
                self.assets = theclient.get_assets()
                cache.set('current_assets', self.assets, timeout=None)
        except Exception as e:
            print(f"Position close failed: {e}")

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"Connection closed: {close_status_code} - {close_msg}")

    def on_open(self, ws):
        print(f"WebSocket connected for {self.symbol} ({self.interval})")

    def connect_to_websocket(self):
        while True:
            try:
                ws = websocket.WebSocketApp(
                    f"wss://stream.binance.com:9443/stream?streams={self.symbol.lower()}@kline_{self.interval}",
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                ws.on_open = self.on_open
                ws.run_forever()
            except Exception as e:
                print(f"Connection error: {e}. Reconnecting in 5s...")
                time.sleep(5)
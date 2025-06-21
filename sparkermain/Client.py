from binance.client import Client
from django.core.cache import cache
from .config import API_KEY, API_SECRET


class Cliente:
    def __init__(self, symbol, tradeside, order_type, quantity):
        self.symbol = symbol
        self.tradeside = tradeside
        self.order_type = order_type
        self.quantity = quantity

    def order(self):
        client = Client(API_KEY, API_SECRET)
        try:
            order = client.create_order(symbol=self.symbol, side=self.tradeside, type=self.order_type, quantity=self.quantity)
        except Exception as e:
            print(f'An exception occurred - {e}')
            return False
        return True

    def get_assets(self):
        asset_list = []
        try:
            client = Client(API_KEY, API_SECRET)
            
            # Get balances
            assets = {
                'ETH': client.get_asset_balance(asset='ETH'),
                'BTC': client.get_asset_balance(asset='BTC'),
                'DOGE': client.get_asset_balance(asset='DOGE'),
                'USDT': client.get_asset_balance(asset='USDT'),
            }

            # Structure data for caching
            formatted_assets = []
            for symbol, balance in assets.items():
                formatted_assets.append({
                    'symbol': symbol,
                    'free': float(balance['free']),
                    'locked': float(balance['locked']),
                    'total': float(balance['free']) + float(balance['locked'])
                })

            # Cache for 5 minutes (300 seconds)
           
            return formatted_assets

        except Exception as e:
            # Fallback to fake data
            return [
                {'symbol': 'ETH', 'free': 200.0, 'locked': 0.0, 'total': 200.0},
                {'symbol': 'BTC', 'free': 0.5, 'locked': 0.0, 'total': 0.5},
                {'symbol': 'DOGE', 'free': 1000.0, 'locked': 0.0, 'total': 1000.0},
                {'symbol': 'USDT', 'free': 500.0, 'locked': 0.0, 'total': 500.0},
            ]
    def get_asset(self,thesymbol):
            client = Client(API_KEY, API_SECRET)
            try:
                 asset = client.get_asset_balance(asset=thesymbol)
                 return asset
            except Exception as e:
                 print(f'An exception occured - {e}')


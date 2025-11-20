
import yfinance as yf
import pandas as pd
import json, os

def load_stock_list():
    path = os.path.join(os.path.dirname(__file__), "stocks.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

STOCKS = load_stock_list()

def smart_symbol_search(query):
    if not isinstance(query, str):
        return query
    q = query.lower().strip()
    for name, symbol in STOCKS.items():
        if q == name.lower() or q == symbol.lower().replace('.ns',''):
            return symbol
    for name, symbol in STOCKS.items():
        if q in name.lower():
            return symbol
    return query

def get_stock_data(symbol, interval='5m', limit=5000):
    symbol = smart_symbol_search(symbol)
    interval_map = {'1m':'7d','2m':'60d','5m':'60d','15m':'60d','30m':'60d','1h':'730d','1d':'max'}
    period = interval_map.get(interval, '60d')
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    except Exception as e:
        print('yfinance error:', e)
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.tail(limit)
    for col in ['Open','High','Low','Close','Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(how='any', subset=['Open','High','Low','Close'], inplace=True)
    return df

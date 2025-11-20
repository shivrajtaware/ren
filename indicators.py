
import pandas as pd
import numpy as np

def EMA_series(series, length):
    return series.ewm(span=length, adjust=False).mean()

def compute_macd(df):
    df['EMA12'] = EMA_series(df['Close'], 12)
    df['EMA26'] = EMA_series(df['Close'], 26)
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_HIST'] = df['MACD'] - df['Signal']
    return df

def compute_rsi(df, length=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def compute_bollinger(df, length=20):
    df['BB_MID'] = df['Close'].rolling(length).mean()
    df['BB_STD'] = df['Close'].rolling(length).std()
    df['BB_UPPER'] = df['BB_MID'] + 2 * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - 2 * df['BB_STD']
    return df

def compute_volumesurge(df):
    if 'Volume' not in df.columns:
        df['VolSurge'] = False
        return df
    df['AvgVol'] = df['Volume'].rolling(20).mean()
    vol = df['Volume'].fillna(0).to_numpy()
    avg = (df['AvgVol'].fillna(0) * 2).to_numpy()
    df['VolSurge'] = vol > avg
    return df

def compute_emas(df, periods):
    for p in periods:
        df[f'EMA{p}'] = EMA_series(df['Close'], p)
    return df

def apply_indicators(df, selected=None, ema_periods=None):
    if df is None or df.empty:
        return df
    if selected is None:
        selected = []
    if ema_periods is None:
        ema_periods = [20,50,200]
    if 'EMA' in selected:
        df = compute_emas(df, ema_periods)
    if 'MACD' in selected:
        df = compute_macd(df)
    if 'RSI' in selected:
        df = compute_rsi(df)
    if 'Bollinger' in selected:
        df = compute_bollinger(df)
    if 'VolSurge' in selected:
        df = compute_volumesurge(df)
    df.fillna(value=pd.NA, inplace=True)
    return df

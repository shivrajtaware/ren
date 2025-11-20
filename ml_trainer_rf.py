
import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from data_fetcher import get_stock_data
from indicators import apply_indicators

MODEL_DIR='models'; os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE=os.path.join(MODEL_DIR,'rf_next_dir.pkl'); SCALER_FILE=os.path.join(MODEL_DIR,'rf_scaler.pkl')

def main(symbol='RELIANCE.NS', interval='5m'):
    df = get_stock_data(symbol, interval=interval, limit=5000)
    if df.empty: print('no data'); return
    df = apply_indicators(df, selected={'EMA','Bollinger','RSI','VolSurge','MACD'}, ema_periods=[9,20,50])
    df['label'] = (df['Close'].shift(-1) > df['Close']).astype(int); df.dropna(inplace=True)
    feature_cols = ['Open','High','Low','Close','Volume','EMA9','EMA20','EMA50','RSI','BB_UPPER','BB_LOWER','AvgVol']
    for c in ['MACD','Signal']:
        if c in df.columns and c not in feature_cols: feature_cols.append(c)
    feature_cols = [c for c in feature_cols if c in df.columns]
    if len(feature_cols)==0: print('no features'); return
    X = df[feature_cols].values; y = df['label'].values
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(Xs, y)
    joblib.dump(clf, MODEL_FILE); joblib.dump(scaler, SCALER_FILE)
    print('RF saved to', MODEL_DIR)

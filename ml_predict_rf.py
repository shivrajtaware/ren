
import os, joblib, numpy as np
from data_fetcher import get_stock_data
from indicators import apply_indicators

MODEL_DIR='models'; MODEL_FILE=os.path.join(MODEL_DIR,'rf_next_dir.pkl'); SCALER_FILE=os.path.join(MODEL_DIR,'rf_scaler.pkl')

def predict_next(symbol, interval='5m', selected=None):
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        return None
    clf = joblib.load(MODEL_FILE); scaler = joblib.load(SCALER_FILE)
    df = get_stock_data(symbol, interval=interval, limit=5000)
    if df.empty: return None
    df = apply_indicators(df, selected=set(selected) if selected else {'EMA','Bollinger','RSI','VolSurge','MACD'}, ema_periods=[9,20,50])
    feature_cols = ['Open','High','Low','Close','Volume','EMA9','EMA20','EMA50','RSI','BB_UPPER','BB_LOWER','AvgVol']
    for c in ['MACD','Signal']:
        if c in df.columns and c not in feature_cols: feature_cols.append(c)
    feature_cols = [c for c in feature_cols if c in df.columns]
    if len(feature_cols)==0: return None
    X = df[feature_cols].values[-1].reshape(1,-1)
    Xs = scaler.transform(X)
    prob = clf.predict_proba(Xs)[0][1]
    return {'prob_up': float(prob), 'direction': 1 if prob>=0.5 else 0}

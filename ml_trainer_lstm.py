
import os, joblib
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from data_fetcher import get_stock_data
from indicators import apply_indicators

MODEL_DIR = 'models'; os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR,'lstm_next_dir.h5'); SCALER_FILE = os.path.join(MODEL_DIR,'lstm_scaler.pkl')
SEQ = 30

def label(df):
    df = df.copy(); df['Close_next'] = df['Close'].shift(-1); df['label'] = (df['Close_next']>df['Close']).astype(int); df.dropna(inplace=True); return df

def build_sequences(arr, seq):
    X=[]; y=[]
    for i in range(len(arr)-seq+1):
        X.append(arr[i:i+seq]); y.append(arr[i+seq-1,-1])
    return np.array(X), np.array(y)

def build_model(n_features):
    model = Sequential()
    model.add(LSTM(64, input_shape=(SEQ, n_features), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main(symbol='RELIANCE.NS', interval='5m'):
    df = get_stock_data(symbol, interval=interval, limit=5000)
    if df.empty: print('no data'); return
    df = apply_indicators(df, selected={'EMA','Bollinger','RSI','VolSurge','MACD'}, ema_periods=[9,20,50])
    df = label(df)
    feature_cols = ['Open','High','Low','Close','Volume','EMA9','EMA20','EMA50','RSI','BB_UPPER','BB_LOWER','AvgVol']
    for c in ['MACD','Signal']:
        if c in df.columns and c not in feature_cols: feature_cols.append(c)
    feature_cols = [c for c in feature_cols if c in df.columns]
    if len(feature_cols)==0: print('no features'); return
    scaler = StandardScaler(); scaled = scaler.fit_transform(df[feature_cols].values)
    X, y = build_sequences(np.hstack([scaled, df[['label']].values]), SEQ)
    if len(X) < 10:
        print('not enough data to train here (need more history)')
        n_features = len(feature_cols)
        model = build_model(n_features)
        model.save(MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        print('Saved default model and scaler for compatibility.')
        return
    split = int(0.8*len(X)); Xtr, Xval = X[:split], X[split:]; ytr, yval = y[:split], y[split:]
    n_features = X.shape[2]-1
    model = build_model(n_features)
    model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=5, batch_size=64, verbose=2)
    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print('Trained and saved model & scaler to', MODEL_DIR)

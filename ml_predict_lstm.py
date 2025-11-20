
import os, joblib, numpy as np
from data_fetcher import get_stock_data
from indicators import apply_indicators

MODEL_DIR = 'models'
MODEL_FILE = os.path.join(MODEL_DIR,'lstm_next_dir.h5')
SCALER_FILE = os.path.join(MODEL_DIR,'lstm_scaler.pkl')
SEQ = 30

def _has_tf_model():
    return os.path.exists(MODEL_FILE)

def predict_next(symbol, interval='5m', selected=None):
    if _has_tf_model():
        try:
            from tensorflow.keras.models import load_model
            model = load_model(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            df = get_stock_data(symbol, interval=interval, limit=5000)
            if df.empty: return None
            df = apply_indicators(df, selected=set(selected) if selected else {'EMA','Bollinger','RSI','VolSurge','MACD'}, ema_periods=[9,20,50])
            feature_cols = ['Open','High','Low','Close','Volume','EMA9','EMA20','EMA50','RSI','BB_UPPER','BB_LOWER','AvgVol']
            for c in ['MACD','Signal']:
                if c in df.columns and c not in feature_cols: feature_cols.append(c)
            feature_cols = [c for c in feature_cols if c in df.columns]
            if len(feature_cols)==0: return None
            if getattr(scaler, 'n_features_in_', None) != len(feature_cols):
                return None
            scaled = scaler.transform(df[feature_cols].values)
            if len(scaled) < SEQ: return None
            seq = scaled[-SEQ:]
            seq = seq.reshape(1, SEQ, len(feature_cols))
            prob = float(model.predict(seq, verbose=0)[0][0])
            return {'prob_up': prob, 'direction': 1 if prob>=0.5 else 0}
        except Exception as e:
            print("LSTM load error, falling back to RF:", e)

    try:
        import ml_predict_rf
        return ml_predict_rf.predict_next(symbol, interval=interval, selected=selected)
    except Exception as e:
        print("Fallback RF predictor failed:", e)
        return None

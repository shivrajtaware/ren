
import streamlit as st
import pandas as pd, time, os
from data_fetcher import get_stock_data
from indicators import apply_indicators
from chart_builder import make_chart
from predictor import predict_rule_based
import ml_predict_lstm, ml_predict_rf

st.set_page_config(page_title='Stock Dashboard Pro', layout='wide')
st.title('📈 Stock Dashboard Pro (Indicators + ML)')

with st.sidebar:
    st.header('Settings')
    stock = st.text_input('Stock (smart search)', 'Reliance')
    interval = st.selectbox('Interval', ['1m','5m','15m','5m','1h','1d'], index=1)
    st.markdown('---')
    st.subheader('Indicators')
    ind = st.multiselect('Select indicators', options=['EMA','Bollinger','MACD','RSI','VolSurge'], default=['EMA','Bollinger','VolSurge'])
    ema_periods = st.multiselect('EMA periods', options=[9,12,20,50,100,200], default=[20,50,200])
    st.markdown('---')
    st.subheader('ML Models')
    use_lstm = st.checkbox('Enable LSTM prediction', value=True)
    use_rf = st.checkbox('Enable RandomForest prediction', value=True)
    seq_len = st.number_input('LSTM sequence length', min_value=20, max_value=240, value=30)
    st.markdown('---')
    st.subheader('Actions')
    live = st.checkbox('Auto-refresh every 10s (use with caution)', value=False)
    if st.button('Refresh now'):
        st.experimental_rerun()
    st.markdown('---')
    if st.button('Retrain LSTM (run ml_trainer_lstm.py in terminal)'):
        st.info('Run training in a separate terminal for best results.')
    if st.button('Retrain RF (run ml_trainer_rf.py in terminal)'):
        st.info('Run training in a separate terminal for best results.')

# placeholders
chart_slot = st.empty()
pred_col1, pred_col2 = st.columns([1,1])

# Single render per run (no while loop)
df = get_stock_data(stock, interval=interval, limit=20000)
if df.empty:
    st.error('No data for symbol/interval. Try another.')
else:
    df = apply_indicators(df, selected=set(ind), ema_periods=ema_periods)
    visible_df = df.copy()
    fig = make_chart(visible_df, selected=set(ind), ema_periods=ema_periods, show_volume=True)
    chart_key = f"chart_{stock.replace(' ','_')}_{interval}"
    chart_slot.plotly_chart(fig, use_container_width=True, key=chart_key)

    with pred_col1:
        st.subheader('Rule-based')
        st.write(predict_rule_based(df, selected=set(ind)))
    with pred_col2:
        st.subheader('ML Predictions')
        if use_rf:
            try:
                rf_res = ml_predict_rf.predict_next(stock if '.' in stock else stock, interval=interval, selected=set(ind))
                if rf_res: st.write(f\"RandomForest: {rf_res['prob_up']*100:.1f}% up\") 
                else: st.write('RandomForest: No model')
            except Exception as e:
                st.write('RandomForest error:', str(e))
        if use_lstm:
            try:
                lstm_res = ml_predict_lstm.predict_next(stock if '.' in stock else stock, interval=interval, selected=set(ind))
                if lstm_res: st.write(f\"LSTM: {lstm_res['prob_up']*100:.1f}% up\") 
                else: st.write('LSTM: No model or not enough history')
            except Exception as e:
                st.write('LSTM error:', str(e))

# auto-refresh logic (optional)
if live:
    time.sleep(10)
    st.experimental_rerun()

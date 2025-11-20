
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def make_chart(df, selected=None, ema_periods=None, show_volume=True):
    if selected is None:
        selected = set()
    if ema_periods is None:
        ema_periods = [20,50,200]
    rows = 1
    has_volume = show_volume and ('Volume' in df.columns)
    if has_volume:
        rows += 1
    macd_on = 'MACD' in selected and 'MACD' in df.columns and 'Signal' in df.columns
    rsi_on = 'RSI' in selected and 'RSI' in df.columns
    if macd_on:
        rows += 1
    if rsi_on:
        rows += 1
    heights = [0.55] + [0.15]*(rows-1)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=heights)
    row = 1
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candles'), row=row, col=1)
    if 'EMA' in selected:
        for p in ema_periods:
            colname = f'EMA{p}'
            if colname in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[colname], name=colname, line=dict(width=1)), row=row, col=1)
    if 'Bollinger' in selected and 'BB_UPPER' in df.columns and 'BB_LOWER' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], name='BB Upper', line=dict(width=1), opacity=0.6), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], name='BB Lower', line=dict(width=1), opacity=0.6, fill='tonexty'), row=row, col=1)
    if has_volume:
        row += 1
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker=dict(opacity=0.6)), row=row, col=1)
    if macd_on:
        row += 1
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal'), row=row, col=1)
        if 'MACD_HIST' in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_HIST'], name='MACD Hist', marker=dict(opacity=0.5)), row=row, col=1)
    if rsi_on:
        row += 1
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=row, col=1)
        fig.add_hline(y=70, line_dash='dash', row=row, col=1, annotation_text='70', opacity=0.5)
        fig.add_hline(y=30, line_dash='dash', row=row, col=1, annotation_text='30', opacity=0.5)
    fig.update_layout(template='plotly_dark', height=900, margin=dict(l=40,r=20,t=60,b=40), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.2, gridcolor='gray')
    return fig


def predict_rule_based(df, selected=None):
    if df is None or df.empty:
        return 'No data'
    sel = set(selected) if selected is not None else None
    last = df.iloc[-1]
    score = 0; reasons = []
    def use(col):
        if sel is not None and col not in sel and col.split('.')[0] not in sel:
            return False
        return col in df.columns
    if use('MACD') and use('Signal'):
        try:
            if last['MACD'] > last['Signal']:
                score += 1; reasons.append('MACD>Signal')
            else:
                score -= 1; reasons.append('MACD<Signal')
        except: pass
    if use('RSI'):
        try:
            r = last['RSI']
            if r < 30: score += 1; reasons.append('RSI low')
            elif r > 70: score -= 1; reasons.append('RSI high')
        except: pass
    if use('EMA20') and use('EMA50'):
        try:
            if last['EMA20'] > last['EMA50']:
                score += 1; reasons.append('EMA20>EMA50')
            else:
                score -= 1; reasons.append('EMA20<EMA50')
        except: pass
    if use('VolSurge'):
        try:
            if bool(last['VolSurge']): score += 1; reasons.append('Vol Surge')
        except: pass
    if score >= 2: return '📈 Uptrend Expected — ' + '; '.join(reasons)
    if score <= -2: return '📉 Downtrend Expected — ' + '; '.join(reasons)
    if reasons: return '⚠️ Sideways — ' + '; '.join(reasons)
    return '⚠️ Not enough data'

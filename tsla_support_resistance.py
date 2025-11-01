# stock_analysis.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import smtplib
import ssl
from email.message import EmailMessage

st.set_page_config(page_title="股票 5分鐘分析", layout="wide")
st.title("股票 5分鐘K線 + 關鍵水平 + 交易建議")

# ==================== 側邊欄 ====================
stock_symbol = st.sidebar.text_input("股票代碼", "TSLA").upper()
lookback = st.sidebar.slider("回看K線數", 20, 200, 50)
tolerance_pct = st.sidebar.slider("價格容忍 (%)", 0.1, 2.0, 0.5) / 100
min_touches = st.sidebar.slider("最少觸及次數", 2, 6, 3)
enable_email = st.sidebar.checkbox("啟用 Email 突破警示")

# ==================== Email 發送 ====================
@st.cache_data
def get_email_config():
    try:
        return {
            "pw": st.secrets["send_email"]["gmail_app_password"],
            "sender": st.secrets["send_email"]["sender_email"],
            "receiver": st.secrets["send_email"]["receiver_email"]
        }
    except:
        return None

def send_email(symbol, direction, price, level):
    if not enable_email: return
    cfg = get_email_config()
    if not cfg: return
    msg = EmailMessage()
    msg['From'] = cfg["sender"]
    msg['To'] = cfg["receiver"]
    msg['Subject'] = f"{symbol} {direction} ${level}"
    msg.set_content(f"股票: {symbol}\n{direction} ${level}\n價格: ${price:.2f}\n時間: {datetime.now():%H:%M}")
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(cfg["sender"], cfg["pw"])
            s.send_message(msg)
        st.sidebar.success(f"Email 已發送")
    except Exception as e:
        st.sidebar.error(f"Email 失敗: {e}")

# ==================== 抓資料 ====================
@st.cache_data(ttl=60)
def fetch_data(symbol):
    try:
        df = yf.Ticker(symbol).history(period="5d", interval="5m")
        if df.empty: return None
        df = df.dropna().copy()
        df.index = df.index.tz_convert('America/New_York')
        return df
    except: return None

with st.spinner(f"抓取 {stock_symbol}..."):
    data = fetch_data(stock_symbol)

if data is None or data.empty:
    st.error("無法取得數據")
    st.stop()

data = data.tail(lookback).copy()
data.reset_index(inplace=True)
data['time'] = data['Datetime'].dt.strftime('%H:%M')

# ==================== RSI / MACD ====================
def calc_rsi(s, p=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(p).mean()
    down = -d.clip(upper=0).rolling(p).mean()
    return 100 - (100 / (1 + up/down))

def calc_macd(s, f=12, sl=26, sig=9):
    emaf = s.ewm(span=f, adjust=False).mean()
    emas = s.ewm(span=sl, adjust=False).mean()
    macd = emaf - emas
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

data['RSI'] = calc_rsi(data['Close'])
data['MACD'], data['MACD_signal'], data['MACD_hist'] = calc_macd(data['Close'])

# ==================== 支撐阻力 ====================
def find_levels(df, tol, min_touch):
    lows, highs, closes = df['Low'].values, df['High'].values, df['Close'].values
    prices = np.concatenate([lows, highs])
    sorted_p = np.sort(prices)
    clusters, cur = [], [sorted_p[0]]
    for p in sorted_p[1:]:
        if p <= cur[-1] * (1 + tol): cur.append(p)
        else: clusters.append(np.mean(cur)); cur = [p]
    if cur: clusters.append(np.mean(cur))

    levels = []
    for lvl in clusters:
        touch = sum(abs(l-lvl)<=lvl*tol for l in lows) + \
                sum(abs(h-lvl)<=lvl*tol for h in highs) + \
                sum(abs(c-lvl)<=lvl*tol for c in closes)
        if touch >= min_touch:
            is_sup = any(abs(l-lvl)<=lvl*tol for l in lows)
            is_res = any(abs(h-lvl)<=lvl*tol for h in highs)
            typ = "S/R" if is_sup and is_res else ("Support" if is_sup else "Resistance")
            levels.append({"price": round(lvl,2), "touches": touch, "type": typ})

    supports = sorted([x for x in levels if "Support" in x["type"]], key=lambda x: x["price"])
    resists  = sorted([x for x in levels if "Resistance" in x["type"]], key=lambda x: x["price"], reverse=True)
    return supports[:2], resists[:2], levels

supports, resists, all_levels = find_levels(data, tolerance_pct, min_touches)

# ==================== 關鍵水平表格 ====================
st.subheader("當前關鍵水平")
level_data = []
for i, r in enumerate(resists):
    level_data.append({
        "類型": f"阻力 R{i+1}",
        "價格": f"${r['price']}",
        "觸及次數": f"{r['touches']} 次",
        "說明": "短期賣壓強" if i == 0 else "中期天花板"
    })
for i, s in enumerate(supports):
    level_data.append({
        "類型": f"支撐 S{i+1}",
        "價格": f"${s['price']}",
        "觸及次數": f"{s['touches']} 次",
        "說明": "強力支撐" if i == 0 else "中間支撐"
    })

if level_data:
    df_levels = pd.DataFrame(level_data)
    st.table(df_levels)
else:
    st.info("未偵測到足夠觸及的水平線")

# ==================== 交易建議 ====================
cur_price = data['Close'].iloc[-1]
prev_price = data['Close'].iloc[-2] if len(data) > 1 else cur_price

# 突破偵測
for r in resists:
    if prev_price < r['price'] <= cur_price:
        st.warning(f"突破阻力 ${r['price']}！")
        send_email(stock_symbol, "上漲突破", cur_price, r['price'])

for s in supports:
    if prev_price > s['price'] >= cur_price:
        st.error(f"跌破支撐 ${s['price']}！")
        send_email(stock_symbol, "下跌突破", cur_price, s['price'])

# 交易建議
st.subheader("交易建議（示意）")
r1 = resists[0]['price'] if resists else None
r2 = resists[1]['price'] if len(resists) > 1 else None
s1 = supports[0]['price'] if supports else None
s2 = supports[1]['price'] if len(supports) > 1 else None

suggestion = ""
if r1 and cur_price > r1:
    suggestion = f"**突破 {r1} 並站穩** → 看多至 {r2 if r2 else '更高阻力'}。"
elif s1 and cur_price < s1:
    suggestion = f"**跌破 {s1}** → 可能下探 {s1-5 if s1 else '更低'} 或更低。"
elif s1 and r1 and s1 < cur_price < r1:
    suggestion = f"目前價格在 **S1 ~ R1 之間震盪**，適合區間操作。"
else:
    suggestion = "觀察中，等待明確訊號。"

st.markdown(suggestion)

# ==================== 圖表 ====================
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    subplot_titles=("K線 + 成交量", "RSI", "MACD", "支撐/阻力"),
                    row_heights=[0.5,0.2,0.2,0.1])

# K線 + 成交量
colors = ['green' if o < c else 'red' for o, c in zip(data['Open'], data['Close'])]
fig.add_trace(go.Candlestick(x=data['time'], open=data['Open'], high=data['High'],
                             low=data['Low'], close=data['Close'], name="K線"), row=1, col=1)
fig.add_trace(go.Bar(x=data['time'], y=data['Volume'], name="成交量", marker_color=colors), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=data['time'], y=data['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

# MACD
fig.add_trace(go.Scatter(x=data['time'], y=data['MACD'], name="MACD", line=dict(color='blue')), row=3, col=1)
fig.add_trace(go.Scatter(x=data['time'], y=data['MACD_signal'], name="Signal", line=dict(color='orange')), row=3, col=1)
fig.add_trace(go.Bar(x=data['time'], y=data['MACD_hist'], name="Hist", marker_color='gray'), row=3, col=1)

# 水平線
for s in supports: fig.add_hline(y=s['price'], line_color="green", line_dash="dash", row=4, col=1)
for r in resists:  fig.add_hline(y=r['price'], line_color="red",   line_dash="dash", row=4, col=1)

fig.update_layout(height=900, title=f"{stock_symbol} 5分鐘圖表", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ==================== 指標 ====================
c1, c2, c3 = st.columns(3)
c1.metric("最新價格", f"${cur_price:.2f}")
c2.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
c3.metric("成交量", f"{data['Volume'].iloc[-1]:,.0f}")

st.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if st.button("刷新"):
    st.cache_data.clear()
    st.rerun()

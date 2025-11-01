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

# ==================== 設定 ====================
st.set_page_config(page_title="股票 5分鐘分析", layout="wide")
st.title("股票 5分鐘K線 + 成交量 + RSI + MACD + 支撐阻力 + Email 警示")

# 側邊欄參數
stock_symbol = st.sidebar.text_input("股票代碼", "TSLA").upper()
lookback = st.sidebar.slider("回看K線數", 20, 200, 50)
tolerance_pct = st.sidebar.slider("價格容忍 (%)", 0.1, 2.0, 0.5) / 100
min_touches = st.sidebar.slider("最少觸及次數", 2, 6, 3)
enable_email = st.sidebar.checkbox("啟用 Email 突破警示")

# ==================== Email 設定 ====================
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
    msg.set_content(f"""
    股票: {symbol}
    突破: {direction} ${level}
    價格: ${price:.2f}
    時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls(context=context)
            server.login(cfg["sender"], cfg["pw"])
            server.send_message(msg)
        st.sidebar.success(f"Email 已發送: {direction} ${level}")
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
    except:
        return None

with st.spinner(f"正在抓取 {stock_symbol} 5分鐘數據..."):
    data = fetch_data(stock_symbol)

if data is None or data.empty:
    st.error(f"無法取得 {stock_symbol} 數據，請檢查代碼或網路")
    st.stop()

data = data.tail(lookback).copy()
data.reset_index(inplace=True)
data['time'] = data['Datetime'].dt.strftime('%H:%M')

# ==================== 手寫 RSI ====================
def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calc_rsi(data['Close'])

# ==================== 手寫 MACD ====================
def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

data['MACD'], data['MACD_signal'], data['MACD_hist'] = calc_macd(data['Close'])

# ==================== 支撐阻力 ====================
def find_levels(df, tol, min_touch):
    lows = df['Low'].values
    highs = df['High'].values
    closes = df['Close'].values
    prices = np.concatenate([lows, highs])
    sorted_p = np.sort(prices)
    clusters = []
    cur = [sorted_p[0]]

    for p in sorted_p[1:]:
        if p <= cur[-1] * (1 + tol):
            cur.append(p)
        else:
            clusters.append(np.mean(cur))
            cur = [p]
    if cur:
        clusters.append(np.mean(cur))

    levels = []
    for lvl in clusters:
        touch = sum(abs(l-lvl) <= lvl*tol for l in lows)
        touch += sum(abs(h-lvl) <= lvl*tol for h in highs)
        touch += sum(abs(c-lvl) <= lvl*tol for c in closes)
        if touch >= min_touch:
            is_sup = any(abs(l-lvl) <= lvl*tol for l in lows)
            is_res = any(abs(h-lvl) <= lvl*tol for h in highs)
            typ = "S/R" if is_sup and is_res else ("Support" if is_sup else "Resistance")
            levels.append({"price": round(lvl, 2), "touches": touch, "type": typ})

    supports = sorted([x for x in levels if "Support" in x["type"]], key=lambda x: x["price"])
    resists  = sorted([x for x in levels if "Resistance" in x["type"]], key=lambda x: x["price"], reverse=True)
    return supports[:3], resists[:3], levels

supports, resists, all_levels = find_levels(data, tolerance_pct, min_touches)

# ==================== 突破偵測 ====================
cur_price = data['Close'].iloc[-1]
prev_price = data['Close'].iloc[-2] if len(data) > 1 else cur_price

for r in resists:
    if prev_price < r['price'] <= cur_price:
        st.warning(f"突破阻力 ${r['price']}")
        send_email(stock_symbol, "上漲突破", cur_price, r['price'])

for s in supports:
    if prev_price > s['price'] >= cur_price:
        st.error(f"跌破支撐 ${s['price']}")
        send_email(stock_symbol, "下跌突破", cur_price, s['price'])

# ==================== 繪圖 ====================
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=("K線 + 成交量", "RSI (14)", "MACD (12,26,9)", "支撐/阻力"),
    row_heights=[0.5, 0.2, 0.2, 0.1]
)

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
fig.add_trace(go.Bar(x=data['time'], y=data['MACD_hist'], name="Histogram", marker_color='gray'), row=3, col=1)

# 支撐阻力
for s in supports:
    fig.add_hline(y=s['price'], line_color="green", line_dash="dash",
                  annotation_text=f"S ${s['price']} ({s['touches']}次)", row=4, col=1)
for r in resists:
    fig.add_hline(y=r['price'], line_color="red", line_dash="dash",
                  annotation_text=f"R ${r['price']} ({r['touches']}次)", row=4, col=1)

fig.update_layout(height=900, title=f"{stock_symbol} 5分鐘圖表（最近 {lookback} 根）", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ==================== 表格與指標 ====================
st.subheader("關鍵支撐與阻力水平")
if all_levels:
    df_levels = pd.DataFrame(all_levels).sort_values("price", ascending=False)
    st.table(df_levels.style.format({"price": "${:.2f}"}))
else:
    st.info("未偵測到符合條件的水平線")

# 最新數據
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("最新價格", f"${cur_price:.2f}")
with col2: st.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
with col3: st.metric("MACD", f"{data['MACD'].iloc[-1]:.3f}")
with col4: st.metric("成交量", f"{data['Volume'].iloc[-1]:,.0f}")

# 提示
if data['RSI'].iloc[-1] > 70:
    st.warning("RSI 超買 (>70)")
elif data['RSI'].iloc[-1] < 30:
    st.success("RSI 超賣 (<30)")

st.caption(f"數據來源：Yahoo Finance | 更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 刷新按鈕
if st.button("刷新數據"):
    st.cache_data.clear()
    st.rerun()

# tsla_support_resistance.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta  # <-- 改用 pandas_ta
from datetime import datetime
import smtplib  # <-- 內建，無需安裝
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

st.set_page_config(page_title="股票 5分鐘 支撐阻力分析", layout="wide")

st.title("股票 5分鐘K線 + 支撐/阻力 + 指標 + 成交量 + 突破警示")

# --- 參數設定 ---
stock_symbol = st.sidebar.text_input("股票代碼", value="TSLA", help="如: AAPL, NVDA, TSLA")
lookback = st.sidebar.slider("回看K線數", min_value=20, max_value=200, value=50, step=10)
tolerance_pct = st.sidebar.slider("價格容忍區間 (%)", 0.1, 2.0, 0.5, 0.1) / 100
min_touches = st.sidebar.slider("最少觸及次數", 2, 6, 3)
enable_email = st.sidebar.checkbox("啟用 Email 突破警示", help="需配置 .streamlit/secrets.toml")

# --- Email 配置 ---
@st.cache_data
def get_email_config():
    try:
        return {
            "gmail_app_password": st.secrets["send_email"]["gmail_app_password"],
            "sender_email": st.secrets["send_email"]["sender_email"],
            "receiver_email": st.secrets["send_email"]["receiver_email"]
        }
    except:
        st.warning("Email 配置未完成，請檢查 `.streamlit/secrets.toml`")
        return None

def send_breakout_email(symbol, direction, price, level, timestamp):
    if not enable_email:
        return
    config = get_email_config()
    if not config:
        return
    try:
        msg = MimeMultipart()
        msg['From'] = config["sender_email"]
        msg['To'] = config["receiver_email"]
        msg['Subject'] = f"突破警示: {symbol} {direction} ${level}"
        
        body = f"""
        股票: {symbol}
        突破方向: {direction} ${level}
        當前價格: ${price:.2f}
        時間: {timestamp}
        """
        msg.attach(MimeText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(config["sender_email"], config["gmail_app_password"])
        server.sendmail(config["sender_email"], config["receiver_email"], msg.as_string())
        server.quit()
        st.sidebar.success(f"Email 已發送: {direction} ${level}")
    except Exception as e:
        st.sidebar.error(f"Email 失敗: {e}")

# --- 抓取數據 ---
@st.cache_data(ttl=60)
def get_data(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="5d", interval="5m")
    if df.empty:
        return None
    df = df.dropna().copy()
    df.index = df.index.tz_convert('America/New_York')
    return df

with st.spinner(f"正在抓取 {stock_symbol} 數據..."):
    data = get_data(stock_symbol.upper())

if data is None or data.empty:
    st.error(f"無法取得 {stock_symbol} 數據")
    st.stop()

data = data.tail(lookback).copy()
data.reset_index(inplace=True)
data['time'] = data['Datetime'].dt.strftime('%H:%M')

# --- 計算指標（使用 pandas_ta）---
data['RSI'] = ta.rsi(data['Close'], length=14)
macd = ta.macd(data['Close'])
data['MACD'] = macd['MACD_12_26_9']
data['MACD_signal'] = macd['MACDs_12_26_9']
data['MACD_histogram'] = macd['MACDh_12_26_9']

# --- 計算支撐阻力 ---
def find_levels(df, tolerance_pct=0.005, min_touches=3):
    lows = df['Low'].values
    highs = df['High'].values
    closes = df['Close'].values
    prices = np.concatenate([lows, highs])
    sorted_prices = np.sort(prices)
    clusters = []
    current = [sorted_prices[0]]

    for p in sorted_prices[1:]:
        if p <= current[-1] * (1 + tolerance_pct):
            current.append(p)
        else:
            clusters.append(np.mean(current))
            current = [p]
    if current:
        clusters.append(np.mean(current))

    levels = []
    for level in clusters:
        touches = sum(abs(l - level) <= level * tolerance_pct for l in lows)
        touches += sum(abs(h - level) <= level * tolerance_pct for h in highs)
        touches += sum(abs(c - level) <= level * tolerance_pct for c in closes)

        if touches >= min_touches:
            is_support = any(abs(l - level) <= level * tolerance_pct for l in lows)
            is_resistance = any(abs(h - level) <= level * tolerance_pct for h in highs)
            typ = "S/R" if is_support and is_resistance else ("Support" if is_support else "Resistance")
            levels.append({'price': round(level, 2), 'touches': touches, 'type': typ})

    supports = sorted([l for l in levels if "Support" in l['type']], key=lambda x: x['price'])
    resistances = sorted([l for l in levels if "Resistance" in l['type']], key=lambda x: x['price'], reverse=True)
    return supports[:3], resistances[:3], levels

supports, resistances, all_levels = find_levels(data, tolerance_pct, min_touches)

# --- 突破偵測 ---
current_price = data['Close'].iloc[-1]
prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price

for r in resistances:
    if prev_price < r['price'] <= current_price:
        st.warning(f"突破阻力 ${r['price']}！")
        send_breakout_email(stock_symbol, "上漲突破", current_price, r['price'], data['time'].iloc[-1])

for s in supports:
    if prev_price > s['price'] >= current_price:
        st.error(f"跌破支撐 ${s['price']}！")
        send_breakout_email(stock_symbol, "下跌突破", current_price, s['price'], data['time'].iloc[-1])

# --- 繪圖 ---
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
    subplot_titles=('K線 + 成交量', 'RSI', 'MACD', '支撐/阻力'),
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
fig.add_trace(go.Bar(x=data['time'], y=data['MACD_histogram'], name="Hist", marker_color='gray'), row=3, col=1)

# 水平線
for s in supports:
    fig.add_hline(y=s['price'], line_color="green", line_dash="dash", row=4, col=1)
for r in resistances:
    fig.add_hline(y=r['price'], line_color="red", line_dash="dash", row=4, col=1)

fig.update_layout(height=900, title=f"{stock_symbol} 5分鐘圖表", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# --- 表格與提示 ---
st.subheader("關鍵水平")
if all_levels:
    df_levels = pd.DataFrame(all_levels).sort_values("price", ascending=False)
    st.table(df_levels.style.format({"price": "${:.2f}"}))

col1, col2, col3 = st.columns(3)
with col1: st.metric("最新價", f"${current_price:.2f}")
with col2: st.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
with col3: st.metric("成交量", f"{data['Volume'].iloc[-1]:,.0f}")

st.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if st.button("刷新"):
    st.cache_data.clear()
    st.rerun()

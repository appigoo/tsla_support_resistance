# tsla_support_resistance.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.trend import MACD
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import json

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
    return {
        "gmail_app_password": st.secrets["send_email"]["gmail_app_password"],
        "sender_email": st.secrets["send_email"]["sender_email"],
        "receiver_email": st.secrets["send_email"]["receiver_email"]
    }

def send_breakout_email(symbol, direction, price, level, timestamp):
    if not enable_email:
        return
    try:
        config = get_email_config()
        msg = MimeMultipart()
        msg['From'] = config["sender_email"]
        msg['To'] = config["receiver_email"]
        msg['Subject'] = f"🚨 {symbol} 突破警示: {direction} ${level}"
        
        body = f"""
        股票: {symbol}
        突破方向: {direction} ${level}
        當前價格: ${price:.2f}
        時間: {timestamp}
        請檢查圖表確認!
        """
        msg.attach(MimeText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(config["sender_email"], config["gmail_app_password"])
        text = msg.as_string()
        server.sendmail(config["sender_email"], config["receiver_email"], text)
        server.quit()
        st.sidebar.success(f"Email 已發送: {direction} 突破 ${level}")
    except Exception as e:
        st.sidebar.error(f"Email 發送失敗: {e}")

# --- 抓取數據 ---
@st.cache_data(ttl=60)  # 每60秒更新一次
def get_data(symbol):
    ticker = yf.Ticker(symbol)
    # 抓取最近5天的5分鐘數據
    df = ticker.history(period="5d", interval="5m")
    df = df.dropna().copy()
    if df.empty:
        return None
    df.index = df.index.tz_convert('America/New_York')
    return df

with st.spinner(f"正在抓取 {stock_symbol} 5分鐘數據..."):
    data = get_data(stock_symbol.upper())

if data is None or data.empty:
    st.error(f"無法取得 {stock_symbol} 數據，請檢查代碼或稍後再試。")
    st.stop()

# 取最近 N 根K線
data = data.tail(lookback).copy()
data.reset_index(inplace=True)
data['time'] = data['Datetime'].dt.strftime('%H:%M')

# --- 計算指標 ---
def calculate_indicators(df):
    # RSI
    rsi = RSIIndicator(close=df['Close'], window=14).rsi()
    df['RSI'] = rsi
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_histogram'] = macd.macd_diff()
    
    return df

data = calculate_indicators(data)

# --- 計算支撐阻力 ---
def find_levels(df, tolerance_pct=0.005, min_touches=3):
    lows = df['Low'].values
    highs = df['High'].values
    closes = df['Close'].values

    prices = np.concatenate([lows, highs])
    sorted_prices = np.sort(prices)
    clusters = []
    current_cluster = [sorted_prices[0]]

    for p in sorted_prices[1:]:
        if p <= current_cluster[-1] * (1 + tolerance_pct):
            current_cluster.append(p)
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [p]
    if current_cluster:
        clusters.append(np.mean(current_cluster))

    levels = []
    for level in clusters:
        touches = 0
        # 低點觸及
        touches += sum(1 for l in lows if abs(l - level) <= level * tolerance_pct)
        # 高點觸及
        touches += sum(1 for h in highs if abs(h - level) <= level * tolerance_pct)
        # 收盤觸及
        touches += sum(1 for c in closes if abs(c - level) <= level * tolerance_pct)

        if touches >= min_touches:
            is_support = any(abs(l - level) <= level * tolerance_pct for l in lows)
            is_resistance = any(abs(h - level) <= level * tolerance_pct for h in highs)
            if is_support and is_resistance:
                typ = "S/R"
            elif is_support:
                typ = "Support"
            else:
                typ = "Resistance"
            levels.append({
                'price': round(level, 2),
                'touches': touches,
                'type': typ
            })

    supports = sorted([l for l in levels if "Support" in l['type']], key=lambda x: x['price'])
    resistances = sorted([l for l in levels if "Resistance" in l['type']], key=lambda x: x['price'], reverse=True)

    return supports[:3], resistances[:3], levels

supports, resistances, all_levels = find_levels(data, tolerance_pct, min_touches)

# --- 偵測突破並發送 Email ---
current_price = data['Close'].iloc[-1]
prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price

# 檢查阻力突破 (上漲)
for r in resistances:
    if prev_price < r['price'] <= current_price:
        st.warning(f"🚨 突破阻力 ${r['price']}！")
        send_breakout_email(stock_symbol, "上漲突破", current_price, r['price'], data['time'].iloc[-1])

# 檢查支撐突破 (下跌)
for s in supports:
    if prev_price > s['price'] >= current_price:
        st.error(f"🚨 跌破支撐 ${s['price']}！")
        send_breakout_email(stock_symbol, "下跌突破", current_price, s['price'], data['time'].iloc[-1])

# --- 繪製圖表 (多子圖) ---
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=('K線 + 成交量', 'RSI (14)', 'MACD', '支撐/阻力水平'),
    row_width=[0.6, 0.2, 0.1, 0.1]
)

# Row 1: K線
fig.add_trace(go.Candlestick(
    x=data['time'], open=data['Open'], high=data['High'],
    low=data['Low'], close=data['Close'], name="K線",
    row=1, col=1
), row=1, col=1)

# 成交量 (Row 1 次級 y 軸)
colors = ['green' if o < c else 'red' for o, c in zip(data['Open'], data['Close'])]
fig.add_trace(go.Bar(x=data['time'], y=data['Volume'], name="成交量",
                     marker_color=colors, showlegend=True,
                     yaxis="y2"), row=1, col=1)

# Row 2: RSI
fig.add_trace(go.Scatter(x=data['time'], y=data['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)  # 超買
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)  # 超賣

# Row 3: MACD
fig.add_trace(go.Scatter(x=data['time'], y=data['MACD'], name="MACD", line=dict(color='blue')), row=3, col=1)
fig.add_trace(go.Scatter(x=data['time'], y=data['MACD_signal'], name="Signal", line=dict(color='orange')), row=3, col=1)
fig.add_trace(go.Bar(x=data['time'], y=data['MACD_histogram'], name="Histogram", marker_color='gray'), row=3, col=1)

# Row 4: 支撐/阻力 (水平線)
for s in supports:
    fig.add_hline(y=s['price'], line_dash="dash", line_color="green", 
                  annotation_text=f"S: ${s['price']}", row=4, col=1)
for r in resistances:
    fig.add_hline(y=r['price'], line_dash="dash", line_color="red", 
                  annotation_text=f"R: ${r['price']}", row=4, col=1)

# 更新布局
fig.update_layout(
    title=f"{stock_symbol} 5分鐘圖表（最近 {lookback} 根） - 包含成交量、RSI、MACD",
    yaxis_title="價格 (USD)",
    yaxis2_title="成交量",  # 次級 y 軸
    height=800,
    template="plotly_dark",
    xaxis_rangeslider_visible=False
)
fig.update_xaxes(title_text="時間", row=4, col=1)
fig.update_yaxes(title_text="RSI", row=2, col=1)
fig.update_yaxes(title_text="MACD", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# --- 顯示表格 ---
st.subheader("關鍵支撐與阻力水平")
level_df = pd.DataFrame(all_levels)
if not level_df.empty:
    level_df = level_df.sort_values("price", ascending=False)
    st.table(level_df.style.format({"price": "${:.2f}"}))
else:
    st.info("未偵測到符合條件的支撐/阻力線（請調整參數）")

# --- 最新價格與指標摘要 ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("最新價格", f"${current_price:.2f}")
with col2:
    st.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}", 
              delta_color="inverse" if data['RSI'].iloc[-1] > 70 else "normal")
with col3:
    st.metric("MACD", f"{data['MACD'].iloc[-1]:.2f}")
with col4:
    st.metric("成交量", f"{data['Volume'].iloc[-1]:,.0f}")

# --- 交易提示 ---
st.subheader("交易提示")
if resistances:
    next_r = min([r['price'] for r in resistances if r['price'] > current_price], default=None)
    if next_r:
        st.success(f"向上阻力：**${next_r}**")

if supports:
    next_s = max([s['price'] for s in supports if s['price'] < current_price], default=None)
    if next_s:
        st.error(f"向下支撐：**${next_s}**")

rsi_val = data['RSI'].iloc[-1]
if rsi_val > 70:
    st.warning("RSI 超買 (>70)，考慮賣出")
elif rsi_val < 30:
    st.success("RSI 超賣 (<30)，考慮買入")

# --- 資料來源 ---
st.caption(f"數據來源：Yahoo Finance | 更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- 重新整理按鈕 ---
if st.button("刷新數據"):
    st.cache_data.clear()
    st.experimental_rerun()

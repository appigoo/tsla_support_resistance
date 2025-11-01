# stock_analysis_multi.py
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
import time

# ==================== 頁面設定 ====================
st.set_page_config(page_title="多股即時分析", layout="wide")
st.title("多股即時分析 - 支援切換 + 自動刷新 + 倒數計時")

# ==================== 側邊欄參數 ====================
with st.sidebar:
    st.header("多股設定")
    symbols_input = st.text_input("股票代碼（逗號分隔）", "TSLA, AAPL, NVDA")
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    # period & interval
    period_options = {"1 天": "1d", "5 天": "5d", "1 個月": "1mo", "3 個月": "3mo", "6 個月": "6mo", "1 年": "1y"}
    period_display = st.selectbox("時間範圍", options=list(period_options.keys()), index=1)
    period = period_options[period_display]

    interval_options = {"1 分鐘": "1m", "5 分鐘": "5m", "15 分鐘": "15m", "30 分鐘": "30m", "60 分鐘": "60m", "1 小時": "1h"}
    interval_display = st.selectbox("K線間隔", options=list(interval_options.keys()), index=1)
    interval = interval_options[interval_display]

    lookback = st.slider("回看K線數", 20, 500, 50)
    tolerance_pct = st.slider("價格容忍 (%)", 0.1, 2.0, 0.5) / 100
    min_touches = st.slider("最少觸及次數", 2, 6, 3)
    enable_email = st.checkbox("啟用 Email 突破警示")

    # 切換間隔
    st.header("切換與刷新")
    switch_options = {"關閉": None, "10 秒": 10, "30 秒": 30, "1 分鐘": 60, "2 分鐘": 120}
    switch_display = st.selectbox("切換間隔", options=list(switch_options.keys()), index=2)
    switch_interval = switch_options[switch_display]

    refresh_options = {"關閉": None, "30 秒": 30, "1 分鐘": 60, "2 分鐘": 120, "5 分鐘": 300}
    refresh_display = st.selectbox("自動刷新", options=list(refresh_options.keys()), index=0)
    refresh_interval = refresh_options[refresh_display]

# ==================== 狀態管理 ====================
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "last_switch" not in st.session_state:
    st.session_state.last_switch = time.time()

# ==================== 倒數計時器 ====================
countdown_placeholder = st.empty()
switch_countdown = st.empty()

if switch_interval and len(symbols) > 1:
    elapsed = time.time() - st.session_state.last_switch
    remaining = max(0, switch_interval - int(elapsed))
    switch_countdown.info(f"切換至下一支：{remaining} 秒")
    if remaining <= 0:
        st.session_state.current_idx = (st.session_state.current_idx + 1) % len(symbols)
        st.session_state.last_switch = time.time()
        st.rerun()

if refresh_interval:
    for i in range(refresh_interval, 0, -1):
        countdown_placeholder.info(f"資料刷新：{i} 秒")
        time.sleep(1)
    st.cache_data.clear()
    st.rerun()

# ==================== 選擇當前股票 ====================
if not symbols:
    st.error("請輸入至少一支股票代碼")
    st.stop()

current_symbol = symbols[st.session_state.current_idx]
st.header(f"目前顯示：{current_symbol}（{st.session_state.current_idx + 1}/{len(symbols)}）")

# ==================== Email ====================
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
        st.sidebar.success(f"{symbol} Email 已發送")
    except Exception as e:
        st.sidebar.error(f"{symbol} Email 失敗")

# ==================== 抓資料 ====================
@st.cache_data(ttl=60)
def fetch_data(symbol, period, interval):
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty: return None
        df = df.dropna().copy()
        df.index = df.index.tz_convert('America/New_York')
        return df
    except: return None

with st.spinner(f"抓取 {current_symbol} 資料..."):
    data = fetch_data(current_symbol, period, interval)

if data is None or data.empty:
    st.error(f"{current_symbol} 無法取得資料")
    st.stop()

data = data.tail(lookback).copy()
data.reset_index(inplace=True)
data['time'] = data['Datetime'].dt.strftime('%H:%M')

# ==================== RSI / MACD ====================
def calc_rsi(s, p=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(p).mean()
    down = -d.clip(upper=0).rolling(p).mean()
    return 100 - (100 / (1 + up / down))

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

supports, resists, _ = find_levels(data, tolerance_pct, min_touches)

# 補足 R1/R2, S1/S2
if not resists:
    resists = [{"price": round(data['High'].max(),2), "touches": 1, "type": "Resistance"}]
while len(resists) < 2:
    second = data['High'].nlargest(2).iloc[-1] if len(data['High'].nlargest(2)) > 1 else resists[-1]["price"]*0.98
    resists.append({"price": round(second,2), "touches": 1, "type": "Resistance"})

if not supports:
    supports = [{"price": round(data['Low'].min(),2), "touches": 1, "type": "Support"}]
while len(supports) < 2:
    second = data['Low'].nsmallest(2).iloc[-1] if len(data['Low'].nsmallest(2)) > 1 else supports[-1]["price"]*1.02
    supports.append({"price": round(second,2), "touches": 1, "type": "Support"})

# ==================== 關鍵水平 ====================
st.subheader("當前關鍵水平")
level_rows = []
for i, r in enumerate(resists):
    level_rows.append({"類型": f"阻力 R{i+1}", "價格": f"${r['price']}", "觸及次數": f"{r['touches']} 次", "說明": "短期賣壓強" if i == 0 else "中期天花板"})
for i, s in enumerate(supports):
    level_rows.append({"類型": f"支撐 S{i+1}", "價格": f"${s['price']}", "觸及次數": f"{s['touches']} 次", "說明": "強力支撐" if i == 0 else "中間支撐"})
st.table(pd.DataFrame(level_rows))

# ==================== 交易建議 ====================
cur_price = data['Close'].iloc[-1]
prev_price = data['Close'].iloc[-2] if len(data) > 1 else cur_price

for r in resists:
    if prev_price < r['price'] <= cur_price:
        st.warning(f"突破阻力 ${r['price']}！")
        send_email(current_symbol, "上漲突破", cur_price, r['price'])
for s in supports:
    if prev_price > s['price'] >= cur_price:
        st.error(f"跌破支撐 ${s['price']}！")
        send_email(current_symbol, "下跌突破", cur_price, s['price'])

st.subheader("交易建議（示意）")
r1_price = resists[0]["price"]
r2_price = resists[1]["price"]
s1_price = supports[0]["price"]

if cur_price > r1_price:
    suggestion = f"**突破 {r1_price} 並站穩** → 看多至 {r2_price}。"
elif cur_price < s1_price:
    suggestion = f"**跌破 {s1_price}** → 可能下探 {s1_price-5 if s1_price > 10 else s1_price*0.95:.2f} 或更低。"
elif s1_price <= cur_price <= r1_price:
    suggestion = f"目前價格在 **S1 ~ R1 之間震盪**，適合區間操作。"
else:
    suggestion = "觀察中，等待明確訊號。"
st.markdown(suggestion)

# ==================== 圖表 ====================
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    subplot_titles=(f"K線 + 成交量", "RSI", "MACD", "支撐/阻力"),
                    row_heights=[0.5,0.2,0.2,0.1])

colors = ['green' if o < c else 'red' for o, c in zip(data['Open'], data['Close'])]
fig.add_trace(go.Candlestick(x=data['time'], open=data['Open'], high=data['High'],
                             low=data['Low'], close=data['Close'], name="K線"), row=1, col=1)
fig.add_trace(go.Bar(x=data['time'], y=data['Volume'], name="成交量", marker_color=colors), row=1, col=1)

fig.add_trace(go.Scatter(x=data['time'], y=data['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.add_trace(go.Scatter(x=data['time'], y=data['MACD'], name="MACD", line=dict(color='blue')), row=3, col=1)
fig.add_trace(go.Scatter(x=data['time'], y=data['MACD_signal'], name="Signal", line=dict(color='orange')), row=3, col=1)
fig.add_trace(go.Bar(x=data['time'], y=data['MACD_hist'], name="Hist", marker_color='gray'), row=3, col=1)

for s in supports:
    fig.add_hline(y=s['price'], line_color="green", line_dash="dash", annotation_text=f"S{supports.index(s)+1}", row=4, col=1)
for r in resists:
    fig.add_hline(y=r['price'], line_color="red", line_dash="dash", annotation_text=f"R{resists.index(r)+1}", row=4, col=1)

fig.update_layout(height=900, title=f"{current_symbol} {interval_display} 圖表", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ==================== 指標 ====================
c1, c2, c3, c4 = st.columns(4)
c1.metric("最新價格", f"${cur_price:.2f}")
c2.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
c3.metric("MACD", f"{data['MACD'].iloc[-1]:.3f}")
c4.metric("成交量", f"{data['Volume'].iloc[-1]:,.0f}")

st.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if st.button("立即刷新"):
    st.cache_data.clear()
    st.rerun()

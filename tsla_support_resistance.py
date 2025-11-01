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
import time

# ==================== 頁面設定 ====================
st.set_page_config(page_title="股票 5分鐘分析", layout="wide")
st.title("股票即時分析 - 可設定時間範圍與自動刷新")

# ==================== 側邊欄參數 ====================
with st.sidebar:
    st.header("資料設定")
    stock_symbol = st.text_input("股票代碼", "TSLA").upper()

    # period 選項
    period_options = {
        "1 天": "1d",
        "5 天": "5d",
        "1 個月": "1mo",
        "3 個月": "3mo",
        "6 個月": "6mo",
        "1 年": "1y"
    }
    period_display = st.selectbox("時間範圍", options=list(period_options.keys()), index=1)
    period = period_options[period_display]

    # interval 選項
    interval_options = {
        "1 分鐘": "1m",
        "5 分鐘": "5m",
        "15 分鐘": "15m",
        "30 分鐘": "30m",
        "60 分鐘": "60m",
        "1 小時": "1h"
    }
    interval_display = st.selectbox("K線間隔", options=list(interval_options.keys()), index=1)
    interval = interval_options[interval_display]

    # 回看根數
    lookback = st.slider("回看K線數", 20, 500, 50)

    # 支撐阻力參數
    tolerance_pct = st.slider("價格容忍 (%)", 0.1, 2.0, 0.5) / 100
    min_touches = st.slider("最少觸及次數", 2, 6, 3)

    # Email 警示
    enable_email = st.checkbox("啟用 Email 突破警示")

    # 自動刷新
    st.header("自動刷新")
    refresh_options = {
        "關閉": None,
        "30 秒": 30,
        "1 分鐘": 60,
        "2 分鐘": 120,
        "5 分鐘": 300
    }
    refresh_display = st.selectbox("自動刷新間隔", options=list(refresh_options.keys()), index=0)
    refresh_interval = refresh_options[refresh_display]

    if refresh_interval:
        st.info(f"每 {refresh_display} 自動刷新")
        time.sleep(refresh_interval)
        st.rerun()

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
        st.sidebar.success("Email 已發送")
    except Exception as e:
        st.sidebar.error(f"Email 失敗: {e}")

# ==================== 抓資料 ====================
@st.cache_data(ttl=60)
def fetch_data(symbol, period, interval):
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty: return None
        df = df.dropna().copy()
        df.index = df.index.tz_convert('America/New_York')
        return df
    except Exception as e:
        st.error(f"抓取資料失敗: {e}")
        return None

with st.spinner(f"正在抓取 {stock_symbol} {interval_display} 資料（{period_display}）..."):
    data = fetch_data(stock_symbol, period, interval)

if data is None or data.empty:
    st.error("無法取得資料，請檢查代碼或網路")
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
        if p <= cur[-1] * (1 + tol):
            cur.append(p)
        else:
            clusters.append(np.mean(cur))
            cur = [p]
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

# ---- 補足 R1/R2, S1/S2 ----
if not resists:
    r1 = data['High'].max()
    resists = [{"price": round(r1,2), "touches": 1, "type": "Resistance"}]
else:
    resists = resists[:2]
while len(resists) < 2:
    second = data['High'].nlargest(2).iloc[-1] if len(data['High'].nlargest(2)) > 1 else resists[-1]["price"]*0.98
    resists.append({"price": round(second,2), "touches": 1, "type": "Resistance"})

if not supports:
    s1 = data['Low'].min()
    supports = [{"price": round(s1,2), "touches": 1, "type": "Support"}]
else:
    supports = supports[:2]
while len(supports) < 2:
    second = data['Low'].nsmallest(2).iloc[-1] if len(data['Low'].nsmallest(2)) > 1 else supports[-1]["price"]*1.02
    supports.append({"price": round(second,2), "touches": 1, "type": "Support"})

# ==================== 關鍵水平表格 ====================
st.subheader("當前關鍵水平")
level_rows = []
for i, r in enumerate(resists):
    level_rows.append({
        "類型": f"阻力 R{i+1}",
        "價格": f"${r['price']}",
        "觸及次數": f"{r['touches']} 次",
        "說明": "短期賣壓強" if i == 0 else "中期天花板"
    })
for i, s in enumerate(supports):
    level_rows.append({
        "類型": f"支撐 S{i+1}",
        "價格": f"${s['price']}",
        "觸及次數": f"{s['touches']} 次",
        "說明": "強力支撐" if i == 0 else "中間支撐"
    })

df_levels = pd.DataFrame(level_rows)
st.table(df_levels)

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
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
    subplot_titles=(f"K線 + 成交量 ({interval_display})", "RSI (14)", "MACD (12,26,9)", "支撐/阻力"),
    row_heights=[0.5,0.2,0.2,0.1]
)

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
    fig.add_hline(y=s['price'], line_color="green", line_dash="dash",
                  annotation_text=f"S{supports.index(s)+1} ${s['price']}", row=4, col=1)
for r in resists:
    fig.add_hline(y=r['price'], line_color="red", line_dash="dash",
                  annotation_text=f"R{resists.index(r)+1} ${r['price']}", row=4, col=1)

fig.update_layout(height=900, title=f"{stock_symbol} {interval_display} 圖表（{period_display}）", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ==================== 指標 ====================
c1, c2, c3, c4 = st.columns(4)
c1.metric("最新價格", f"${cur_price:.2f}")
c2.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
c3.metric("MACD", f"{data['MACD'].iloc[-1]:.3f}")
c4.metric("成交量", f"{data['Volume'].iloc[-1]:,.0f}")

st.caption(f"數據來源：Yahoo Finance | 更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 手動刷新按鈕
if st.button("立即刷新"):
    st.cache_data.clear()
    st.rerun()

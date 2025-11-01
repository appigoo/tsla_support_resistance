# tsla_support_resistance.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

st.set_page_config(page_title="TSLA 5分鐘 支撐阻力分析", layout="wide")

st.title("TSLA 5分鐘K線 + 自動支撐/阻力線")

# --- 參數設定 ---
lookback = st.sidebar.slider("回看K線數", min_value=20, max_value=200, value=50, step=10)
tolerance_pct = st.sidebar.slider("價格容忍區間 (%)", 0.1, 2.0, 0.5, 0.1) / 100
min_touches = st.sidebar.slider("最少觸及次數", 2, 6, 3)

# --- 抓取數據 ---
@st.cache_data(ttl=60)  # 每60秒更新一次
def get_data():
    ticker = yf.Ticker("TSLA")
    # 抓取最近2天的5分鐘數據（包含盤前盤後）
    df = ticker.history(period="5d", interval="5m")
    df = df.dropna().copy()
    df.index = df.index.tz_convert('America/New_York')  # 轉為美東時間
    return df

with st.spinner("正在抓取 TSLA 5分鐘數據..."):
    data = get_data()

if data.empty:
    st.error("無法取得數據，請檢查網路或稍後再試。")
    st.stop()

# 取最近 N 根K線
data = data.tail(lookback).copy()
data.reset_index(inplace=True)
data['time'] = data['Datetime'].dt.strftime('%H:%M')

# --- 計算支撐阻力 ---
def find_levels(df, tolerance_pct=0.005, min_touches=3):
    lows = df['Low'].values
    highs = df['High'].values
    closes = df['Close'].values

    levels = []
    prices = np.concatenate([lows, highs])

    # 合併相近價格
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

    # 計算觸及次數
    support_levels = []
    resistance_levels = []

    for level in clusters:
        touches = 0
        # 支撐：低點接近 level
        if min(abs(l - level) / level for l in lows) <= tolerance_pct:
            touches += sum(1 for l in lows if abs(l - level) <= level * tolerance_pct)
        # 阻力：高點接近 level
        if min(abs(h - level) / level for h in highs) <= tolerance_pct:
            touches += sum(1 for h in highs if abs(h - level) <= level * tolerance_pct)
        # 收盤價反彈也算
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

    # 分類
    supports = sorted([l for l in levels if "Support" in l['type']], key=lambda x: x['price'])
    resistances = sorted([l for l in levels if "Resistance" in l['type']], key=lambda x: x['price'], reverse=True)

    return supports[:3], resistances[:3], levels

supports, resistances, all_levels = find_levels(data, tolerance_pct, min_touches)

# --- 繪製蠟燭圖 ---
fig = go.Figure()

# 蠟燭
fig.add_trace(go.Candlestick(
    x=data['time'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="TSLA"
))

# 支撐線
for i, s in enumerate(supports):
    fig.add_shape(type="line",
                  x0=data['time'].iloc[0], y0=s['price'],
                  x1=data['time'].iloc[-1], y1=s['price'],
                  line=dict(color="green", width=2, dash="dash"),
                  name=f"S{i+1}"
                  )
    fig.add_annotation(x=data['time'].iloc[-1], y=s['price'],
                       text=f"S{i+1}: ${s['price']} ({s['touches']}次)",
                       showarrow=True, arrowhead=2, arrowsize=1,
                       arrowcolor="green", font=dict(color="green"),
                       xanchor="left", yanchor="middle", xshift=5)

# 阻力線
for i, r in enumerate(resistances):
    fig.add_shape(type="line",
                  x0=data['time'].iloc[0], y0=r['price'],
                  x1=data['time'].iloc[-1], y1=r['price'],
                  line=dict(color="red", width=2, dash="dash"),
                  name=f"R{i+1}"
                  )
    fig.add_annotation(x=data['time'].iloc[-1], y=r['price'],
                       text=f"R{i+1}: ${r['price']} ({r['touches']}次)",
                       showarrow=True, arrowhead=2, arrowsize=1,
                       arrowcolor="red", font=dict(color="red"),
                       xanchor="left", yanchor="middle", xshift=5)

fig.update_layout(
    title=f"TSLA 5分鐘K線圖（最近 {lookback} 根）",
    xaxis_title="時間",
    yaxis_title="價格 (USD)",
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# --- 顯示表格 ---
st.subheader("關鍵支撐與阻力水平")
level_df = pd.DataFrame(all_levels)
if not level_df.empty:
    level_df = level_df.sort_values("price", ascending=False)
    st.table(level_df.style.format({"price": "${:.2f}"}))
else:
    st.info("未偵測到符合條件的支撐/阻力線（請調整參數）")

# --- 交易提示 ---
current_price = data['Close'].iloc[-1]
st.markdown(f"**最新價格：${current_price:.2f}**")

if resistances:
    next_r = min([r['price'] for r in resistances if r['price'] > current_price], default=None)
    if next_r:
        st.success(f"向上阻力：**${next_r}**")

if supports:
    next_s = max([s['price'] for s in supports if s['price'] < current_price], default=None)
    if next_s:
        st.error(f"向下支撐：**${next_s}**")

# --- 資料來源 ---
st.caption(f"數據來源：Yahoo Finance | 更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- 重新整理按鈕 ---
if st.button("刷新數據"):
    st.experimental_rerun()

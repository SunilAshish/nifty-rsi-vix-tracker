import math
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from nifty_signal import position_advice, run_dashboard

st.set_page_config(page_title="Nifty Signal App", page_icon="📈", layout="centered")

st.markdown(
    """
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 760px;}
        .signal-card {
            border-radius: 16px;
            padding: 14px 16px;
            color: white;
            font-weight: 700;
            text-align: center;
            margin-bottom: 12px;
            font-size: 1.2rem;
        }
        .buy {background: #0f9d58;}
        .sell {background: #db4437;}
        .hold {background: #5f6368;}
        .small-note {font-size: 0.9rem; color: #4b5563;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Nifty Signal Dashboard")

# Auto-refresh every 60 seconds for near real-time behavior.
st_autorefresh(interval=60_000, key="market_live_refresh")


def render_timeseries(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for col in df.columns:
        ax.plot(df.index, df[col], linewidth=2, label=col)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


lookback_days = st.slider("Time-series Lookback (days)", min_value=30, max_value=365, value=120, step=10)
capital = st.number_input("Planned Capital (INR)", min_value=10000, value=500000, step=10000)

if st.button("Refresh Signal", use_container_width=True):
    st.cache_data.clear()


@st.cache_data(ttl=60)
def get_data(days: int):
    return run_dashboard(lookback_days=days)


try:
    result, warning, nifty_ts, vix_ts = get_data(lookback_days)
except Exception as exc:
    st.error(f"Could not fetch market data: {exc}")
    st.stop()

if warning:
    st.warning(warning)

signal_class = "hold"
if result.signal == "BUY":
    signal_class = "buy"
elif result.signal == "SELL":
    signal_class = "sell"

st.markdown(f"<div class='signal-card {signal_class}'>{result.signal} | {result.strength}</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.metric("Date", result.date)
    st.metric("Nifty Close", f"{result.close:,.2f}")
    st.metric("1D Change", f"{result.pct_change_1d:.2f}%")
with c2:
    rsi_text = "N/A" if math.isnan(result.rsi_14) else f"{result.rsi_14:.2f}"
    ma20_text = "N/A" if math.isnan(result.ma_20) else f"{result.ma_20:,.2f}"
    ma50_text = "N/A" if math.isnan(result.ma_50) else f"{result.ma_50:,.2f}"

    st.metric("RSI(14)", rsi_text)
    st.metric("MA20", ma20_text)
    st.metric("MA50", ma50_text)

vix_text = "Not available" if result.vix is None else f"{result.vix:.2f}"
st.write(f"**Trend:** {result.trend}")
st.write(f"**India VIX:** {vix_text}")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.subheader("Time-Series")
st.caption("Historical closing values used for the current signal.")

st.write("**Nifty 50 Close**")
render_timeseries(nifty_ts, "Nifty 50 Close")

if vix_ts is not None:
    st.write("**India VIX Close**")
    render_timeseries(vix_ts, "India VIX Close")
else:
    st.info("India VIX time-series is currently unavailable.")

st.write("**Growth Comparison (Base = 100)**")
growth_df = pd.DataFrame(index=nifty_ts.index)
growth_df["Nifty Growth"] = (nifty_ts["nifty_close"] / nifty_ts["nifty_close"].iloc[0]) * 100

if vix_ts is not None:
    aligned = nifty_ts.join(vix_ts, how="inner")
    if not aligned.empty:
        growth_df = pd.DataFrame(index=aligned.index)
        growth_df["Nifty Growth"] = (aligned["nifty_close"] / aligned["nifty_close"].iloc[0]) * 100
        growth_df["VIX Growth"] = (aligned["india_vix"] / aligned["india_vix"].iloc[0]) * 100

render_timeseries(growth_df, "Growth Comparison (Base = 100)")

st.subheader("Reasons")
for i, reason in enumerate(result.reasons, start=1):
    st.write(f"{i}. {reason}")

st.subheader("Position Advice")
st.success(position_advice(result.signal, result.strength, total_capital=float(capital)))

st.caption("Educational tool only, not financial advice.")
st.markdown("<div class='small-note'>Tip: Open this app in your phone browser and add to home screen for app-like use.</div>", unsafe_allow_html=True)

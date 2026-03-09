import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional


NIFTY_SYMBOL = "^NSEI"
VIX_SYMBOL = "^INDIAVIX"
LOOKBACK_DAYS = 120

BIG_DROP_BUY_THRESHOLD = -1.5
SECOND_DROP_BUY_THRESHOLD = -1.0
BOUNCE_SELL_THRESHOLD = 1.0

RSI_BUY_LEVEL = 35
RSI_SELL_LEVEL = 65

VIX_HIGH = 18
VIX_VERY_HIGH = 22


@dataclass
class SignalResult:
    date: str
    close: float
    pct_change_1d: float
    rsi_14: float
    ma_20: float
    ma_50: float
    trend: str
    vix: Optional[float]
    signal: str
    strength: str
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    return df.rename(columns=str.lower)


def generate_signal(nifty_df: pd.DataFrame, vix_value: Optional[float] = None) -> SignalResult:
    df = nifty_df.copy()

    df["pct_change_1d"] = df["close"].pct_change() * 100
    df["rsi_14"] = compute_rsi(df["close"], period=14)
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    close_price = float(latest["close"])
    pct_1d = float(latest["pct_change_1d"])
    rsi_14 = float(latest["rsi_14"]) if pd.notna(latest["rsi_14"]) else np.nan
    ma_20 = float(latest["ma_20"]) if pd.notna(latest["ma_20"]) else np.nan
    ma_50 = float(latest["ma_50"]) if pd.notna(latest["ma_50"]) else np.nan

    signal = "HOLD"
    strength = "NEUTRAL"
    reasons: list[str] = []

    if pct_1d <= BIG_DROP_BUY_THRESHOLD:
        reasons.append(f"Nifty fell {pct_1d:.2f}% today (big drop).")

        if pd.notna(rsi_14) and rsi_14 <= RSI_BUY_LEVEL:
            reasons.append(f"RSI is oversold at {rsi_14:.2f}.")
            signal = "BUY"
            strength = "MEDIUM"

            if vix_value is not None and vix_value >= VIX_HIGH:
                reasons.append(f"India VIX is elevated at {vix_value:.2f}, suggesting panic/volatility.")
                strength = "STRONG"

    if prev is not None:
        prev_pct = float(prev["pct_change_1d"]) if pd.notna(prev["pct_change_1d"]) else 0
        if prev_pct <= SECOND_DROP_BUY_THRESHOLD and pct_1d <= SECOND_DROP_BUY_THRESHOLD:
            reasons.append("Back-to-back negative sessions detected.")
            if signal == "BUY":
                strength = "STRONG"
            else:
                signal = "BUY"
                strength = "MEDIUM"

    if pct_1d >= BOUNCE_SELL_THRESHOLD:
        if pd.notna(rsi_14) and rsi_14 >= RSI_SELL_LEVEL:
            reasons.append(f"Nifty bounced {pct_1d:.2f}% today and RSI is high at {rsi_14:.2f}.")
            signal = "SELL"
            strength = "MEDIUM"

    if pd.notna(ma_20) and close_price > ma_20 * 1.02 and pd.notna(rsi_14) and rsi_14 >= 68:
        reasons.append("Price is stretched above 20-day MA with high RSI.")
        signal = "SELL"
        strength = "STRONG"

    if vix_value is not None and vix_value >= VIX_VERY_HIGH and signal == "BUY":
        reasons.append("VIX is very high, so buy only in small tranches.")
        strength = "CAUTIOUS BUY"

    trend = "SIDEWAYS"
    if pd.notna(ma_20) and pd.notna(ma_50):
        if ma_20 > ma_50:
            trend = "UPTREND"
        elif ma_20 < ma_50:
            trend = "DOWNTREND"

    if signal == "BUY" and trend == "DOWNTREND":
        reasons.append("Trend is still down, so do staggered buying, not full capital.")
    if signal == "SELL" and trend == "UPTREND":
        reasons.append("Trend is up, so this may be partial profit booking rather than full exit.")

    if not reasons:
        reasons.append("No strong setup based on drop %, RSI, and VIX rules.")

    return SignalResult(
        date=str(df.index[-1].date()),
        close=close_price,
        pct_change_1d=pct_1d,
        rsi_14=rsi_14,
        ma_20=ma_20,
        ma_50=ma_50,
        trend=trend,
        vix=vix_value,
        signal=signal,
        strength=strength,
        reasons=reasons,
    )


def position_advice(signal: str, strength: str, total_capital: float = 100000) -> str:
    if signal == "BUY":
        if strength == "STRONG":
            return f"Buy 30% of planned capital now: Rs {0.30 * total_capital:,.0f}"
        if strength == "MEDIUM":
            return f"Buy 20% of planned capital now: Rs {0.20 * total_capital:,.0f}"
        if strength == "CAUTIOUS BUY":
            return (
                "Buy only 10-15% now due to high VIX: "
                f"Rs {0.10 * total_capital:,.0f} to Rs {0.15 * total_capital:,.0f}"
            )
        return "Small trial buy only."

    if signal == "SELL":
        if strength == "STRONG":
            return "Sell 40-50% of current holdings / book profit."
        return "Sell 20-30% of current holdings / partial booking."

    return "No action. Wait for a better setup."


def run_dashboard(
    lookback_days: int = LOOKBACK_DAYS,
) -> tuple[SignalResult, Optional[str], pd.DataFrame, Optional[pd.DataFrame]]:
    end_date = datetime.today().date() + timedelta(days=1)
    start_date = datetime.today().date() - timedelta(days=lookback_days)

    nifty_df = fetch_data(NIFTY_SYMBOL, str(start_date), str(end_date))

    nifty_ts = nifty_df[["close"]].rename(columns={"close": "nifty_close"})

    vix_value: Optional[float] = None
    vix_ts: Optional[pd.DataFrame] = None
    warning: Optional[str] = None

    try:
        vix_df = fetch_data(VIX_SYMBOL, str(start_date), str(end_date))
        vix_value = float(vix_df["close"].iloc[-1])
        vix_ts = vix_df[["close"]].rename(columns={"close": "india_vix"})
    except Exception as exc:
        warning = f"Could not fetch India VIX. Reason: {exc}"

    result = generate_signal(nifty_df, vix_value=vix_value)
    return result, warning, nifty_ts, vix_ts


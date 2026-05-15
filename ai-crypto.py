#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║         🚀 CRYPTO TRADING AI — SIGNAL GENERATOR v2.0           ║
║  ✅ Vote-based signals (no more perpetual HOLD)                 ║
║  ✅ Stochastic + EMA + RSI + MACD + BB + Volume                 ║
║  ✅ Multi-pair scanner                                           ║
║  ✅ Auto-refresh + clean Streamlit UI                           ║
╚══════════════════════════════════════════════════════════════════╝

Run: streamlit run crypto_signals.py
"""

# ── API KEYS ──────────────────────────────────────────────────────────────────
NVIDIA_API_KEY   = ""   # https://build.nvidia.com  (optional – AI commentary)
BINANCE_API_KEY  = ""   # Optional – only needed for authenticated endpoints
BINANCE_API_SECRET = ""

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import sys, time, logging, traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import ccxt
import ta

# ── CONFIG ────────────────────────────────────────────────────────────────────
class Config:
    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    NVIDIA_MODEL    = "meta/llama-3.1-70b-instruct"

    PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
    TIMEFRAMES: Dict[str, str] = {
        "5m":  "5m",
        "15m": "15m",
        "1h":  "1h",
        "4h":  "4h",
        "1D":  "1d",
    }

    # Signal fires when bull_score OR bear_score reaches this threshold
    SIGNAL_THRESHOLD = 5   # out of ~20 possible weighted points
    # Auto-refresh interval (seconds)
    AUTO_REFRESH_SEC = 60

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("CryptoAI")

# ─────────────────────────────────────────────────────────────────────────────
# 📡  MARKET DATA
# ─────────────────────────────────────────────────────────────────────────────
class MarketData:
    _exchange: Optional[ccxt.Exchange] = None

    @classmethod
    def _get_exchange(cls) -> ccxt.Exchange:
        if cls._exchange is None:
            cls._exchange = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
                **({"apiKey": BINANCE_API_KEY, "secret": BINANCE_API_SECRET}
                   if BINANCE_API_KEY else {}),
            })
        return cls._exchange

    @classmethod
    def fetch(cls, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        try:
            ex = cls._get_exchange()
            raw = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            return df
        except Exception as e:
            log.error(f"fetch_ohlcv failed for {symbol}/{timeframe}: {e}")
            return pd.DataFrame()

    @classmethod
    def price(cls, symbol: str) -> float:
        try:
            return cls._get_exchange().fetch_ticker(symbol)["last"]
        except Exception:
            return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 📊  INDICATOR ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class Indicators:
    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["close"]

        # Trend
        df["ema9"]  = ta.trend.ema_indicator(c, window=9)
        df["ema20"] = ta.trend.ema_indicator(c, window=20)
        df["ema50"] = ta.trend.ema_indicator(c, window=50)
        df["ema200"]= ta.trend.ema_indicator(c, window=200)

        # Momentum
        df["rsi"]   = ta.momentum.rsi(c, window=14)
        stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], c, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # MACD
        macd = ta.trend.MACD(c, window_fast=12, window_slow=26, window_sign=9)
        df["macd"]      = macd.macd()
        df["macd_sig"]  = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # Bollinger
        bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
        df["bb_upper"]  = bb.bollinger_hband()
        df["bb_mid"]    = bb.bollinger_mavg()
        df["bb_lower"]  = bb.bollinger_lband()
        df["bb_pct"]    = bb.bollinger_pband()   # 0 = lower, 1 = upper

        # ATR & Volume
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], c, window=14)
        df["vol_sma20"] = df["volume"].rolling(20).mean()

        return df

# ─────────────────────────────────────────────────────────────────────────────
# 🎯  SIGNAL GENERATOR  (vote-based — no more cancel-out HOLD)
# ─────────────────────────────────────────────────────────────────────────────
class SignalEngine:
    """
    Each indicator casts a weighted BULL or BEAR vote.
    Signals are kept SEPARATE and never subtracted from each other.
    Final decision = whichever side wins AND crosses SIGNAL_THRESHOLD.
    """

    @staticmethod
    def generate(df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < 60:
            return _empty_signal("Insufficient data (need 60+ candles)")

        df = df.dropna()
        if len(df) < 3:
            return _empty_signal("Not enough valid rows after dropna")

        now  = df.iloc[-1]
        prev = df.iloc[-2]
        ago2 = df.iloc[-3]

        bull: List[Tuple[str, float]] = []
        bear: List[Tuple[str, float]] = []

        # ── 1. EMA STACK ──────────────────────────────────────────────────────
        if now["ema9"] > now["ema20"] > now["ema50"]:
            bull.append(("EMA stack aligned bullish (9>20>50)", 3))
        elif now["ema9"] < now["ema20"] < now["ema50"]:
            bear.append(("EMA stack aligned bearish (9<20<50)", 3))

        # ── 2. EMA 20/50 CROSSOVER ────────────────────────────────────────────
        if now["ema20"] > now["ema50"] and prev["ema20"] <= prev["ema50"]:
            bull.append(("Golden Cross: EMA20 crossed above EMA50", 5))
        elif now["ema20"] < now["ema50"] and prev["ema20"] >= prev["ema50"]:
            bear.append(("Death Cross: EMA20 crossed below EMA50", 5))

        # ── 3. PRICE vs EMA20 ─────────────────────────────────────────────────
        if now["close"] > now["ema20"]:
            bull.append(("Price above EMA20", 1))
        else:
            bear.append(("Price below EMA20", 1))

        # ── 4. PRICE vs EMA200 (major bias) ──────────────────────────────────
        if now["close"] > now["ema200"]:
            bull.append(("Price above EMA200 (long-term uptrend)", 2))
        else:
            bear.append(("Price below EMA200 (long-term downtrend)", 2))

        # ── 5. RSI ────────────────────────────────────────────────────────────
        rsi = now["rsi"]
        if rsi < 30:
            bull.append((f"RSI deeply oversold ({rsi:.1f})", 4))
        elif rsi < 40 and rsi > prev["rsi"]:
            bull.append((f"RSI recovering from oversold ({rsi:.1f} ↑)", 3))
        elif rsi > 70:
            bear.append((f"RSI deeply overbought ({rsi:.1f})", 4))
        elif rsi > 60 and rsi < prev["rsi"]:
            bear.append((f"RSI rolling over from overbought ({rsi:.1f} ↓)", 3))
        # Neutral zone RSI still contributes directionally
        elif rsi > 50:
            bull.append((f"RSI bullish zone ({rsi:.1f})", 1))
        else:
            bear.append((f"RSI bearish zone ({rsi:.1f})", 1))

        # ── 6. MACD CROSSOVER ─────────────────────────────────────────────────
        if now["macd_hist"] > 0 and prev["macd_hist"] <= 0:
            bull.append(("MACD histogram flipped positive (fresh bullish cross)", 4))
        elif now["macd_hist"] < 0 and prev["macd_hist"] >= 0:
            bear.append(("MACD histogram flipped negative (fresh bearish cross)", 4))
        elif now["macd"] > now["macd_sig"] and now["macd_hist"] > prev["macd_hist"]:
            bull.append(("MACD above signal & histogram expanding", 2))
        elif now["macd"] < now["macd_sig"] and now["macd_hist"] < prev["macd_hist"]:
            bear.append(("MACD below signal & histogram shrinking", 2))

        # ── 7. BOLLINGER BANDS ────────────────────────────────────────────────
        bp = now["bb_pct"]  # 0=lower band, 1=upper band
        if bp < 0.05:
            bull.append(("Price touching/below lower Bollinger Band", 4))
        elif bp < 0.2:
            bull.append(("Price in lower Bollinger zone", 2))
        elif bp > 0.95:
            bear.append(("Price touching/above upper Bollinger Band", 4))
        elif bp > 0.8:
            bear.append(("Price in upper Bollinger zone", 2))

        # Squeeze breakout (bands contracting → expansion)
        band_width_now  = (now["bb_upper"]  - now["bb_lower"])  / now["bb_mid"]
        band_width_prev = (prev["bb_upper"] - prev["bb_lower"]) / prev["bb_mid"]
        if band_width_now > band_width_prev * 1.1:
            if now["close"] > now["bb_mid"]:
                bull.append(("Bollinger squeeze breakout to upside", 3))
            else:
                bear.append(("Bollinger squeeze breakout to downside", 3))

        # ── 8. STOCHASTIC ─────────────────────────────────────────────────────
        sk, sd = now["stoch_k"], now["stoch_d"]
        if sk < 20 and sk > prev["stoch_k"] and sk > sd:
            bull.append((f"Stochastic oversold crossover ({sk:.1f})", 3))
        elif sk < 20:
            bull.append((f"Stochastic in oversold zone ({sk:.1f})", 2))
        elif sk > 80 and sk < prev["stoch_k"] and sk < sd:
            bear.append((f"Stochastic overbought crossover ({sk:.1f})", 3))
        elif sk > 80:
            bear.append((f"Stochastic in overbought zone ({sk:.1f})", 2))

        # ── 9. VOLUME CONFIRMATION ────────────────────────────────────────────
        vol_ratio = now["volume"] / now["vol_sma20"] if now["vol_sma20"] > 0 else 1
        if vol_ratio > 1.5:
            if now["close"] >= now["open"]:
                bull.append((f"High-volume bullish candle ({vol_ratio:.1f}x avg)", 2))
            else:
                bear.append((f"High-volume bearish candle ({vol_ratio:.1f}x avg)", 2))

        # ── 10. CANDLE PATTERN (simple) ───────────────────────────────────────
        body = abs(now["close"] - now["open"])
        total_range = now["high"] - now["low"] or 1
        if body / total_range > 0.7:
            if now["close"] > now["open"]:
                bull.append(("Strong bullish candle (large body)", 1))
            else:
                bear.append(("Strong bearish candle (large body)", 1))

        # ── TALLY ─────────────────────────────────────────────────────────────
        b_score = sum(w for _, w in bull)
        s_score = sum(w for _, w in bear)
        total   = b_score + s_score or 1

        # ── DECISION ──────────────────────────────────────────────────────────
        threshold = Config.SIGNAL_THRESHOLD
        if b_score > s_score and b_score >= threshold:
            signal = "BUY"
            conf   = min(int(b_score / (b_score + s_score) * 100), 97)
            reasons = (
                [f"✅ {r}" for r, _ in bull] +
                ([f"⚠️ {r}" for r, _ in bear] if bear else [])
            )
        elif s_score > b_score and s_score >= threshold:
            signal = "SELL"
            conf   = min(int(s_score / (b_score + s_score) * 100), 97)
            reasons = (
                [f"❌ {r}" for r, _ in bear] +
                ([f"⚠️ {r}" for r, _ in bull] if bull else [])
            )
        else:
            signal = "HOLD"
            conf   = 50
            reasons = (
                [f"🟡 Mixed — Bull {b_score:.0f} pts vs Bear {s_score:.0f} pts"] +
                [f"✅ {r}" for r, _ in bull[:3]] +
                [f"❌ {r}" for r, _ in bear[:3]]
            )

        # ── LEVELS ────────────────────────────────────────────────────────────
        price = now["close"]
        atr   = now["atr"] if now["atr"] > 0 else price * 0.015

        if signal == "BUY":
            entry  = price
            sl     = price - atr * 2.0
            tp     = price + atr * 4.0
        elif signal == "SELL":
            entry  = price
            sl     = price + atr * 2.0
            tp     = price - atr * 4.0
        else:
            entry = sl = tp = None

        rr = (abs(tp - entry) / abs(entry - sl)) if entry and sl and entry != sl else None

        return {
            "signal":       signal,
            "confidence":   conf,
            "reasons":      reasons,
            "entry":        _r(entry),
            "stop_loss":    _r(sl),
            "take_profit":  _r(tp),
            "risk_reward":  round(rr, 2) if rr else None,
            "bull_score":   round(b_score, 1),
            "bear_score":   round(s_score, 1),
            "current_price": _r(price),
            "rsi":          round(now["rsi"], 1),
            "stoch_k":      round(now["stoch_k"], 1),
            "atr":          _r(atr),
            "vol_ratio":    round(vol_ratio, 2),
        }


def _r(v):
    """Round to sensible precision."""
    if v is None:
        return None
    if v >= 1000:
        return round(v, 2)
    if v >= 1:
        return round(v, 4)
    return round(v, 8)


def _empty_signal(reason: str) -> Dict[str, Any]:
    return {
        "signal": "HOLD", "confidence": 0,
        "reasons": [reason],
        "entry": None, "stop_loss": None, "take_profit": None,
        "risk_reward": None, "bull_score": 0, "bear_score": 0,
        "current_price": 0, "rsi": 0, "stoch_k": 0,
        "atr": 0, "vol_ratio": 1,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 🤖  OPTIONAL NVIDIA AI COMMENTARY
# ─────────────────────────────────────────────────────────────────────────────
class AICommentary:
    def __init__(self):
        self.ok = False
        if not NVIDIA_API_KEY or NVIDIA_API_KEY.startswith("nvapi-YOUR"):
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=Config.NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, timeout=20)
            self.ok = True
        except Exception as e:
            log.warning(f"AI init failed: {e}")

    def comment(self, symbol: str, sig: Dict) -> str:
        if not self.ok:
            return ""
        try:
            prompt = (
                f"Crypto signal for {symbol}:\n"
                f"Signal: {sig['signal']}  Confidence: {sig['confidence']}%\n"
                f"Price: {sig['current_price']}  RSI: {sig['rsi']}\n"
                f"Bull score: {sig['bull_score']}  Bear score: {sig['bear_score']}\n"
                f"Top reasons: {'; '.join(sig['reasons'][:4])}\n\n"
                "Give a 2-sentence trader commentary. Be direct and concise."
            )
            resp = self._client.chat.completions.create(
                model=Config.NVIDIA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=120,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"AI commentary failed: {e}")
            return ""

# ─────────────────────────────────────────────────────────────────────────────
# 📈  CHART
# ─────────────────────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, symbol: str, sig: Dict) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=(f"📊 {symbol}", "RSI (14)", "MACD"),
    )

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#00C853", decreasing_line_color="#FF1744",
        name="Price",
    ), row=1, col=1)

    # EMAs
    for col_name, color, label in [
        ("ema9",  "#FFD600", "EMA9"),
        ("ema20", "#2979FF", "EMA20"),
        ("ema50", "#FF6D00", "EMA50"),
    ]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name],
                line=dict(color=color, width=1),
                name=label, opacity=0.8,
            ), row=1, col=1)

    # Bollinger Bands
    for col_name, color in [("bb_upper", "#7E57C2"), ("bb_mid", "#455A64"), ("bb_lower", "#7E57C2")]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name],
                line=dict(color=color, width=1, dash="dot"),
                name=col_name.replace("bb_", "BB ").title(),
                opacity=0.6,
            ), row=1, col=1)

    # Signal marker
    if sig["signal"] != "HOLD" and sig["entry"]:
        marker_color = "#00C853" if sig["signal"] == "BUY" else "#FF1744"
        fig.add_trace(go.Scatter(
            x=[df.index[-1]], y=[sig["entry"]],
            mode="markers+text",
            marker=dict(symbol="triangle-up" if sig["signal"] == "BUY" else "triangle-down",
                        size=16, color=marker_color),
            text=[f" {sig['signal']}"], textposition="middle right",
            name="Signal",
        ), row=1, col=1)

    # ── RSI ──
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi"],
        line=dict(color="#E040FB", width=1.5),
        name="RSI",
    ), row=2, col=1)
    for level, color in [(70, "rgba(255,23,68,0.3)"), (30, "rgba(0,200,83,0.3)")]:
        fig.add_hline(y=level, line_color=color, line_dash="dot", row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)", line_width=0, row=2, col=1)

    # ── MACD ──
    if "macd_hist" in df.columns:
        hist_colors = ["#00C853" if v >= 0 else "#FF1744" for v in df["macd_hist"]]
        fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], marker_color=hist_colors, name="MACD Hist"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], line=dict(color="#2979FF", width=1), name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_sig"], line=dict(color="#FF6D00", width=1), name="Signal"), row=3, col=1)

    fig.update_layout(
        height=620,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        xaxis_rangeslider_visible=False,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 🖥️  STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
def signal_color(s: str) -> str:
    return {"BUY": "#00C853", "SELL": "#FF1744", "HOLD": "#FFB300"}.get(s, "#888")

def signal_emoji(s: str) -> str:
    return {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(s, "⚪")


def render_signal_card(sig: Dict):
    s = sig["signal"]
    color = signal_color(s)
    icon  = signal_emoji(s)

    if s == "HOLD":
        st.markdown(f"""
        <div style="border:2px solid #FFB300; border-radius:12px; padding:20px; text-align:center;
                    background:rgba(255,179,0,0.08);">
          <h2 style="color:#FFB300; margin:0;">🟡 HOLD</h2>
          <p style="color:#aaa; margin:8px 0 0;">Bull {sig['bull_score']} pts  ·  Bear {sig['bear_score']} pts
          <br>Need {Config.SIGNAL_THRESHOLD} pts to trigger</p>
        </div>""", unsafe_allow_html=True)
        return

    entry_str = f"${sig['entry']:,.4f}"   if sig['entry']       else "—"
    sl_str    = f"${sig['stop_loss']:,.4f}" if sig['stop_loss']  else "—"
    tp_str    = f"${sig['take_profit']:,.4f}" if sig['take_profit'] else "—"
    rr_str    = f"1 : {sig['risk_reward']}"  if sig['risk_reward'] else "—"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{color}22,{color}08);
                border:3px solid {color}; border-radius:15px; padding:24px; margin:6px 0;">
      <h2 style="color:{color}; margin:0 0 18px; text-align:center; font-size:2em;">
        {icon} {s} SIGNAL
      </h2>
      <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:14px;">
        {''.join([
            f'<div style="text-align:center;padding:12px;background:rgba(0,0,0,0.3);border-radius:10px;">'
            f'<div style="font-size:.8em;color:#888;">{label}</div>'
            f'<div style="font-size:1.4em;color:{vcolor};font-weight:bold;">{val}</div></div>'
            for label, val, vcolor in [
                ("🎯 Entry",      entry_str, color),
                ("🛑 Stop Loss",  sl_str,    "#FF5252"),
                ("💰 Take Profit",tp_str,    "#00E676"),
            ]
        ])}
      </div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
        <div style="text-align:center;padding:12px;background:rgba(0,0,0,0.3);border-radius:10px;">
          <div style="font-size:.8em;color:#888;">📊 Confidence</div>
          <div style="font-size:1.6em;color:{color};font-weight:bold;">{sig['confidence']}%</div>
        </div>
        <div style="text-align:center;padding:12px;background:rgba(0,0,0,0.3);border-radius:10px;">
          <div style="font-size:.8em;color:#888;">⚖️ Risk : Reward</div>
          <div style="font-size:1.6em;color:#fff;font-weight:bold;">{rr_str}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_score_bar(bull: float, bear: float):
    total = bull + bear or 1
    bp = int(bull / total * 100)
    sp = 100 - bp
    st.markdown(f"""
    <div style="margin:10px 0;">
      <div style="display:flex; justify-content:space-between; font-size:.85em; color:#aaa; margin-bottom:4px;">
        <span>🐂 Bull {bull:.0f} pts</span>
        <span>🐻 Bear {bear:.0f} pts</span>
      </div>
      <div style="background:#333; border-radius:8px; height:14px; overflow:hidden;">
        <div style="background:linear-gradient(90deg,#00C853,#69F0AE);
                    width:{bp}%; height:100%; float:left; border-radius:8px 0 0 8px;"></div>
        <div style="background:linear-gradient(90deg,#FF5252,#FF1744);
                    width:{sp}%; height:100%; float:right; border-radius:0 8px 8px 0;"></div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_scanner_row(sym: str, tf: str) -> Dict:
    """Fetch data and generate signal for scanner table — cached per symbol/tf."""
    df = MarketData.fetch(sym, Config.TIMEFRAMES[tf])
    if df.empty:
        return _empty_signal(f"No data for {sym}")
    df = Indicators.add_all(df)
    return SignalEngine.generate(df)


@st.cache_data(ttl=Config.AUTO_REFRESH_SEC, show_spinner=False)
def cached_signal(symbol: str, timeframe: str) -> Dict:
    df = MarketData.fetch(symbol, Config.TIMEFRAMES[timeframe])
    if df.empty:
        return _empty_signal("Failed to fetch data")
    df = Indicators.add_all(df)
    sig = SignalEngine.generate(df)
    return sig


@st.cache_data(ttl=Config.AUTO_REFRESH_SEC, show_spinner=False)
def cached_df(symbol: str, timeframe: str) -> pd.DataFrame:
    df = MarketData.fetch(symbol, Config.TIMEFRAMES[timeframe])
    if df.empty:
        return df
    return Indicators.add_all(df)


def print_signal(symbol: str, tf: str, sig: Dict):
    sep = "═" * 65
    log.info(f"\n{sep}")
    log.info(f"  {symbol}  [{tf}]   {sig['signal']}   conf={sig['confidence']}%   "
             f"bull={sig['bull_score']}  bear={sig['bear_score']}")
    log.info(f"  Price: {sig['current_price']}   ATR: {sig['atr']}")
    if sig["entry"]:
        log.info(f"  Entry={sig['entry']}  SL={sig['stop_loss']}  TP={sig['take_profit']}  RR={sig['risk_reward']}")
    for r in sig["reasons"][:6]:
        log.info(f"    {r}")
    log.info(sep)


# ─────────────────────────────────────────────────────────────────────────────
# 🚀  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="🚀 Crypto Signals",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── CSS ──
    st.markdown("""
    <style>
      .block-container { padding-top: 1rem; }
      [data-testid="stSidebar"] { background: #0D0D0D; }
      .stMetric label { font-size: .85em !important; }
    </style>""", unsafe_allow_html=True)

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        symbol    = st.selectbox("Trading Pair", Config.PAIRS, index=0)
        tf_label  = st.selectbox("Timeframe",    list(Config.TIMEFRAMES.keys()), index=2)
        threshold = st.slider("Signal Threshold (bull/bear pts)", 3, 12, Config.SIGNAL_THRESHOLD)
        Config.SIGNAL_THRESHOLD = threshold

        st.markdown("---")
        show_scanner = st.checkbox("📡 Multi-Pair Scanner", value=True)

        st.markdown("---")
        auto_refresh = st.checkbox("🔄 Auto-Refresh", value=False)
        if auto_refresh:
            interval = st.slider("Interval (sec)", 15, 300, Config.AUTO_REFRESH_SEC, step=15)
            st.caption(f"Next refresh in ~{interval}s")
            time.sleep(interval)
            st.cache_data.clear()
            st.rerun()

        if st.button("🔄 Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.caption("⚠️ Educational use only. Not financial advice.")

    # ── MAIN HEADER ──
    st.markdown("# 🚀 Crypto Trading AI — Real-Time Signals")

    # ── FETCH DATA ──
    with st.spinner(f"Fetching {symbol} / {tf_label}…"):
        df  = cached_df(symbol, tf_label)
        sig = cached_signal(symbol, tf_label)

    if df.empty:
        st.error("❌ Failed to fetch market data. Check internet / Binance availability.")
        return

    print_signal(symbol, tf_label, sig)

    # ── LAYOUT ──
    left, right = st.columns([2, 1], gap="medium")

    with left:
        # Chart
        fig = build_chart(df.tail(150), symbol, sig)
        st.plotly_chart(fig, use_container_width=True)

        # Metrics row
        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        price_delta = latest["close"] - prev["close"]
        price_delta_pct = price_delta / prev["close"] * 100

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("💵 Price",    f"${latest['close']:,.4f}",   f"{price_delta_pct:+.2f}%")
        m2.metric("📊 RSI",      f"{latest['rsi']:.1f}",       f"{latest['rsi']-prev['rsi']:+.1f}")
        m3.metric("📈 Stoch K",  f"{latest['stoch_k']:.1f}")
        m4.metric("📉 ATR",      f"{latest['atr']:.4f}")
        m5.metric("🔊 Vol ×",    f"{sig['vol_ratio']:.1f}x")

    with right:
        # Signal card
        render_signal_card(sig)

        # Score bar
        st.markdown("#### 📊 Bull vs Bear")
        render_score_bar(sig["bull_score"], sig["bear_score"])

        # Reasons
        with st.expander("🔍 Signal Breakdown", expanded=True):
            for r in sig["reasons"]:
                st.markdown(f"<div style='font-size:.9em; padding:3px 0;'>{r}</div>",
                            unsafe_allow_html=True)

        # AI commentary
        if NVIDIA_API_KEY and not NVIDIA_API_KEY.startswith("nvapi-YOUR"):
            with st.expander("🤖 AI Commentary", expanded=False):
                with st.spinner("Asking AI…"):
                    ai = AICommentary()
                    comment = ai.comment(symbol, sig)
                if comment:
                    st.info(comment)
                else:
                    st.caption("AI unavailable — check API key")

    # ── MULTI-PAIR SCANNER ──
    if show_scanner:
        st.markdown("---")
        st.markdown("### 📡 Multi-Pair Scanner")
        st.caption(f"Scanning {', '.join(Config.PAIRS)} on {tf_label}")

        scanner_cols = st.columns(len(Config.PAIRS))
        for col, pair in zip(scanner_cols, Config.PAIRS):
            with col:
                with st.spinner(f"{pair}…"):
                    s = cached_signal(pair, tf_label)
                color = signal_color(s["signal"])
                emoji = signal_emoji(s["signal"])
                st.markdown(f"""
                <div style="border:2px solid {color}; border-radius:10px; padding:12px;
                            text-align:center; background:{color}11; min-height:90px;">
                  <div style="font-size:.85em; color:#aaa;">{pair}</div>
                  <div style="font-size:1.5em; color:{color}; font-weight:bold;">{emoji} {s['signal']}</div>
                  <div style="font-size:.8em; color:#888;">{s['confidence']}%  
                    <span style="color:#00C853;">▲{s['bull_score']:.0f}</span> 
                    <span style="color:#FF1744;">▼{s['bear_score']:.0f}</span>
                  </div>
                  <div style="font-size:.8em; color:#aaa;">${s['current_price']:,.4f}</div>
                </div>""", unsafe_allow_html=True)

    # ── FOOTER ──
    st.markdown("---")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.caption(f"🕐 Last updated: {ts}  ·  Threshold: {Config.SIGNAL_THRESHOLD} pts  ·  Timeframe: {tf_label}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
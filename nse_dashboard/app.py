import os
import math
import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

from nse_data import fetch_option_chain, parse_option_chain, generate_signal
from demo_data import generate_demo_option_chain
from stock_list import NSE_STOCKS, INDICES, FNO_INDICES, INDEX_YF_SYMBOLS, search_stocks
import paper_trade as pt
import chat_bot
import backtest
import model_train

# ── Technical indicator helpers ───────────────────────────────────────────────

def compute_technicals(df):
    """
    Compute VWAP, EMA-9, EMA-20, RSI-14, PVT and Net Volume from an OHLCV
    DataFrame with columns: datetime, open, high, low, close, volume.
    Returns a copy with new columns added.
    """
    d = df.copy().reset_index(drop=True)
    # VWAP (cumulative session-style over the full window)
    hlc3          = (d["high"] + d["low"] + d["close"]) / 3
    cum_vol       = d["volume"].cumsum().replace(0, np.nan)
    d["vwap"]     = (hlc3 * d["volume"]).cumsum() / cum_vol
    # EMA 9 & 20
    d["ema9"]     = d["close"].ewm(span=9,  adjust=False).mean()
    d["ema20"]    = d["close"].ewm(span=20, adjust=False).mean()
    # RSI 14
    delta         = d["close"].diff()
    gain          = delta.clip(lower=0)
    loss          = (-delta).clip(lower=0)
    avg_gain      = gain.ewm(com=13, min_periods=1).mean()
    avg_loss      = loss.ewm(com=13, min_periods=1).mean()
    d["rsi"]      = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-9))
    # MACD (12, 26, 9)
    ema12          = d["close"].ewm(span=12, adjust=False).mean()
    ema26          = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]
    # PVT (Price Volume Trend)
    pct_chg       = d["close"].pct_change().fillna(0)
    d["pvt"]      = (pct_chg * d["volume"]).cumsum()
    # Net Volume (positive when close rises, negative when it falls)
    direction     = np.sign(d["close"].diff().fillna(0))
    d["net_vol"]  = d["volume"] * direction
    return d


def tech_signal(d):
    """
    Derive a BULLISH / BEARISH / NEUTRAL signal from computed_technicals output.
    Returns (signal_str, color_str, score_int, reasons_list).
    """
    if len(d) < 5:
        return "WAIT", "orange", 0, ["Not enough data"]
    r    = d.iloc[-1]
    score   = 0
    reasons = []
    # 1. Price vs VWAP
    if r["close"] > r["vwap"]:
        score += 1
        reasons.append("✅ Price {:.1f} > VWAP {:.1f} → Bullish".format(r["close"], r["vwap"]))
    else:
        score -= 1
        reasons.append("❌ Price {:.1f} < VWAP {:.1f} → Bearish".format(r["close"], r["vwap"]))
    # 2. EMA crossover
    if r["ema9"] > r["ema20"]:
        score += 1
        reasons.append("✅ EMA9 {:.1f} > EMA20 {:.1f} → Bullish crossover".format(r["ema9"], r["ema20"]))
    else:
        score -= 1
        reasons.append("❌ EMA9 {:.1f} < EMA20 {:.1f} → Bearish crossover".format(r["ema9"], r["ema20"]))
    # 3. RSI
    rsi_val = float(r["rsi"])
    if rsi_val > 60:
        score += 1
        reasons.append("✅ RSI {:.1f} → Bullish momentum".format(rsi_val))
    elif rsi_val < 40:
        score -= 1
        reasons.append("❌ RSI {:.1f} → Bearish pressure".format(rsi_val))
    else:
        reasons.append("⚠️ RSI {:.1f} → Neutral zone (40–60)".format(rsi_val))
    # 4. PVT trend (3-bar slope)
    if len(d) >= 4:
        pvt_slope = float(d["pvt"].iloc[-1] - d["pvt"].iloc[-4])
        if pvt_slope > 0:
            score += 1
            reasons.append("✅ PVT rising → Volume supporting bulls")
        else:
            score -= 1
            reasons.append("❌ PVT falling → Volume supporting bears")
    # 5. Net Volume (last 3 bars)
    if len(d) >= 3:
        net_sum = float(d["net_vol"].iloc[-3:].sum())
        if net_sum > 0:
            score += 1
            reasons.append("✅ Net volume positive → Buying pressure")
        else:
            score -= 1
            reasons.append("❌ Net volume negative → Selling pressure")
    if score >= 3:
        return "STRONG BULLISH", "green", score, reasons
    elif score >= 1:
        return "BULLISH", "#8BC34A", score, reasons
    elif score <= -3:
        return "STRONG BEARISH", "red", score, reasons
    elif score <= -1:
        return "BEARISH", "#ef5350", score, reasons
    return "NEUTRAL", "orange", score, reasons


_STRIKE_STEP = {
    "NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50,
    "MIDCPNIFTY": 25, "SENSEX": 100,
}

def _parse_expiry_date(expiry_str):
    """Return a date object from any common expiry string, or None."""
    from datetime import date as _date
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d %b %Y", "%d-%B-%Y", "%d%b%Y"):
        try:
            return datetime.strptime(expiry_str.upper(), fmt.upper()).date()
        except Exception:
            pass
    return None


def generate_trade_setups(oi_sig, tech_df, underlying, expiry, symbol):
    """
    Combine OI signal + MACD + RSI + EMA + VWAP + PVT into per-timeframe
    trade setup cards: 10min / 15min / 30min / 1hr.

    Returns dict  tf → {direction, confidence, strike, premium_est,
                         sl_pct, tgt_pct, dte, dte_note, score, bullets}
    """
    from datetime import date as _date
    TIMEFRAMES = ["10min", "15min", "30min", "1hr"]

    # ── DTE calculation ───────────────────────────────────────────────────────
    exp_date = _parse_expiry_date(expiry)
    dte = int((exp_date - _date.today()).days) if exp_date else 7
    dte = max(0, dte)

    # ── DTE risk profile (SL%, Target%) ──────────────────────────────────────
    if dte == 0:
        dte_note = "🔴 EXPIRY DAY — Avoid option buys (max theta risk)"
        dte_sl, dte_tgt = 15, 20
    elif dte == 1:
        dte_note = "⚠️ 1 DTE — Scalp only, very tight size"
        dte_sl, dte_tgt = 20, 30
    elif dte <= 3:
        dte_note = "⏳ {} DTE — Short-expiry scalp, ATM preferred".format(dte)
        dte_sl, dte_tgt = 30, 50
    elif dte <= 7:
        dte_note = "📅 {} DTE — Weekly expiry window, intraday/swing OK".format(dte)
        dte_sl, dte_tgt = 35, 65
    else:
        dte_note = "📆 {} DTE — Positional friendly, spreads recommended".format(dte)
        dte_sl, dte_tgt = 40, 80

    # ── Technical composite score (from intraday candles) ────────────────────
    ts      = 0
    bullets = []

    if tech_df is not None and not tech_df.empty and len(tech_df) >= 5:
        r    = tech_df.iloc[-1]
        prev = tech_df.iloc[-2]

        # 1. MACD
        macd_val  = float(r.get("macd",     r.get("macd",     0)))
        macd_sig  = float(r.get("macd_sig", r.get("macd_signal", 0)))
        macd_hist = float(r.get("macd_hist", 0))
        if macd_val > macd_sig:
            ts += 2
            strength = "rising ↑" if macd_hist > float(prev.get("macd_hist", 0)) else "above signal"
            bullets.append("✅ MACD {} → Bullish".format(strength))
        else:
            ts -= 2
            bullets.append("❌ MACD below signal → Bearish")

        # 2. RSI
        rsi_v = float(r.get("rsi", r.get("rsi14", 50)))
        if rsi_v > 65:
            ts += 2; bullets.append("✅ RSI {:.0f} → Strong bullish momentum".format(rsi_v))
        elif rsi_v > 55:
            ts += 1; bullets.append("✅ RSI {:.0f} → Mild bullish".format(rsi_v))
        elif rsi_v < 35:
            ts -= 2; bullets.append("❌ RSI {:.0f} → Strong bearish pressure".format(rsi_v))
        elif rsi_v < 45:
            ts -= 1; bullets.append("❌ RSI {:.0f} → Mild bearish".format(rsi_v))
        else:
            bullets.append("⚪ RSI {:.0f} → Neutral".format(rsi_v))

        # 3. EMA crossover
        e9, e20 = float(r.get("ema9", 0)), float(r.get("ema20", r.get("ema21", 0)))
        if e9 > e20:
            ts += 1; bullets.append("✅ EMA9 > EMA20 → Uptrend")
        else:
            ts -= 1; bullets.append("❌ EMA9 < EMA20 → Downtrend")

        # 4. Price vs VWAP
        close_v, vwap_v = float(r.get("close", 0)), float(r.get("vwap", 0))
        if vwap_v > 0:
            if close_v > vwap_v:
                ts += 1; bullets.append("✅ Price above VWAP → Intraday bullish")
            else:
                ts -= 1; bullets.append("❌ Price below VWAP → Intraday bearish")

        # 5. PVT slope (3-bar)
        if "pvt" in tech_df.columns and len(tech_df) >= 4:
            pvt_slope = float(tech_df["pvt"].iloc[-1] - tech_df["pvt"].iloc[-4])
            if pvt_slope > 0:
                ts += 1; bullets.append("✅ PVT rising → Volume confirming bulls")
            else:
                ts -= 1; bullets.append("❌ PVT falling → Volume confirming bears")

    else:
        bullets.append("⚠️ Intraday candles unavailable — using OI signal only")

    # ── OI signal score (normalised to ±5) ───────────────────────────────────
    oi_raw   = oi_sig.get("score", 0)
    oi_norm  = max(-5, min(5, round(oi_raw / 20)))
    oi_dir   = oi_sig.get("signal", "AVOID / WAIT")
    oi_pcr   = oi_sig.get("pcr", 1.0)
    bullets.append("📊 OI score {:+d} (PCR={}, Signal: {})".format(oi_norm, oi_pcr, oi_dir))

    combined = ts + oi_norm  # range: about -12 to +12

    # ── Direction ─────────────────────────────────────────────────────────────
    if combined >= 3:
        direction, dir_col = "BUY CE 🟢", "green"
    elif combined <= -3:
        direction, dir_col = "BUY PE 🔴", "red"
    elif combined >= 1:
        direction, dir_col = "MILD BULLISH ↑", "#8BC34A"
    elif combined <= -1:
        direction, dir_col = "MILD BEARISH ↓", "#ef5350"
    else:
        direction, dir_col = "WAIT / NEUTRAL ⚪", "orange"

    # ── Strike selection ──────────────────────────────────────────────────────
    step = _STRIKE_STEP.get(symbol.upper(), 50)
    atm  = round(underlying / step) * step
    if abs(combined) >= 6:
        rec_strike = atm                     # ATM — max gamma
    elif abs(combined) >= 3:
        rec_strike = (atm + step) if combined > 0 else (atm - step)  # 1 OTM
    else:
        rec_strike = atm

    premium_est = round(underlying * 0.008, 0)   # ~0.8% ATM premium estimate
    confidence  = min(92, max(10, 50 + combined * 6))

    # ── Per-timeframe SL/Target multipliers ──────────────────────────────────
    tf_mults = {
        "10min": (0.50, 0.45, "Scalp   ~10 min"),
        "15min": (0.65, 0.60, "Scalp   ~15 min"),
        "30min": (0.80, 0.75, "Intraday 30 min"),
        "1hr":   (1.00, 1.00, "Intraday  1 hr "),
    }

    setups = {}
    for tf in TIMEFRAMES:
        sl_m, tgt_m, tf_label = tf_mults[tf]
        sl_pct  = round(dte_sl  * sl_m,  1)
        tgt_pct = round(dte_tgt * tgt_m, 1)
        setups[tf] = {
            "label":        tf_label,
            "direction":    direction,
            "dir_col":      dir_col,
            "confidence":   confidence,
            "combined":     combined,
            "tech_score":   ts,
            "oi_score":     oi_norm,
            "atm":          atm,
            "rec_strike":   rec_strike,
            "premium_est":  premium_est,
            "sl_pct":       sl_pct,
            "tgt_pct":      tgt_pct,
            "dte":          dte,
            "dte_note":     dte_note,
            "bullets":      bullets,
        }
    return setups


APP_VERSION = "v2.4"

st.set_page_config(
    page_title="NSE Options Intelligence {}".format(APP_VERSION),
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<style>.stMetric > div { font-size: 18px; }</style>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("NSE Options Intelligence")
    st.divider()

    instrument_type = st.radio("Instrument Type", ["Index", "Stock"])

    if instrument_type == "Index":
        # Only F&O indices have an option chain — others are analysis-only
        fno_indices   = {k: v for k, v in INDICES.items() if k in FNO_INDICES}
        index_options = ["{} — {}".format(k, v) for k, v in fno_indices.items()]
        selected_idx  = st.selectbox("Select Index", index_options,
                                     help="For all indices go to 🇮🇳 Index Analysis tab")
        symbol        = selected_idx.split(" — ")[0]
        is_index      = True
    else:
        search_query = st.text_input(
            "Search Stock", value="SBIN",
            placeholder="Type symbol or name: SBIN, HDFC, TATA..."
        ).upper().strip()
        suggestions = search_stocks(search_query)
        if suggestions:
            opts     = ["{} — {}".format(s, n) for s, n in suggestions]
            selected = st.selectbox("Matches", opts, index=0)
            symbol   = selected.split(" — ")[0]
        else:
            st.warning("No match. Using: " + search_query)
            symbol = search_query
        is_index = False

    strike_mode = st.radio("Strike Selection", ["By Count", "By Points Range"], horizontal=True)
    if strike_mode == "By Count":
        num_strikes     = st.slider("Strikes around ATM", 10, 80, 20, step=2)
        strike_range_pts = None
    else:
        strike_range_pts = st.number_input(
            "Range from ATM (±points)",
            value=500, min_value=50, max_value=5000, step=50,
            help="e.g. 500 → show all strikes within ±500 of ATM. NIFTY strikes are in steps of 50."
        )
        num_strikes = 80
    auto_refresh = st.checkbox("Auto Refresh", value=False)
    refresh_interval = st.selectbox(
        "Refresh every (sec)", [1, 5, 10, 30, 60, 120], index=3
    )

    st.divider()

    data_source = st.radio(
        "Data Source",
        ["Angel One (Live)", "NSE Direct", "NSE CSV Upload", "Demo Mode"],
        help="Angel One = live via SmartAPI | NSE Direct = requires Indian IP | "
             "CSV = upload from nseindia.com | Demo = simulated data",
    )

    csv_file = None
    if data_source == "NSE CSV Upload":
        csv_file = st.file_uploader(
            "Upload NSE Option Chain CSV",
            type=["csv"],
            help="Go to nseindia.com/option-chain, select expiry, click Download (CSV), upload here.",
        )

    st.divider()
    col_r, col_t = st.columns(2)
    with col_r:
        if st.button("Refresh", use_container_width=True, type="primary"):
            st.cache_data.clear()
    with col_t:
        test_mode = st.button("Test NSE", use_container_width=True)

    st.caption("Updated: {}  |  {}".format(datetime.now().strftime("%H:%M:%S"), APP_VERSION))

# ── Test NSE connection ───────────────────────────────────────────────────────
if test_mode:
    with st.spinner("Testing NSE connection for {} ...".format(symbol)):
        raw = fetch_option_chain(symbol, is_index)
    if "error" in raw:
        st.error("NSE connection FAILED: " + raw["error"])
        st.info("Try during market hours (9:15 AM – 3:30 PM IST) on Indian internet.")
    elif not raw:
        st.error("NSE returned empty response.")
    else:
        keys = list(raw.keys())
        st.success("NSE connection OK! Keys: " + str(keys))
        if "records" in raw:
            rec = raw["records"]
            st.write("underlyingValue:", rec.get("underlyingValue"))
            st.write("expiryDates:", rec.get("expiryDates", [])[:3])
            st.write("record count:", len(rec.get("data", [])))
    st.stop()

# ── Cached data fetchers (module-level for st.cache_data) ─────────────────────

@st.cache_data(ttl=60)
def _get_nse_data(sym, idx, strikes):
    raw      = fetch_option_chain(sym, idx)
    df, meta = parse_option_chain(raw, strikes)
    return df, meta

@st.cache_data(ttl=60)
def _get_angelone_data(sym, idx, strikes, range_pts=None):
    from angelone_data import fetch_option_chain_angelone
    return fetch_option_chain_angelone(sym, idx, strikes, range_pts)

@st.cache_data(ttl=300)
def _get_candles(sym, idx, days, src):
    # 1. Angel One live historical candles
    if src == "Angel One (Live)":
        try:
            from angelone_data import fetch_historical_candles
            c = fetch_historical_candles(sym, idx, days)
            if c is not None and not c.empty:
                return c
        except Exception:
            pass
    # 2. yfinance fallback (works for all NSE/BSE indices without auth)
    yf_sym = INDEX_YF_SYMBOLS.get(sym.upper())
    if yf_sym:
        try:
            import yfinance as yf
            tk   = yf.Ticker(yf_sym)
            hist = tk.history(period="{}d".format(days + 15))
            if not hist.empty:
                hist = hist.reset_index()
                date_col = "Date" if "Date" in hist.columns else "Datetime"
                return pd.DataFrame({
                    "datetime": pd.to_datetime(hist[date_col]).dt.tz_localize(None),
                    "open":     hist["Open"].astype(float),
                    "high":     hist["High"].astype(float),
                    "low":      hist["Low"].astype(float),
                    "close":    hist["Close"].astype(float),
                    "volume":   hist["Volume"].astype(float),
                }).tail(days).reset_index(drop=True)
        except Exception:
            pass
    # 3. Synthetic demo fallback
    from demo_data import generate_demo_option_chain
    _, dm  = generate_demo_option_chain(sym if idx else "NIFTY")
    base   = dm["underlying"]
    np.random.seed(42)
    prices = base * np.cumprod(1 + np.random.normal(0, 0.01, days + 10))
    prices = np.clip(prices[:days], base * 0.7, base * 1.3)
    dates  = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
    return pd.DataFrame({
        "datetime": dates,
        "open":     prices * 0.998,
        "high":     prices * 1.005,
        "low":      prices * 0.995,
        "close":    prices,
        "volume":   np.random.randint(100000, 500000, days).astype(float),
    })

@st.cache_data(ttl=60)
def _get_intraday_candles_cached(sym, tf, src):
    """
    Fetch intraday OHLCV for the live-dashboard trade-setup panel.
    Tries Angel One first, then yfinance intraday, then returns empty.
    tf: '5min' | '10min' | '15min' | '30min' | '1hr'
    """
    # 1. Angel One
    if src == "Angel One (Live)":
        try:
            from angelone_data import fetch_intraday_candles
            c = fetch_intraday_candles(sym, True, tf)
            if c is not None and not c.empty:
                return c
        except Exception:
            pass
    # 2. yfinance intraday
    yf_sym = INDEX_YF_SYMBOLS.get(sym.upper())
    if yf_sym:
        try:
            import yfinance as yf
            yf_interval = {"5min": "5m", "10min": "10m", "15min": "15m",
                           "30min": "30m", "1hr": "60m"}.get(tf, "15m")
            yf_period   = {"5min": "1d", "10min": "2d", "15min": "5d",
                           "30min": "10d", "1hr": "20d"}.get(tf, "5d")
            tk   = yf.Ticker(yf_sym)
            hist = tk.history(period=yf_period, interval=yf_interval)
            if not hist.empty:
                hist = hist.reset_index()
                dc   = "Datetime" if "Datetime" in hist.columns else "Date"
                return pd.DataFrame({
                    "datetime": pd.to_datetime(hist[dc]).dt.tz_localize(None),
                    "open":    hist["Open"].astype(float),
                    "high":    hist["High"].astype(float),
                    "low":     hist["Low"].astype(float),
                    "close":   hist["Close"].astype(float),
                    "volume":  hist["Volume"].astype(float),
                }).dropna(subset=["close"]).reset_index(drop=True)
        except Exception:
            pass
    return pd.DataFrame()


# ── Data fetch ────────────────────────────────────────────────────────────────

df   = pd.DataFrame()
meta = {}

if data_source == "Demo Mode":
    df, meta = generate_demo_option_chain(symbol if is_index else "NIFTY")
    st.info("Demo Mode ON — simulated data. Switch Data Source for live data.")

elif data_source == "NSE CSV Upload":
    if csv_file is None:
        st.warning("Upload a CSV file from nseindia.com/option-chain to proceed.")
        st.markdown(
            "**How to download:**\n"
            "1. Go to nseindia.com/option-chain\n"
            "2. Select Index / Stock and Expiry\n"
            "3. Click **Download (CSV)** button\n"
            "4. Upload the file here"
        )
        st.stop()
    from csv_parser import parse_nse_csv
    with st.spinner("Parsing uploaded CSV..."):
        df, meta = parse_nse_csv(csv_file)
    if "error" in meta:
        st.error("CSV parse error: " + meta["error"])
        st.stop()

elif data_source == "NSE Direct":
    with st.spinner("Fetching {} from NSE...".format(symbol)):
        df, meta = _get_nse_data(symbol, is_index, num_strikes)
    if "error" in meta:
        st.error("NSE fetch failed: " + meta["error"])
        st.warning(
            "NSE requires Indian internet + market hours.\n\n"
            "1. Click **Test NSE** to diagnose\n"
            "2. Switch to **Angel One (Live)** or **Demo Mode**"
        )
        st.stop()

else:  # Angel One (Live)
    with st.spinner("Fetching {} via Angel One SmartAPI...".format(symbol)):
        df, meta = _get_angelone_data(symbol, is_index, num_strikes, strike_range_pts)
    if "error" in meta:
        st.error("Angel One fetch failed: " + meta["error"])
        st.warning(
            "Possible causes:\n"
            "- Angel One account dormant / not activated\n"
            "- TOTP mismatch (check system clock)\n"
            "- Market closed (pre-market / weekend)\n\n"
            "Switch to **Demo Mode** to preview the dashboard."
        )
        st.stop()

if df.empty:
    st.warning("No option chain data. Enable Demo Mode to preview.")
    st.stop()

# ── Shared signal computation ─────────────────────────────────────────────────
sig        = generate_signal(df, meta)
underlying = meta["underlying"]
expiry     = meta["expiry"]
pcr        = sig["pcr"]
max_pain   = sig["max_pain"]
source_tag = meta.get("source", data_source)
meta["symbol"] = symbol

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_bt, tab_pt, tab_chat, tab_idx = st.tabs(
    ["📊 Live Dashboard", "📈 Backtest", "💼 Paper Trade", "💬 Chat", "🇮🇳 Index Analysis"]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
with tab_live:
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## {} Option Chain  —  Expiry: **{}**".format(symbol, expiry))
    st.markdown("**Spot: ₹{:,.2f}**  |  ATM: **{}**  |  Source: `{}`  |  `{}`".format(
        underlying, meta["atm"], source_tag, datetime.now().strftime("%H:%M:%S")))

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION A — MULTI-TIMEFRAME TRADE SETUP
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🎯 Live Trade Setup")

    # Fetch 15-min intraday candles for technical analysis
    with st.spinner("Loading intraday technicals…"):
        intra_raw   = _get_intraday_candles_cached(symbol, "15min", data_source)
        intra_tech  = compute_technicals(intra_raw) if not intra_raw.empty else pd.DataFrame()

    setups = generate_trade_setups(sig, intra_tech, underlying, expiry, symbol)

    if setups:
        # DTE warning banner
        dte_val  = list(setups.values())[0]["dte"]
        dte_note = list(setups.values())[0]["dte_note"]
        if dte_val <= 1:
            st.error(dte_note)
        elif dte_val <= 3:
            st.warning(dte_note)
        else:
            st.info(dte_note)

        # 4 timeframe cards
        tf_cols = st.columns(4)
        tf_order = ["10min", "15min", "30min", "1hr"]
        for col, tf in zip(tf_cols, tf_order):
            s = setups[tf]
            bg_c  = {"green": "#0d2e0d", "red": "#2e0d0d",
                     "#8BC34A": "#1a2e0d", "#ef5350": "#2e1010", "orange": "#2e2200"}
            brd_c = {"green": "#4CAF50", "red": "#f44336",
                     "#8BC34A": "#8BC34A", "#ef5350": "#ef5350", "orange": "#FF9800"}
            bg  = bg_c.get(s["dir_col"],  "#1e1e2e")
            brd = brd_c.get(s["dir_col"], "#888")
            col.markdown(
                '<div style="background:{bg};border:2px solid {brd};padding:14px;'
                'border-radius:12px;text-align:center;margin-bottom:4px;">'
                '<div style="color:#aaa;font-size:11px;letter-spacing:1px;">{label}</div>'
                '<div style="color:{brd};font-size:17px;font-weight:bold;margin:6px 0">{dir}</div>'
                '<div style="color:#ccc;font-size:12px;">Strike: <b>₹{strike:,.0f}</b></div>'
                '<div style="color:#ccc;font-size:12px;">Entry ~<b>₹{prem:.0f}</b></div>'
                '<div style="color:#ef5350;font-size:12px;">SL: <b>{sl}%</b></div>'
                '<div style="color:#26a69a;font-size:12px;">Tgt: <b>{tgt}%</b></div>'
                '<div style="color:#FFD700;font-size:13px;margin-top:6px;font-weight:bold;">'
                'Conf: {conf}%</div>'
                '</div>'.format(
                    bg=bg, brd=brd,
                    label=s["label"].strip(),
                    dir=s["direction"],
                    strike=s["rec_strike"],
                    prem=s["premium_est"],
                    sl=s["sl_pct"], tgt=s["tgt_pct"],
                    conf=s["confidence"],
                ),
                unsafe_allow_html=True,
            )

        # Score breakdown
        st.divider()
        score_col, detail_col = st.columns([1, 2])
        with score_col:
            first = list(setups.values())[0]
            ts_v, oi_v, tot_v = first["tech_score"], first["oi_score"], first["combined"]
            score_color = "#4CAF50" if tot_v > 0 else "#ef5350" if tot_v < 0 else "#FF9800"
            st.markdown(
                '<div style="background:#1e1e2e;padding:18px;border-radius:12px;text-align:center;">'
                '<div style="color:#aaa;font-size:12px;">COMPOSITE SCORE</div>'
                '<div style="font-size:40px;font-weight:bold;color:{c};">{t:+d}</div>'
                '<div style="color:#ccc;font-size:12px;margin-top:8px;">'
                'Tech: <b style="color:{tc}">{ts:+d}</b> &nbsp;|&nbsp; '
                'OI: <b style="color:{oc}">{oi:+d}</b>'
                '</div>'
                '<div style="color:#aaa;font-size:11px;margin-top:6px;">'
                '(+12 Strong Bull → −12 Strong Bear)</div>'
                '</div>'.format(
                    c=score_color, t=tot_v,
                    tc="#4CAF50" if ts_v > 0 else "#ef5350",
                    ts=ts_v,
                    oc="#4CAF50" if oi_v > 0 else "#ef5350",
                    oi=oi_v,
                ),
                unsafe_allow_html=True,
            )
        with detail_col:
            st.markdown("##### Indicator Breakdown (15-min candles + OI)")
            for b in first["bullets"]:
                st.markdown(b)

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION B — MARKET METRICS + OI INTELLIGENCE
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 📊 Market Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spot Price",    "₹{:,.1f}".format(underlying))
    c2.metric("PCR",           str(pcr), delta="Bullish" if pcr > 1.0 else "Bearish")
    c3.metric("Max Pain",      "₹{:,.0f}".format(max_pain))
    c4.metric("CE Resistance", "₹{:,.0f}".format(sig["max_ce_resistance"]), delta="Sell Wall")
    c5.metric("PE Support",    "₹{:,.0f}".format(sig["max_pe_support"]),    delta="Buy Wall")

    oi1, oi2, oi3, oi4, oi5 = st.columns(5)
    call_sum_v = sig.get("call_sum", 0)
    put_sum_v  = sig.get("put_sum", 0)
    oi_diff_v  = sig.get("oi_difference", 0)
    itm_r      = sig.get("itm_ratio", 0)
    strat_type = sig.get("strategy_type", "WAIT")
    oi1.metric("Call Sum (ATM±1)", "{:+.1f}K".format(call_sum_v),
               delta="CE writing ↑" if call_sum_v > 0 else "CE covering ↓")
    oi2.metric("Put Sum (ATM±1)", "{:+.1f}K".format(put_sum_v),
               delta="PE writing ↑" if put_sum_v > 0 else "PE covering ↓")
    oi3.metric("OI Difference", "{:+.1f}K".format(oi_diff_v),
               delta="Bearish" if oi_diff_v > 0 else "Bullish")
    oi4.metric("ITM Ratio", "{:.2f}x".format(itm_r),
               delta="Bullish" if itm_r > 1.5 else ("Bearish" if itm_r < 0.67 and itm_r > 0 else "Neutral"))
    oi5.metric("Suggested Strategy", strat_type)

    st.divider()

    # ── OI Signal analysis (collapsible) ─────────────────────────────────────
    with st.expander("📋 OI Signal Analysis & Key Levels", expanded=False):
        sig_col, reason_col = st.columns([1, 2])
        with sig_col:
            bg     = {"green": "#1a7a1a", "red": "#8b1a1a", "orange": "#7a5c00"}.get(sig["color"], "#333")
            border = {"green": "#4CAF50", "red": "#f44336", "orange": "#FF9800"}.get(sig["color"], "#888")
            st.markdown(
                '<div style="background:{};border:3px solid {};padding:25px;border-radius:14px;text-align:center;">'
                '<div style="font-size:14px;color:#ccc;margin-bottom:6px;">OI SIGNAL</div>'
                '<div style="font-size:28px;font-weight:bold;color:{};">{}</div>'
                '<div style="font-size:16px;color:#ddd;margin-top:8px;">Confidence</div>'
                '<div style="font-size:42px;font-weight:bold;color:white;">{}%</div>'
                '</div>'.format(bg, border, border, sig["signal"], sig["confidence"]),
                unsafe_allow_html=True
            )
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=sig["confidence"],
                title={"text": "OI Signal Strength", "font": {"size": 13}},
                gauge={
                    "axis": {"range": [0, 100]}, "bar": {"color": border},
                    "steps": [
                        {"range": [0,  40], "color": "#3d0000"},
                        {"range": [40, 65], "color": "#3d3d00"},
                        {"range": [65,100], "color": "#003d00"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": sig["confidence"]},
                },
                number={"suffix": "%"},
            ))
            fig_g.update_layout(height=200, margin=dict(t=30, b=0, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_g, use_container_width=True)
        with reason_col:
            st.markdown("##### OI Factors")
            for r in sig["reasons"]:
                rl   = r.lower()
                icon = "🟢" if any(w in rl for w in ["bullish","support","upward","pe writing","put sum","itm ratio","covering"]) else \
                       "🔴" if any(w in rl for w in ["bearish","resistance","downward","ce writing","call sum"]) else "🟡"
                st.markdown(icon + " " + r)
            st.markdown("---")
            st.dataframe(pd.DataFrame({
                "Level": ["Spot Price","Max Pain","CE Resistance (OI)","PE Support (OI)"],
                "Price": [underlying, max_pain, sig["max_ce_resistance"], sig["max_pe_support"]],
            }), hide_index=True, use_container_width=True)
            strat_note = sig.get("strategy_note", "")
            if strat_note:
                strat_c = "#4CAF50" if "condor" in strat_note.lower() or "spread" in strat_note.lower() else \
                          "#26a69a" if "buy" in strat_note.lower() else "#FF9800"
                st.markdown(
                    '<div style="background:#1e1e2e;padding:10px;border-radius:8px;border-left:4px solid {};">'
                    '<b style="color:{};">{}</b><br>'
                    '<span style="color:#ccc;font-size:13px;">{}</span>'
                    '</div>'.format(strat_c, strat_c, strat_type, strat_note),
                    unsafe_allow_html=True
                )

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION C — INTRADAY TECHNICAL CHART (MACD + RSI + EMA + VWAP)
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 📈 Intraday Technical Chart")

    # Intraday timeframe selector for the chart
    chart_tf = st.radio(
        "Chart Interval", ["5min", "10min", "15min", "30min", "1hr"],
        index=2, horizontal=True, key="live_chart_tf",
        help="Fetch intraday candles at this interval for MACD/RSI/VWAP/EMA chart"
    )
    if chart_tf != "15min":
        chart_raw  = _get_intraday_candles_cached(symbol, chart_tf, data_source)
        chart_tech = compute_technicals(chart_raw) if not chart_raw.empty else intra_tech
    else:
        chart_raw  = intra_raw
        chart_tech = intra_tech

    if chart_tech is not None and not chart_tech.empty and len(chart_tech) >= 5:
        from plotly.subplots import make_subplots as _msp
        xc = chart_tech["datetime"].astype(str)
        fig_intra = _msp(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.48, 0.18, 0.18, 0.16],
            vertical_spacing=0.025,
            subplot_titles=[
                "Price · VWAP (orange) · EMA9 (blue) · EMA20 (amber)",
                "MACD (12,26,9)",
                "RSI 14",
                "Net Volume",
            ],
        )
        # Panel 1 — Candles + indicators
        fig_intra.add_trace(go.Candlestick(
            x=xc, open=chart_tech["open"], high=chart_tech["high"],
            low=chart_tech["low"], close=chart_tech["close"], name="OHLC",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ), row=1, col=1)
        fig_intra.add_trace(go.Scatter(x=xc, y=chart_tech["vwap"], name="VWAP",
            line=dict(color="#FF6F00", width=1.5, dash="dot")), row=1, col=1)
        fig_intra.add_trace(go.Scatter(x=xc, y=chart_tech["ema9"],  name="EMA9",
            line=dict(color="#42A5F5", width=1.5)), row=1, col=1)
        fig_intra.add_trace(go.Scatter(x=xc, y=chart_tech["ema20"], name="EMA20",
            line=dict(color="#FFA726", width=1.5)), row=1, col=1)
        # Panel 2 — MACD
        hist_col = ["#26a69a" if v >= 0 else "#ef5350" for v in chart_tech["macd_hist"]]
        fig_intra.add_trace(go.Bar(x=xc, y=chart_tech["macd_hist"], name="MACD Hist",
            marker_color=hist_col, opacity=0.7), row=2, col=1)
        fig_intra.add_trace(go.Scatter(x=xc, y=chart_tech["macd"],     name="MACD",
            line=dict(color="#42A5F5", width=1.5)), row=2, col=1)
        fig_intra.add_trace(go.Scatter(x=xc, y=chart_tech["macd_sig"], name="Signal",
            line=dict(color="#FFA726", width=1.5)), row=2, col=1)
        fig_intra.add_hline(y=0, line_dash="dash", line_color="#555", row=2, col=1)
        # Panel 3 — RSI
        fig_intra.add_trace(go.Scatter(x=xc, y=chart_tech["rsi"], name="RSI 14",
            line=dict(color="#CE93D8", width=1.5),
            fill="tozeroy", fillcolor="rgba(206,147,216,0.08)"), row=3, col=1)
        for lvl, lc in [(70, "#ef5350"), (50, "#888"), (30, "#26a69a")]:
            fig_intra.add_hline(y=lvl, line_dash="dash", line_color=lc,
                                line_width=1, row=3, col=1)
        # Panel 4 — Net Volume
        nv_col = ["#26a69a" if v >= 0 else "#ef5350" for v in chart_tech["net_vol"]]
        fig_intra.add_trace(go.Bar(x=xc, y=chart_tech["net_vol"], name="Net Vol",
            marker_color=nv_col, opacity=0.8), row=4, col=1)

        fig_intra.update_layout(
            height=780, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=10, l=60, r=10),
            legend=dict(orientation="h", y=1.04, x=0, font=dict(size=10)),
            xaxis_rangeslider_visible=False,
        )
        for i in range(1, 5):
            fig_intra.update_xaxes(gridcolor="#333", row=i, col=1)
            fig_intra.update_yaxes(gridcolor="#333", row=i, col=1)
        st.plotly_chart(fig_intra, use_container_width=True)
    else:
        st.info(
            "Intraday candle data unavailable for this interval/source.\n\n"
            "**Angel One (Live)** provides real 5/15/30-min candles. "
            "**yfinance fallback** works for most NSE indices but may be delayed."
        )

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION D — OI CHARTS
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 📊 Open Interest Analysis")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("#### Open Interest Distribution")
        f = go.Figure()
        f.add_trace(go.Bar(x=df["strike"], y=df["ce_oi"]/1000, name="Call OI", marker_color="#ef5350", opacity=0.85))
        f.add_trace(go.Bar(x=df["strike"], y=df["pe_oi"]/1000, name="Put OI",  marker_color="#26a69a", opacity=0.85))
        f.add_vline(x=underlying, line_dash="dash", line_color="white",
                    annotation_text="Spot {}".format(int(underlying)), annotation_font_color="white")
        f.add_vline(x=max_pain, line_dash="dot", line_color="yellow",
                    annotation_text="MaxPain {}".format(int(max_pain)), annotation_font_color="yellow")
        f.update_layout(barmode="group", height=320,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", y=1.1),
                        xaxis=dict(title="Strike", gridcolor="#333"),
                        yaxis=dict(title="OI (thousands)", gridcolor="#333"),
                        margin=dict(t=20, b=10))
        st.plotly_chart(f, use_container_width=True)

    with cc2:
        st.markdown("#### Change in OI")
        f2 = go.Figure()
        f2.add_trace(go.Bar(x=df["strike"], y=df["ce_chg_oi"]/1000, name="CE Chg OI", marker_color="#ef5350", opacity=0.85))
        f2.add_trace(go.Bar(x=df["strike"], y=df["pe_chg_oi"]/1000, name="PE Chg OI", marker_color="#26a69a", opacity=0.85))
        f2.add_vline(x=underlying, line_dash="dash", line_color="white",
                     annotation_text="Spot {}".format(int(underlying)), annotation_font_color="white")
        f2.update_layout(barmode="group", height=320,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         legend=dict(orientation="h", y=1.1),
                         xaxis=dict(title="Strike", gridcolor="#333"),
                         yaxis=dict(title="Chg OI (thousands)", gridcolor="#333"),
                         margin=dict(t=20, b=10))
        st.plotly_chart(f2, use_container_width=True)

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("#### Implied Volatility Smile")
        iv_df = df[(df["ce_iv"] > 0) | (df["pe_iv"] > 0)]
        fi    = go.Figure()
        if not iv_df.empty:
            fi.add_trace(go.Scatter(x=iv_df["strike"], y=iv_df["ce_iv"], mode="lines+markers",
                                    name="CE IV", line=dict(color="#ef5350", width=2)))
            fi.add_trace(go.Scatter(x=iv_df["strike"], y=iv_df["pe_iv"], mode="lines+markers",
                                    name="PE IV", line=dict(color="#26a69a", width=2)))
            fi.add_vline(x=underlying, line_dash="dash", line_color="white")
        fi.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         legend=dict(orientation="h", y=1.1),
                         xaxis=dict(title="Strike", gridcolor="#333"),
                         yaxis=dict(title="IV %", gridcolor="#333"),
                         margin=dict(t=20, b=10))
        st.plotly_chart(fi, use_container_width=True)

    with ic2:
        st.markdown("#### PCR Interpretation")
        if   pcr > 1.5: interp, pcr_col = "EXTREMELY BULLISH",           "#00e676"
        elif pcr > 1.2: interp, pcr_col = "BULLISH — strong put writing", "#4CAF50"
        elif pcr > 0.8: interp, pcr_col = "NEUTRAL to BULLISH",           "#8BC34A"
        elif pcr > 0.5: interp, pcr_col = "NEUTRAL to BEARISH",           "#FF9800"
        else:           interp, pcr_col = "BEARISH — heavy call writing",  "#f44336"
        st.markdown(
            '<div style="background:#1e1e2e;padding:20px;border-radius:12px;">'
            '<div style="font-size:48px;font-weight:bold;color:{};text-align:center;">{}</div>'
            '<div style="font-size:16px;color:{};text-align:center;margin-top:8px;">{}</div>'
            '<hr style="border-color:#333;margin:15px 0;">'
            '<div style="font-size:13px;color:#aaa;">'
            'PCR &lt;0.5=Bearish | 0.5–0.8=Neutral-Bearish<br>'
            'PCR 0.8–1.2=Neutral | 1.2–1.5=Bullish | &gt;1.5=Extremely Bullish'
            '</div></div>'.format(pcr_col, pcr, pcr_col, interp),
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown("### Tomorrow's Prediction")

    atm_iv = float(df.iloc[(df["strike"] - underlying).abs().idxmin()]["ce_iv"])
    if atm_iv == 0:
        atm_iv = float(df["ce_iv"][df["ce_iv"] > 0].mean()) if (df["ce_iv"] > 0).any() else 15.0

    daily_move  = underlying * (atm_iv / 100) * math.sqrt(1 / 252)
    upper_level = underlying + daily_move
    lower_level = underlying - daily_move
    pain_diff   = max_pain - underlying
    bias_score  = sig["score"]
    if bias_score > 20:
        direction, dir_color = "BULLISH", "#4CAF50"
    elif bias_score < -20:
        direction, dir_color = "BEARISH", "#f44336"
    else:
        direction, dir_color = "NEUTRAL", "#FF9800"

    tp1, tp2, tp3 = st.columns(3)
    with tp1:
        st.markdown(
            '<div style="background:#1e1e2e;padding:18px;border-radius:12px;text-align:center;">'
            '<div style="color:#aaa;font-size:13px;">Expected Daily Range (IV={}%)</div>'
            '<div style="font-size:22px;font-weight:bold;color:#26a69a;margin:8px 0;">±₹{:,.0f}</div>'
            '<div style="color:#ccc;font-size:14px;">Upper: <b>₹{:,.0f}</b></div>'
            '<div style="color:#ccc;font-size:14px;">Lower: <b>₹{:,.0f}</b></div>'
            '</div>'.format(round(atm_iv, 1), daily_move, upper_level, lower_level),
            unsafe_allow_html=True
        )
    with tp2:
        st.markdown(
            '<div style="background:#1e1e2e;padding:18px;border-radius:12px;text-align:center;">'
            '<div style="color:#aaa;font-size:13px;">Directional Bias</div>'
            '<div style="font-size:32px;font-weight:bold;color:{};margin:8px 0;">{}</div>'
            '<div style="color:#ccc;font-size:13px;">Score: {}</div>'
            '<div style="color:#ccc;font-size:13px;">PCR: {}</div>'
            '</div>'.format(dir_color, direction, bias_score, pcr),
            unsafe_allow_html=True
        )
    with tp3:
        mp_dir  = "above" if pain_diff > 0 else "below"
        mp_pull = "Upward pull" if pain_diff > 0 else "Downward pull"
        st.markdown(
            '<div style="background:#1e1e2e;padding:18px;border-radius:12px;text-align:center;">'
            '<div style="color:#aaa;font-size:13px;">Max Pain Magnet</div>'
            '<div style="font-size:22px;font-weight:bold;color:#FFD700;margin:8px 0;">₹{:,.0f}</div>'
            '<div style="color:#ccc;font-size:13px;">₹{:,.0f} {} spot</div>'
            '<div style="color:#ccc;font-size:13px;">{}</div>'
            '</div>'.format(max_pain, abs(pain_diff), mp_dir, mp_pull),
            unsafe_allow_html=True
        )

    st.markdown(
        '<div style="background:#1e1e2e;padding:15px;border-radius:10px;margin-top:10px;">'
        '<b style="color:#aaa;">Key Levels for Tomorrow</b><br>'
        '<span style="color:#ef5350;">Resistance: ₹{:,.0f} (max CE OI)  |  Upper: ₹{:,.0f}</span><br>'
        '<span style="color:#26a69a;">Support: ₹{:,.0f} (max PE OI)  |  Lower: ₹{:,.0f}</span><br>'
        '<span style="color:#FFD700;">Max Pain: ₹{:,.0f}  |  Spot: ₹{:,.0f}</span>'
        '</div>'.format(
            sig["max_ce_resistance"], upper_level,
            sig["max_pe_support"],    lower_level,
            max_pain, underlying
        ),
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("#### Full Option Chain Table")
    tdf     = df.copy()
    atm_idx = (tdf["strike"] - underlying).abs().idxmin()
    tdf     = tdf.rename(columns={
        "ce_oi":"CE OI","ce_chg_oi":"CE Chg OI","ce_volume":"CE Vol",
        "ce_iv":"CE IV%","ce_ltp":"CE LTP","strike":"STRIKE",
        "pe_ltp":"PE LTP","pe_iv":"PE IV%","pe_volume":"PE Vol",
        "pe_chg_oi":"PE Chg OI","pe_oi":"PE OI",
    })
    cols = ["CE OI","CE Chg OI","CE Vol","CE IV%","CE LTP","STRIKE","PE LTP","PE IV%","PE Vol","PE Chg OI","PE OI"]
    tdf  = tdf[cols]

    def hl_atm(row):
        return ["background-color:#2d2d00;font-weight:bold"]*len(row) if row.name == atm_idx else [""]*len(row)

    fmt = {
        "CE OI":"{:,.0f}","CE Chg OI":"{:,.0f}","CE Vol":"{:,.0f}",
        "CE IV%":"{:.1f}","CE LTP":"{:.2f}",
        "PE LTP":"{:.2f}","PE IV%":"{:.1f}",
        "PE Vol":"{:,.0f}","PE Chg OI":"{:,.0f}","PE OI":"{:,.0f}",
    }
    st.dataframe(tdf.style.apply(hl_atm, axis=1).format(fmt), use_container_width=True, height=400)

    st.markdown("---")
    st.caption("Data: {}  |  For educational purposes only. Not financial advice.".format(source_tag))


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — BACKTEST & MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.markdown("## Backtest & Model Training")
    st.markdown("Analyse historical signal accuracy and P&L using index OHLCV data.")

    left_bt, right_bt = st.columns([1, 3])

    with left_bt:
        bt_mode = st.radio(
            "Mode",
            ["A — Signal Accuracy", "B — P&L Simulation", "C — Forward Tracker", "Train Model"],
        )
        bt_days = st.slider("Historical Days", 20, 90, 30, step=5)
        if data_source == "Angel One (Live)":
            st.success("Using Angel One historical candles")
        else:
            st.info("Using synthetic candles\n(select Angel One for real data)")

    with right_bt:
        candles = _get_candles(symbol, is_index, bt_days, data_source)

        # ── Mode A: Signal Accuracy ───────────────────────────────────────────
        if bt_mode.startswith("A"):
            st.markdown("### Signal Accuracy Backtest")
            st.markdown(
                "Uses **RSI-14 + EMA-9/EMA-21 crossover + volume ratio** consensus "
                "on {} daily candles. Signals fire only when 2+ indicators agree — "
                "higher selectivity means fewer trades but better quality signals. "
                "**Note:** Synthetic demo data is a random walk; near-50% win rate "
                "on synthetic data is expected. Use Angel One for real results.".format(bt_days)
            )
            if st.button("Run Accuracy Backtest", type="primary", key="run_acc"):
                with st.spinner("Running backtest…"):
                    metrics, rdf = backtest.run_accuracy_backtest(candles)
                if not metrics:
                    st.error("Not enough data (need ≥12 candles).")
                else:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Win Rate",      "{}%".format(metrics["win_rate"]))
                    m2.metric("Call Accuracy", "{}%".format(metrics["call_accuracy"]))
                    m3.metric("Put Accuracy",  "{}%".format(metrics["put_accuracy"]))
                    m4.metric("Total Traded",  str(metrics["total_traded"]))
                    m5.metric("Signal Rate",   "{}%".format(metrics.get("signal_rate", 0)),
                               delta="of candles had signal")

                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=rdf["date"], y=rdf["equity"],
                        mode="lines+markers",
                        line=dict(color="#26a69a", width=2),
                        name="Equity Curve (cumulative wins-losses)"
                    ))
                    fig_eq.update_layout(
                        title="Equity Curve",
                        height=280,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(gridcolor="#333", tickangle=-45),
                        yaxis=dict(gridcolor="#333"),
                        margin=dict(t=40, b=30)
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)
                    st.dataframe(rdf[["date","close","signal","next_chg","correct"]],
                                 use_container_width=True, height=280)

                    backtest.save_forward_signal(
                        symbol, sig["signal"], sig["confidence"],
                        underlying, float(pcr), float(max_pain)
                    )
                    st.success("Today's signal auto-saved to Forward Tracker.")

        # ── Mode B: P&L Simulation ────────────────────────────────────────────
        elif bt_mode.startswith("B"):
            st.markdown("### P&L Simulation (₹)")
            st.markdown(
                "Buys 1 lot ATM option at estimated premium (0.8% of index price). "
                "Win = 1.5× premium, Loss = 0.4× premium (approximate)."
            )
            lot_defaults = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 40,
                            "MIDCPNIFTY": 50, "SENSEX": 20}
            sim_lot  = st.number_input("Lot Size", value=lot_defaults.get(symbol.upper(), 75),
                                        min_value=1, max_value=500)
            sim_comm = st.number_input("Commission per trade (₹)", value=40,
                                        min_value=0, max_value=500)

            if st.button("Run P&L Simulation", type="primary", key="run_pnl"):
                with st.spinner("Simulating P&L…"):
                    metrics, rdf = backtest.run_pnl_simulation(candles, int(sim_lot), int(sim_comm))
                if not metrics:
                    st.error("Not enough data (need ≥12 candles).")
                else:
                    pnl_color = "#26a69a" if metrics["total_pnl"] >= 0 else "#ef5350"
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total P&L", "₹{:,}".format(metrics["total_pnl"]),
                               delta="Profit" if metrics["total_pnl"] >= 0 else "Loss")
                    m2.metric("Win Rate",   "{}%".format(metrics["win_rate"]))
                    m3.metric("Max Drawdown", "₹{:,}".format(metrics["max_drawdown"]))
                    m4.metric("Sharpe",     str(metrics["sharpe_ratio"]))

                    m5, m6, m7 = st.columns(3)
                    m5.metric("Best Trade",  "₹{:,}".format(metrics["best_trade"]))
                    m6.metric("Worst Trade", "₹{:,}".format(metrics["worst_trade"]))
                    m7.metric("Avg P&L",     "₹{:,}".format(int(metrics["avg_pnl"])))

                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Scatter(
                        x=rdf["date"], y=rdf["equity"],
                        mode="lines", fill="tozeroy",
                        line=dict(color=pnl_color, width=2),
                        name="Cumulative P&L (₹)"
                    ))
                    fig_pnl.update_layout(
                        title="Cumulative P&L Curve",
                        height=280,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(gridcolor="#333", tickangle=-45),
                        yaxis=dict(title="₹", gridcolor="#333"),
                        margin=dict(t=40, b=30)
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)
                    st.dataframe(rdf[["date","close","signal","premium","pnl","result","equity"]],
                                 use_container_width=True, height=280)

        # ── Mode C: Forward Tracker ───────────────────────────────────────────
        elif bt_mode.startswith("C"):
            st.markdown("### Forward Paper Signal Tracker")
            st.markdown(
                "Save today's signal → auto-checked against next trading day's close. "
                "Builds a real-time accuracy log over time."
            )

            backtest.update_forward_outcomes(candles)
            fwd_signals = backtest.load_forward_signals()

            save_col, stat_col = st.columns(2)
            with save_col:
                if st.button("Save Today's Signal", type="primary", key="save_fwd"):
                    backtest.save_forward_signal(
                        symbol, sig["signal"], sig["confidence"],
                        underlying, float(pcr), float(max_pain)
                    )
                    st.success("Saved: {} {} @ ₹{:,.0f}".format(
                        sig["signal"], symbol, underlying))
                    st.rerun()

            with stat_col:
                if fwd_signals:
                    fdf_stat = pd.DataFrame(fwd_signals)
                    resolved = fdf_stat[fdf_stat["outcome"] != "PENDING"]
                    total    = len(fwd_signals)
                    if not resolved.empty:
                        acc = round((resolved["outcome"] == "CORRECT").mean() * 100, 1)
                        st.metric("Forward Accuracy", "{}%".format(acc),
                                   delta="{} signals tracked".format(total))
                    else:
                        st.info("{} signals tracked — awaiting resolution".format(total))
                else:
                    st.info("No signals tracked yet.")

            if fwd_signals:
                fdf = pd.DataFrame(fwd_signals)

                def _oc(val):
                    if val == "CORRECT": return "color: #4CAF50"
                    if val == "WRONG":   return "color: #f44336"
                    return "color: #FF9800"

                st.dataframe(
                    fdf.style.applymap(_oc, subset=["outcome"]),
                    use_container_width=True, height=350
                )
            else:
                st.info("Click **Save Today's Signal** above to begin tracking.")

        # ── Mode: Train Model ─────────────────────────────────────────────────
        else:
            st.markdown("### Model Weight Calibration")
            st.markdown(
                "Grid-searches over RSI thresholds, volume confirmation, and minimum "
                "indicator agreement (54 combinations) to find params that maximise "
                "signal accuracy. Works best with **Angel One** live candle data."
            )

            if st.button("Run Grid Search (54 combinations)", type="primary", key="run_calib"):
                with st.spinner("Calibrating — scanning 576 parameter combinations…"):
                    st.session_state["_calib"] = model_train.calibrate_weights(candles)

            calib = st.session_state.get("_calib")
            if calib:
                best     = calib.get("best_params", {})
                best_acc = calib.get("best_accuracy", 0)
                grid_df  = calib.get("accuracy_grid", pd.DataFrame())

                st.success("Best accuracy: **{}%**".format(best_acc))
                st.json(best)

                apply_col, reset_col = st.columns(2)
                if apply_col.button("Apply These Weights", type="primary"):
                    model_train.apply_weights(best)
                    del st.session_state["_calib"]
                    st.success("Weights saved to .model_weights.json. Reload app to use them.")
                    st.rerun()
                if reset_col.button("Reset to Defaults"):
                    wf = os.path.join(os.path.dirname(__file__), ".model_weights.json")
                    if os.path.exists(wf):
                        os.remove(wf)
                    st.session_state.pop("_calib", None)
                    st.success("Weights reset.")
                    st.rerun()

                if not grid_df.empty:
                    st.markdown("#### Top 20 Configurations")
                    st.dataframe(grid_df.head(20), use_container_width=True)

                    fig_acc = go.Figure(go.Bar(
                        x=list(range(min(20, len(grid_df)))),
                        y=grid_df["accuracy"].head(20).tolist(),
                        marker_color="#26a69a",
                    ))
                    fig_acc.update_layout(
                        title="Top-20 Config Accuracy", height=220,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title="Config #", gridcolor="#333"),
                        yaxis=dict(title="%", gridcolor="#333"),
                        margin=dict(t=40, b=10)
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)

            st.divider()
            st.markdown("#### Current Active Weights")
            st.json(backtest.load_weights())


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — PAPER TRADE
# ═════════════════════════════════════════════════════════════════════════════
with tab_pt:
    pt.init_portfolio(st.session_state)
    summary = pt.get_summary(st.session_state, df, underlying)

    st.markdown("## Paper Trading  —  Virtual ₹1,00,000 Capital")
    st.caption("All trades are simulated. No real money involved. "
               "Trades persist within a session and are saved locally.")

    # Portfolio metrics
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Available Capital", "₹{:,.0f}".format(summary["capital"]))
    pc2.metric("Realised P&L",
               "₹{:,.0f}".format(summary["realized_pnl"]),
               delta="+" + str(summary["realized_pnl"]) if summary["realized_pnl"] >= 0
                     else str(summary["realized_pnl"]))
    pc3.metric("Unrealised P&L",    "₹{:,.0f}".format(summary["unrealized_pnl"]))
    pc4.metric("Total Return",
               "{}%".format(summary["total_return_pct"]),
               delta="{:+,.0f}".format(summary["total_pnl"]))

    st.divider()

    # ── Place Trade ───────────────────────────────────────────────────────────
    st.markdown("### Place New Paper Trade")

    with st.form("place_trade_form"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            pt_sym  = st.selectbox("Symbol", list(INDICES.keys()), index=0)
            pt_type = st.radio("Type", ["BUY CALL", "BUY PUT"])
        with fc2:
            def_strike = float(int(meta.get("atm", underlying) / 50) * 50)
            pt_strike  = st.number_input("Strike Price ₹", value=def_strike,
                                          step=50.0, format="%.1f")
            pt_expiry  = st.text_input("Expiry", value=expiry)
        with fc3:
            pt_lots  = st.number_input("Lots", value=1, min_value=1, max_value=50)
            atm_row  = df.iloc[(df["strike"] - underlying).abs().idxmin()] if not df.empty else None
            if atm_row is not None:
                sugg = float(atm_row["ce_ltp"] if pt_type == "BUY CALL" else atm_row["pe_ltp"])
                sugg = max(sugg, 1.0)
            else:
                sugg = max(round(float(underlying) * 0.008, 2), 1.0)
            pt_entry = st.number_input("Entry Premium ₹/unit", value=round(sugg, 2),
                                        min_value=0.05, step=0.5, format="%.2f")

        if st.form_submit_button("Place Trade", type="primary", use_container_width=True):
            ok, msg = pt.place_trade(
                st.session_state, pt_sym, pt_type,
                float(pt_strike), pt_expiry, int(pt_lots), float(pt_entry)
            )
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.divider()

    # ── Open Trades ───────────────────────────────────────────────────────────
    open_trades = summary["open_trades"]
    st.markdown("### Open Trades ({})".format(len(open_trades)))

    if open_trades:
        open_df = pd.DataFrame([{
            "ID":       t["id"],
            "Symbol":   t["symbol"],
            "Type":     t["type"],
            "Strike":   t["strike"],
            "Expiry":   t["expiry"],
            "Lots":     t["lots"],
            "Entry ₹":  t["entry_price"],
            "Cost ₹":   round(t["entry_price"] * t["lots"] * t["lot_size"], 2),
            "Opened":   t["entry_time"],
        } for t in open_trades])
        st.dataframe(open_df, use_container_width=True, hide_index=True)

        st.markdown("**Close a Trade:**")
        with st.form("close_trade_form"):
            trade_ids    = [t["id"] for t in open_trades]
            trade_labels = [
                "#{} {} {} @{}".format(t["id"], t["type"], t["symbol"], int(t["strike"]))
                for t in open_trades
            ]
            cl1, cl2 = st.columns(2)
            with cl1:
                close_sel = st.selectbox("Select Trade", trade_labels)
                close_tid = trade_ids[trade_labels.index(close_sel)]
            with cl2:
                close_exit = st.number_input("Exit Premium ₹/unit", value=1.0,
                                              min_value=0.05, step=0.5, format="%.2f")
            if st.form_submit_button("Close Trade", type="primary", use_container_width=True):
                ok, msg = pt.close_trade(st.session_state, close_tid, float(close_exit))
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    else:
        st.info("No open trades. Place a trade above.")

    # ── Closed Trades ─────────────────────────────────────────────────────────
    closed_trades = summary["closed_trades"]
    st.markdown("### Closed Trades ({})".format(len(closed_trades)))

    if closed_trades:
        closed_df = pd.DataFrame([{
            "ID":      t["id"],
            "Symbol":  t["symbol"],
            "Type":    t["type"],
            "Strike":  t["strike"],
            "Lots":    t["lots"],
            "Entry ₹": t["entry_price"],
            "Exit ₹":  t["exit_price"],
            "P&L ₹":   t["pnl"],
            "Exited":  t["exit_time"],
        } for t in closed_trades])

        def _pnl_color(val):
            if val > 0:   return "color: #4CAF50"
            if val < 0:   return "color: #f44336"
            return "color: #888"

        st.dataframe(
            closed_df.style.applymap(_pnl_color, subset=["P&L ₹"]),
            use_container_width=True, hide_index=True
        )

        csv_data = pt.export_to_csv(st.session_state)
        if csv_data:
            st.download_button(
                "Download Trade History CSV",
                data=csv_data,
                file_name="paper_trades_{}.csv".format(datetime.now().strftime("%Y%m%d")),
                mime="text/csv",
            )
    else:
        st.info("No closed trades yet.")

    # ── Reset ─────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Reset Portfolio")
    if "pt_confirm_reset" not in st.session_state:
        st.session_state.pt_confirm_reset = False

    if not st.session_state.pt_confirm_reset:
        if st.button("Reset Portfolio to ₹1,00,000", type="secondary"):
            st.session_state.pt_confirm_reset = True
            st.rerun()
    else:
        st.warning("This will clear ALL trades and reset capital. This cannot be undone.")
        yes_col, no_col = st.columns(2)
        if yes_col.button("Yes, Reset Everything", type="primary"):
            pt.reset_portfolio(st.session_state)
            st.session_state.pt_confirm_reset = False
            st.success("Portfolio reset to ₹1,00,000!")
            st.rerun()
        if no_col.button("Cancel"):
            st.session_state.pt_confirm_reset = False
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — CHAT
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("## Chat with the Data")
    st.markdown(
        "Discuss the trade before you place it — ask for a **Trade Decision** or run "
        "the **Scalping Checklist**, then drill into any detail."
    )

    try:
        _api_key_present = bool(st.secrets.get("anthropic_api_key", ""))
    except Exception:
        _api_key_present = False

    if _api_key_present:
        st.success("Claude AI (claude-haiku-4-5-20251001) enabled")
    else:
        st.info("Keyword mode active — add `anthropic_api_key` to Streamlit secrets for AI answers")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Timeframe selector ────────────────────────────────────────────────────
    st.markdown("#### Select Timeframe")
    tf_options = {
        "5-min":  "5-min Scalp",
        "15-min": "15-min Intraday",
        "30-min": "30-min Swing",
        "1-hr":   "1-hr Intraday",
        "2-hr":   "2-hr Positional",
        "daily":  "Daily Swing",
    }
    selected_tf = st.radio(
        "Timeframe",
        list(tf_options.keys()),
        format_func=lambda k: tf_options[k],
        index=3,  # default 1-hr
        horizontal=True,
        key="chat_tf",
        label_visibility="collapsed",
    )

    tf_info = {
        "5-min":  "10-15 min hold · SL 20% · Target 30% · Best: 9:20-10:00 AM",
        "15-min": "45-60 min hold · SL 30% · Target 50% · Best: 9:30-11:30 AM",
        "30-min": "1.5-2 hr hold  · SL 35% · Target 60% · Best: 9:45 AM-12:00 PM",
        "1-hr":   "2-3 hr hold    · SL 40% · Target 65% · Best: 9:30-11:30 AM",
        "2-hr":   "1-day hold     · SL 45% · Target 80% · Use credit spreads",
        "daily":  "2-3 day hold   · SL 50% · Target 100% · Use next-week expiry",
    }
    st.caption("⏱ " + tf_info[selected_tf])

    # ── Primary action buttons ────────────────────────────────────────────────
    st.markdown("#### Start here:")
    pa1, pa2 = st.columns(2)
    if pa1.button(
        "🎯 Get Trade Decision",
        use_container_width=True,
        type="primary",
        key="pa_decision",
        help="Full BUY/AVOID analysis with entry, stop-loss, and target",
    ):
        q = "trade decision"
        st.session_state.chat_history.append({"role": "user", "content": q})
        a = chat_bot.answer(q, df, meta, sig, timeframe=selected_tf)
        st.session_state.chat_history.append({"role": "assistant", "content": a})
        st.rerun()

    checklist_label = "📋 {} Checklist".format(tf_options[selected_tf])
    if pa2.button(
        checklist_label,
        use_container_width=True,
        type="secondary",
        key="pa_scalp",
        help="Dynamic pass/fail checklist for {}".format(tf_options[selected_tf]),
    ):
        q = "checklist"
        st.session_state.chat_history.append({"role": "user", "content": q})
        a = chat_bot.answer(q, df, meta, sig, timeframe=selected_tf)
        st.session_state.chat_history.append({"role": "assistant", "content": a})
        st.rerun()

    st.divider()

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(
        "Ask: trade decision · checklist · signal · pcr · IV · tomorrow · help"
    )
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = chat_bot.answer(user_input, df, meta, sig,
                                         timeframe=selected_tf)
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.markdown("---")
    st.markdown("**More quick questions:**")
    quick_qs = [
        ("Signal?",     "What is the current trade signal?"),
        ("PCR?",        "What is the put call ratio?"),
        ("Tomorrow?",   "What is tomorrow's prediction?"),
        ("Key Levels?", "What are the support and resistance levels?"),
        ("IV?",         "What is the implied volatility?"),
        ("OI?",         "Show me the open interest summary"),
    ]
    q_cols = st.columns(len(quick_qs))
    for i, (label, question) in enumerate(quick_qs):
        if q_cols[i].button(label, use_container_width=True, key="qq_{}".format(i)):
            st.session_state.chat_history.append({"role": "user", "content": question})
            answer = chat_bot.answer(question, df, meta, sig, timeframe=selected_tf)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — INDEX ANALYSIS  (VWAP · EMA9/20 · RSI · PVT · Net Volume)
# ═════════════════════════════════════════════════════════════════════════════
with tab_idx:
    from plotly.subplots import make_subplots

    st.markdown("## 🇮🇳 Indian Index Technical Analysis")
    st.caption(
        "VWAP · EMA-9 · EMA-20 · RSI-14 · PVT · Net Volume  |  "
        "Data: Angel One (live) → yfinance (fallback) → Demo"
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        idx_opts      = ["{} — {}".format(k, v) for k, v in INDICES.items()]
        idx_sel       = st.selectbox("Select Index", idx_opts, key="ia_idx")
        ia_symbol     = idx_sel.split(" — ")[0]
    with ctrl2:
        ia_days = st.slider("History (days)", 15, 90, 30, step=5, key="ia_days")
    with ctrl3:
        ia_fno_note = "✅ F&O index" if ia_symbol in FNO_INDICES else "ℹ️ Spot-only index"
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(ia_fno_note)

    # ── Fetch & compute ───────────────────────────────────────────────────────
    with st.spinner("Fetching {} candles…".format(ia_symbol)):
        ia_candles = _get_candles(ia_symbol, True, ia_days, data_source)

    if ia_candles is None or ia_candles.empty:
        st.warning("No candle data available for {}. Try switching to Angel One.".format(ia_symbol))
    else:
        ia_tech = compute_technicals(ia_candles)
        sig_str, sig_col, sig_score, sig_reasons = tech_signal(ia_tech)
        latest  = ia_tech.iloc[-1]
        prev    = ia_tech.iloc[-2] if len(ia_tech) > 1 else latest
        chg_pct = (latest["close"] - prev["close"]) / prev["close"] * 100

        # ── Signal header ─────────────────────────────────────────────────────
        h1, h2, h3, h4, h5, h6 = st.columns(6)
        h1.metric("Spot", "₹{:,.2f}".format(latest["close"]),
                  delta="{:+.2f}%".format(chg_pct))
        h2.metric("VWAP",  "₹{:,.2f}".format(latest["vwap"]))
        h3.metric("EMA 9", "₹{:,.2f}".format(latest["ema9"]))
        h4.metric("EMA 20","₹{:,.2f}".format(latest["ema20"]))
        h5.metric("RSI 14","{:.1f}".format(latest["rsi"]),
                  delta="Overbought" if latest["rsi"] > 70 else
                        ("Oversold" if latest["rsi"] < 30 else "Neutral"))
        h6.metric("Signal", sig_str)

        st.divider()

        # ── 4-panel technical chart ───────────────────────────────────────────
        fig_tech = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            row_heights=[0.50, 0.18, 0.16, 0.16],
            vertical_spacing=0.03,
            subplot_titles=[
                "{} — Candles · VWAP · EMA9 · EMA20".format(ia_symbol),
                "RSI 14",
                "PVT (Price Volume Trend)",
                "Net Volume",
            ],
        )

        x = ia_tech["datetime"].astype(str)

        # — Panel 1: Candlestick + indicators —
        fig_tech.add_trace(go.Candlestick(
            x=x,
            open=ia_tech["open"], high=ia_tech["high"],
            low=ia_tech["low"],  close=ia_tech["close"],
            name="OHLC",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ), row=1, col=1)
        fig_tech.add_trace(go.Scatter(
            x=x, y=ia_tech["vwap"], name="VWAP",
            line=dict(color="#FF6F00", width=2, dash="dot"),
        ), row=1, col=1)
        fig_tech.add_trace(go.Scatter(
            x=x, y=ia_tech["ema9"], name="EMA 9",
            line=dict(color="#42A5F5", width=1.5),
        ), row=1, col=1)
        fig_tech.add_trace(go.Scatter(
            x=x, y=ia_tech["ema20"], name="EMA 20",
            line=dict(color="#FFA726", width=1.5),
        ), row=1, col=1)

        # — Panel 2: RSI —
        fig_tech.add_trace(go.Scatter(
            x=x, y=ia_tech["rsi"], name="RSI 14",
            line=dict(color="#CE93D8", width=1.5),
            fill="tozeroy", fillcolor="rgba(206,147,216,0.08)",
        ), row=2, col=1)
        for level, lcolor in [(70, "#ef5350"), (50, "#888"), (30, "#26a69a")]:
            fig_tech.add_hline(y=level, line_dash="dash", line_color=lcolor,
                               line_width=1, row=2, col=1)

        # — Panel 3: PVT —
        pvt_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in ia_tech["pvt"]]
        fig_tech.add_trace(go.Bar(
            x=x, y=ia_tech["pvt"], name="PVT",
            marker_color=pvt_colors, opacity=0.85,
        ), row=3, col=1)

        # — Panel 4: Net Volume —
        nv_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in ia_tech["net_vol"]]
        fig_tech.add_trace(go.Bar(
            x=x, y=ia_tech["net_vol"], name="Net Vol",
            marker_color=nv_colors, opacity=0.85,
        ), row=4, col=1)

        fig_tech.update_layout(
            height=820,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=60, b=20, l=60, r=20),
            legend=dict(orientation="h", y=1.04, x=0),
            xaxis_rangeslider_visible=False,
            font=dict(size=11),
        )
        for i in range(1, 5):
            fig_tech.update_xaxes(gridcolor="#333", row=i, col=1)
            fig_tech.update_yaxes(gridcolor="#333", row=i, col=1)

        st.plotly_chart(fig_tech, use_container_width=True)

        # ── Signal breakdown ──────────────────────────────────────────────────
        st.divider()
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            bg_map  = {"green":"#1a7a1a","#8BC34A":"#2a4a10","orange":"#7a5c00",
                       "#ef5350":"#5a1010","red":"#4a0808"}
            brd_map = {"green":"#4CAF50","#8BC34A":"#8BC34A","orange":"#FF9800",
                       "#ef5350":"#ef5350","red":"#f44336"}
            bg  = bg_map.get(sig_col, "#333")
            brd = brd_map.get(sig_col, "#888")
            st.markdown(
                '<div style="background:{};border:3px solid {};padding:24px;'
                'border-radius:14px;text-align:center;">'
                '<div style="font-size:13px;color:#ccc;">TECHNICAL SIGNAL</div>'
                '<div style="font-size:26px;font-weight:bold;color:{};margin:8px 0;">{}</div>'
                '<div style="font-size:14px;color:#ddd;">Score: {:+d} / 5</div>'
                '</div>'.format(bg, brd, brd, sig_str, sig_score),
                unsafe_allow_html=True,
            )
        with sc2:
            st.markdown("#### Indicator Breakdown")
            for r in sig_reasons:
                st.markdown(r)

        # ── Index Overview table ──────────────────────────────────────────────
        st.divider()
        st.markdown("### 📋 All-Index Overview")
        st.caption("Fetches last 30 days for each index. May take a few seconds.")

        if st.button("Load All-Index Snapshot", key="ia_overview_btn"):
            rows = []
            prog = st.progress(0)
            idx_list = list(INDICES.items())
            for i, (sym, name) in enumerate(idx_list):
                prog.progress((i + 1) / len(idx_list), text=sym)
                try:
                    c = _get_candles(sym, True, 30, data_source)
                    if c is not None and not c.empty:
                        td = compute_technicals(c)
                        s, sc, ss, _ = tech_signal(td)
                        lat = td.iloc[-1]
                        prv = td.iloc[-2] if len(td) > 1 else lat
                        chg = (lat["close"] - prv["close"]) / prv["close"] * 100
                        rows.append({
                            "Symbol": sym,
                            "Name":   name,
                            "Spot ₹": round(lat["close"], 2),
                            "Chg %":  round(chg, 2),
                            "EMA9":   round(lat["ema9"], 2),
                            "EMA20":  round(lat["ema20"], 2),
                            "VWAP":   round(lat["vwap"], 2),
                            "RSI":    round(lat["rsi"], 1),
                            "Signal": s,
                            "Score":  ss,
                            "F&O":    "✅" if sym in FNO_INDICES else "",
                        })
                except Exception:
                    pass
            prog.empty()

            if rows:
                ov_df = pd.DataFrame(rows)

                def _sig_color(val):
                    if "BULLISH" in val:  return "color:#26a69a;font-weight:bold"
                    if "BEARISH" in val:  return "color:#ef5350;font-weight:bold"
                    return "color:#FF9800"

                def _chg_color(val):
                    return "color:#26a69a" if val >= 0 else "color:#ef5350"

                st.dataframe(
                    ov_df.style
                        .applymap(_sig_color, subset=["Signal"])
                        .applymap(_chg_color, subset=["Chg %"])
                        .format({"Spot ₹": "{:,.2f}", "EMA9": "{:,.2f}",
                                 "EMA20": "{:,.2f}", "VWAP": "{:,.2f}",
                                 "RSI": "{:.1f}", "Chg %": "{:+.2f}%"}),
                    use_container_width=True,
                    hide_index=True,
                    height=560,
                )

                # Quick bar chart — RSI across indices
                fig_rsi = go.Figure(go.Bar(
                    x=ov_df["Symbol"],
                    y=ov_df["RSI"],
                    marker_color=[
                        "#ef5350" if v > 70 else "#26a69a" if v < 30 else "#42A5F5"
                        for v in ov_df["RSI"]
                    ],
                    name="RSI 14",
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef5350", line_width=1)
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="#26a69a", line_width=1)
                fig_rsi.add_hline(y=50, line_dash="dot",  line_color="#888",    line_width=1)
                fig_rsi.update_layout(
                    title="RSI 14 Across All Indian Indices",
                    height=320,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="#333", tickangle=-45),
                    yaxis=dict(gridcolor="#333", range=[0, 100]),
                    margin=dict(t=40, b=60),
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
            else:
                st.warning("Could not fetch data for any index. Check data source.")


# ── Auto Refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()

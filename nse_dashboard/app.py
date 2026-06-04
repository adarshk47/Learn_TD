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
from stock_list import NSE_STOCKS, INDICES, INDEX_YF_SYMBOLS, search_stocks
import paper_trade as pt
import chat_bot
import backtest
import model_train
import scanner as sc
import expiry_analysis as ea
import smart_analysis as sa
import oi_tracker as oit

APP_VERSION = "v2.3"

st.set_page_config(
    page_title="NSE Options Intelligence {}".format(APP_VERSION),
    page_icon="ðŸ"ˆ",
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
        index_options = ["{} — {}".format(k, v) for k, v in INDICES.items()]
        selected_idx  = st.selectbox("Select Index", index_options)
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

    num_strikes  = st.slider("Strikes around ATM", 10, 40, 20, step=2)
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
        st.info("Try during market hours (9:15 AM " 3:30 PM IST) on Indian internet.")
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
def _get_angelone_data(sym, idx, strikes):
    from angelone_data import fetch_option_chain_angelone
    return fetch_option_chain_angelone(sym, idx, strikes)

@st.cache_data(ttl=300)
def _get_candles(sym, idx, days, src):
    # 1. Angel One live
    if src == "Angel One (Live)":
        try:
            from angelone_data import fetch_historical_candles
            c = fetch_historical_candles(sym, idx, days)
            if c is not None and not c.empty:
                return c
        except Exception:
            pass
    # 2. yfinance fallback (works for all NSE/BSE indices)
    yf_sym = INDEX_YF_SYMBOLS.get(sym.upper())
    if yf_sym:
        try:
            import yfinance as yf
            hist = yf.Ticker(yf_sym).history(period="{}d".format(days + 15))
            if not hist.empty:
                hist = hist.reset_index()
                dc   = "Date" if "Date" in hist.columns else "Datetime"
                return pd.DataFrame({
                    "datetime": pd.to_datetime(hist[dc]).dt.tz_localize(None),
                    "open":  hist["Open"].astype(float),
                    "high":  hist["High"].astype(float),
                    "low":   hist["Low"].astype(float),
                    "close": hist["Close"].astype(float),
                    "volume":hist["Volume"].astype(float),
                }).tail(days).reset_index(drop=True)
        except Exception:
            pass
    # 3. Synthetic fallback
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
def _get_intraday(sym, tf, src):
    """Fetch intraday candles: Angel One → yfinance → empty."""
    if src == "Angel One (Live)":
        try:
            from angelone_data import fetch_intraday_candles
            c = fetch_intraday_candles(sym, True, tf)
            if c is not None and not c.empty:
                return c
        except Exception:
            pass
    yf_sym = INDEX_YF_SYMBOLS.get(sym.upper())
    if yf_sym:
        try:
            import yfinance as yf
            yf_iv = {"5min":"5m","10min":"10m","15min":"15m","30min":"30m","1hr":"60m"}.get(tf,"15m")
            yf_pd = {"5min":"1d","10min":"2d","15min":"5d","30min":"10d","1hr":"20d"}.get(tf,"5d")
            hist  = yf.Ticker(yf_sym).history(period=yf_pd, interval=yf_iv)
            if not hist.empty:
                hist = hist.reset_index()
                dc   = "Datetime" if "Datetime" in hist.columns else "Date"
                return pd.DataFrame({
                    "datetime": pd.to_datetime(hist[dc]).dt.tz_localize(None),
                    "open":  hist["Open"].astype(float),
                    "high":  hist["High"].astype(float),
                    "low":   hist["Low"].astype(float),
                    "close": hist["Close"].astype(float),
                    "volume":hist["Volume"].astype(float),
                }).dropna(subset=["close"]).reset_index(drop=True)
        except Exception:
            pass
    return pd.DataFrame()


def _compute_tech(df):
    """VWAP, EMA9/20, MACD, RSI14, PVT, Net Volume from OHLCV DataFrame."""
    d = df.copy().reset_index(drop=True)
    hlc3        = (d["high"] + d["low"] + d["close"]) / 3
    cum_vol     = d["volume"].cumsum().replace(0, np.nan)
    d["vwap"]   = (hlc3 * d["volume"]).cumsum() / cum_vol
    d["ema9"]   = d["close"].ewm(span=9,  adjust=False).mean()
    d["ema20"]  = d["close"].ewm(span=20, adjust=False).mean()
    ema12       = d["close"].ewm(span=12, adjust=False).mean()
    ema26       = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]
    delta       = d["close"].diff()
    ag          = delta.clip(lower=0).ewm(com=13, min_periods=1).mean()
    al          = (-delta).clip(lower=0).ewm(com=13, min_periods=1).mean()
    d["rsi"]    = 100 - 100 / (1 + ag / (al + 1e-9))
    pct         = d["close"].pct_change().fillna(0)
    d["pvt"]    = (pct * d["volume"]).cumsum()
    d["net_vol"]= d["volume"] * np.sign(d["close"].diff().fillna(0))
    return d

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
        df, meta = _get_angelone_data(symbol, is_index, num_strikes)
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

# ── Persist OI snapshot on every fetch (survives browser refresh) ─────────────
try:
    oit.record_snapshot(
        df, underlying, pcr,
        call_sum_atm=sig.get("call_sum", 0) * 1000,   # tracker stores raw units
        put_sum_atm =sig.get("put_sum",  0) * 1000,
        symbol=symbol,
    )
except Exception:
    pass

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_scan, tab_exp, tab_bt, tab_pt, tab_chat = st.tabs(
    ["Live Dashboard", "Market Scanner", "Expiry Signals",
     "Backtest", "Paper Trade", "Chat"]
)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 1 — LIVE DASHBOARD
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_live:
    st.markdown("## {} Option Chain  —  Expiry: **{}**".format(symbol, expiry))
    st.markdown("**Spot: ₹{:,.2f}**  |  ATM: **{}**  |  Source: `{}`".format(
        underlying, meta["atm"], source_tag))
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spot Price",    "₹{:,.1f}".format(underlying))
    c2.metric("PCR",           str(pcr), delta="Bullish" if pcr > 1.0 else "Bearish")
    c3.metric("Max Pain",      "₹{:,.0f}".format(max_pain))
    c4.metric("CE Resistance", "₹{:,.0f}".format(sig["max_ce_resistance"]), delta="Sell Wall")
    c5.metric("PE Support",    "₹{:,.0f}".format(sig["max_pe_support"]),    delta="Buy Wall")

    # ── OI Intelligence row (from VarunS2002 3-strike sum + ITM ratio research) ─
    oi1, oi2, oi3, oi4, oi5 = st.columns(5)
    call_sum_v = sig.get("call_sum", 0)
    put_sum_v  = sig.get("put_sum", 0)
    oi_diff_v  = sig.get("oi_difference", 0)
    itm_r      = sig.get("itm_ratio", 0)
    oi1.metric("Call Sum (ATM±1)", "{:+.1f}K".format(call_sum_v),
               delta="CE writing â†'" if call_sum_v > 0 else "CE covering â†"")
    oi2.metric("Put Sum (ATM±1)", "{:+.1f}K".format(put_sum_v),
               delta="PE writing â†'" if put_sum_v > 0 else "PE covering â†"")
    oi3.metric("OI Difference", "{:+.1f}K".format(oi_diff_v),
               delta="Bearish" if oi_diff_v > 0 else "Bullish")
    oi4.metric("ITM Ratio", "{:.2f}x".format(itm_r),
               delta="Bullish" if itm_r > 1.5 else ("Bearish" if itm_r < 0.67 and itm_r > 0 else "Neutral"))
    strat_type = sig.get("strategy_type", "WAIT")
    oi5.metric("Suggested Strategy", strat_type)
    st.divider()

    sig_col, reason_col = st.columns([1, 2])
    with sig_col:
        bg     = {"green": "#1a7a1a", "red": "#8b1a1a", "orange": "#7a5c00"}.get(sig["color"], "#333")
        border = {"green": "#4CAF50", "red": "#f44336", "orange": "#FF9800"}.get(sig["color"], "#888")
        st.markdown(
            '<div style="background:{};border:3px solid {};padding:25px;border-radius:14px;text-align:center;">'
            '<div style="font-size:14px;color:#ccc;margin-bottom:6px;">TRADE SIGNAL</div>'
            '<div style="font-size:32px;font-weight:bold;color:{};">{}</div>'
            '<div style="font-size:16px;color:#ddd;margin-top:8px;">Confidence</div>'
            '<div style="font-size:42px;font-weight:bold;color:white;">{}%</div>'
            '</div>'.format(bg, border, border, sig["signal"], sig["confidence"]),
            unsafe_allow_html=True
        )
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=sig["confidence"],
            title={"text": "Signal Strength", "font": {"size": 14}},
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
        st.markdown("#### Analysis Breakdown")
        for r in sig["reasons"]:
            rl   = r.lower()
            if any(w in rl for w in ["bullish","support","upward","pe writing","put sum","itm ratio","covering"]):
                st.markdown('<span style="color:#26a69a">▲</span> ' + r, unsafe_allow_html=True)
            elif any(w in rl for w in ["bearish","resistance","downward","ce writing","call sum"]):
                st.markdown('<span style="color:#ef5350">▼</span> ' + r, unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#FF9800">◆</span> ' + r, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### Key Levels")
        st.dataframe(pd.DataFrame({
            "Level": ["Spot Price","Max Pain","CE Resistance (OI)","PE Support (OI)"],
            "Price": [underlying, max_pain, sig["max_ce_resistance"], sig["max_pe_support"]],
        }), hide_index=True, use_container_width=True)
        st.markdown("---")
        strat_note = sig.get("strategy_note", "")
        if strat_note:
            st.markdown("#### Strategy Recommendation")
            strat_col = "#4CAF50" if "condor" in strat_note.lower() or "spread" in strat_note.lower() else \
                        "#26a69a" if "buy" in strat_note.lower() else "#FF9800"
            st.markdown(
                '<div style="background:#1e1e2e;padding:12px;border-radius:8px;border-left:4px solid {};">'
                '<b style="color:{};">{}</b><br><span style="color:#ccc;font-size:13px;">{}</span>'
                '</div>'.format(strat_col, strat_col, strat_type, strat_note),
                unsafe_allow_html=True
            )

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # OI TREND HISTORY — CE/PE/Net Flow change over time windows
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 📉 OI Trend History — Change per Time Window")
    st.caption(
        "Tracks CE OI, PE OI, Net Flow and PCR every refresh and compares with "
        "snapshots from 5 min · 15 min · 30 min · 1 hr · 2 hr · 3 hr · 4 hr · 5 hr ago. "
        "Data persists across browser refreshes (saved to disk)."
    )

    # Legend / explanation box
    with st.expander("📖 What do these columns mean? (click to read)", expanded=False):
        st.markdown("""
| Column | What it is | Bullish if… | Bearish if… |
|--------|-----------|-------------|-------------|
| **CE OI Chg** | Change in total Call Open Interest | Negative (calls being closed/covered) | Positive (new calls being written = resistance building) |
| **PE OI Chg** | Change in total Put Open Interest | Positive (new puts being written = support building) | Negative (puts being closed = support weakening) |
| **Net Flow** | PE OI Chg − CE OI Chg | Positive (more put writing than call writing) | Negative (more call writing) |
| **PCR Chg** | Change in Put-Call Ratio | Rising (more puts vs calls) | Falling |
| **Spot Chg** | Price change since that snapshot | Rising | Falling |
| **Trend** | 5-factor consensus | BULLISH 🟢 | BEARISH 🔴 |

**Simple rule**: If CE OI is going DOWN and PE OI is going UP → Smart money is BULLISH (defending puts, covering calls).
If CE OI is going UP and PE OI is going DOWN → Smart money is BEARISH.
""")

    # ── Live Technical Trend (EMA crossover + VWAP HLC3) ─────────────────────
    st.markdown("#### Live Technical Indicators")
    if _intra_t is not None and not _intra_t.empty and len(_intra_t) >= 5:
        _lt  = _intra_t.iloc[-1]
        _e9  = float(_lt.get("ema9",  0))
        _e20 = float(_lt.get("ema20", 0))
        _vwap  = float(_lt.get("vwap",  0))
        _close = float(_lt.get("close", underlying))
        _rsi   = float(_lt.get("rsi",   50))
        _macd_v = float(_lt.get("macd",     0))
        _macd_s = float(_lt.get("macd_sig", 0))

        # EMA crossover: find how many bars ago it last crossed
        _ema_bull_now  = _intra_t["ema9"] > _intra_t["ema20"]
        _cross_changes = _ema_bull_now != _ema_bull_now.shift(1)
        _cross_indices = _intra_t.index[_cross_changes & (_intra_t.index > 0)].tolist()
        _cross_bars_ago = (len(_intra_t) - 1 - _cross_indices[-1]) if _cross_indices else None
        _cross_dir = "Bullish cross UP" if bool(_ema_bull_now.iloc[-1]) else "Bearish cross DOWN"
        _cross_txt = "{} ({} candles ago)".format(_cross_dir, _cross_bars_ago) \
                     if _cross_bars_ago is not None else "No crossover in this window"

        _ema_status  = "EMA9 > EMA20 — Uptrend" if _e9 > _e20 else "EMA9 < EMA20 — Downtrend"
        _vwap_diff   = _close - _vwap
        _vwap_pct    = _vwap_diff / _vwap * 100 if _vwap > 0 else 0
        _vwap_status = ("Above VWAP +{:.0f} pts ({:+.2f}%) — Intraday Bullish".format(_vwap_diff, _vwap_pct)
                        if _vwap_diff >= 0 else
                        "Below VWAP {:.0f} pts ({:+.2f}%) — Intraday Bearish".format(_vwap_diff, _vwap_pct))
        _macd_status = ("MACD {:.1f} > Signal {:.1f} — Momentum Bullish".format(_macd_v, _macd_s)
                        if _macd_v > _macd_s else
                        "MACD {:.1f} < Signal {:.1f} — Momentum Bearish".format(_macd_v, _macd_s))
        _rsi_status  = ("RSI {:.0f} — Overbought (sell pressure)".format(_rsi) if _rsi > 70 else
                        "RSI {:.0f} — Oversold (watch for bounce)".format(_rsi) if _rsi < 30 else
                        "RSI {:.0f} — {}".format(_rsi, "Bullish range (55-70)" if _rsi > 55 else
                                                         "Bearish range (30-45)" if _rsi < 45 else "Neutral zone (45-55)"))

        _tech_df = pd.DataFrame([
            {
                "Indicator":   "EMA 9 / EMA 20",
                "Current":     "EMA9 Rs.{:,.1f}  |  EMA20 Rs.{:,.1f}".format(_e9, _e20),
                "Status":      _ema_status,
                "Cross event": _cross_txt,
                "Meaning":     "EMA9 > EMA20 = short-term uptrend. Cross UP = new bull trend starting.",
            },
            {
                "Indicator":   "VWAP HLC3 (Session)",
                "Current":     "VWAP Rs.{:,.1f}  |  Price Rs.{:,.1f}".format(_vwap, _close),
                "Status":      _vwap_status,
                "Cross event": "Live",
                "Meaning":     "Price above VWAP = buyers in control. Below VWAP = sellers in control.",
            },
            {
                "Indicator":   "MACD (12, 26, 9)",
                "Current":     "MACD {:.1f}  |  Signal {:.1f}  |  Hist {:+.1f}".format(_macd_v, _macd_s, _macd_v - _macd_s),
                "Status":      _macd_status,
                "Cross event": "Histogram growing" if abs(_macd_v - _macd_s) > abs(float(_intra_t.iloc[-2].get("macd_hist", 0))) else "Histogram shrinking",
                "Meaning":     "MACD crosses Signal = trend change. Histogram growing = momentum increasing.",
            },
            {
                "Indicator":   "RSI 14",
                "Current":     "RSI {:.1f}".format(_rsi),
                "Status":      _rsi_status,
                "Cross event": ">70 = Overbought  |  <30 = Oversold",
                "Meaning":     "RSI >60 = bullish push. RSI <40 = bearish pressure. 40-60 = consolidation.",
            },
        ])

        def _s_col(v):
            v = str(v)
            if any(w in v.lower() for w in ["bullish","uptrend","above","bounce"]):
                return "color:#26a69a;font-weight:bold"
            if any(w in v.lower() for w in ["bearish","downtrend","below","overbought","sell"]):
                return "color:#ef5350;font-weight:bold"
            return "color:#FF9800"

        st.dataframe(
            _tech_df.style.map(_s_col, subset=["Status"]),
            use_container_width=True, hide_index=True, height=195,
        )
    else:
        st.info("Intraday candles loading — EMA/VWAP indicators appear after first fetch.")

    st.divider()

    # Build current state dict for delta computation
    _oi_current = {
        "ce_oi_total":  float(df["ce_oi"].sum()),
        "pe_oi_total":  float(df["pe_oi"].sum()),
        "call_sum_atm": sig.get("call_sum", 0) * 1000,
        "put_sum_atm":  sig.get("put_sum",  0) * 1000,
        "pcr":          pcr,
        "underlying":   underlying,
    }

    _trend_rows = oit.get_trend_table(symbol, _oi_current)
    _first_snap = oit.first_snapshot_time(symbol)
    if _first_snap:
        # first_snapshot_time() returns "HH:MM IST" formatted string directly
        st.caption("Tracking since **{}** today. Snapshots auto-saved every refresh.".format(_first_snap))
    else:
        st.info("First snapshot saving now. Refresh in 5 minutes to see 5-min window data.")

    # Render table
    _display_rows = []
    for r in _trend_rows:
        if not r["has_data"]:
            _display_rows.append({
                "Window":         r["window"],
                "Snap @":         r.get("snap_time") or "—",
                "CE OI Chg":      r["age_str"],
                "PE OI Chg":      "—",
                "Net Flow":       "—",
                "PCR Chg":        "—",
                "Spot Chg":       "—",
                "Trend":          r["trend"],
                "What it means":  r["interpretation"],
            })
        else:
            def _fmt_oi(v):
                if v is None: return "—"
                sign = "+" if v >= 0 else ""
                if abs(v) >= 100000:
                    return "{}{:.1f}L".format(sign, v/100000)
                elif abs(v) >= 1000:
                    return "{}{:.1f}K".format(sign, v/1000)
                return "{}{}".format(sign, int(v))

            _display_rows.append({
                "Window":        r["window"],
                "Snap @":        r["snap_time"],
                "CE OI Chg":     _fmt_oi(r["ce_chg"]),
                "PE OI Chg":     _fmt_oi(r["pe_chg"]),
                "Net Flow":      _fmt_oi(r["net_flow"]),
                "PCR Chg":       "{:+.3f}".format(r["pcr_chg"]) if r["pcr_chg"] is not None else "—",
                "Spot Chg":      "{:+.1f}".format(r["spot_chg"]) if r["spot_chg"] is not None else "—",
                "Trend":         r["trend"],
                "What it means": r["interpretation"],
            })

    if _display_rows:
        _tdf = pd.DataFrame(_display_rows)

        def _trend_col(v):
            if "BULL" in str(v):  return "color:#26a69a;font-weight:bold"
            if "BEAR" in str(v):  return "color:#ef5350;font-weight:bold"
            return "color:#FF9800"

        def _num_col(v):
            v = str(v)
            if v.startswith("+") and v != "+0" and v != "+0.0":
                return "color:#26a69a"
            if v.startswith("-"):
                return "color:#ef5350"
            return ""

        st.dataframe(
            _tdf.style
                .map(_trend_col, subset=["Trend"])
                .map(_num_col,   subset=["CE OI Chg", "PE OI Chg", "Net Flow",
                                         "PCR Chg", "Spot Chg"]),
            use_container_width=True,
            hide_index=True,
            height=340,
        )
    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # SMART TRADE INTELLIGENCE — Mark Douglas · Livermore · ICT · Van Tharp
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🧠 Smart Trade Intelligence")
    st.caption("Mark Douglas · Jesse Livermore · ICT/Smart Money · Van Tharp · Historical Expiry Learning")

    with st.spinner("Running multi-framework analysis…"):
        _daily_c   = _get_candles(symbol, is_index, 40, data_source)
        _intra_raw = _get_intraday(symbol, "15min", data_source)
        _intra_t   = _compute_tech(_intra_raw) if not _intra_raw.empty else pd.DataFrame()
        _dte_val   = max(0, int(ea.hours_to_expiry(expiry) / 6.25))
        _sig_ext   = {**sig, "underlying_proxy": underlying}

        _douglas   = sa.mark_douglas_score(_sig_ext, _intra_t, pcr, _dte_val)
        _livermore = sa.livermore_analysis(_daily_c, underlying, symbol)
        _ict       = sa.ict_analysis(df, underlying, _sig_ext)
        _history   = sa.learn_from_expiry_history(_daily_c)
        _atm_prem  = float(df.iloc[(df["strike"]-underlying).abs().idxmin()]["ce_ltp"]) \
                     if not df.empty else underlying * 0.008
        _tharp     = sa.van_tharp_sizing(underlying, max(_atm_prem, underlying * 0.005),
                                         _dte_val, symbol)
        _conf, _dir, _reasons = sa.master_confidence(_douglas, _livermore, _ict, _history)

    _conf_col = "#4CAF50" if "CE" in _dir else "#ef5350" if "PE" in _dir else "#FF9800"

    mc1, mc2, mc3 = st.columns([1, 1, 2])
    with mc1:
        st.markdown(
            '<div style="background:#0d1a0d;border:3px solid {c};padding:20px;'
            'border-radius:14px;text-align:center;">'
            '<div style="color:#aaa;font-size:12px;letter-spacing:1px;">MASTER CONFIDENCE</div>'
            '<div style="font-size:52px;font-weight:bold;color:{c};">{p}%</div>'
            '<div style="font-size:17px;font-weight:bold;color:{c};">{d}</div>'
            '</div>'.format(c=_conf_col, p=_conf, d=_dir),
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            '<div style="background:#1e1e2e;padding:16px;border-radius:12px;">'
            '<div style="color:#aaa;font-size:11px;">MARK DOUGLAS VERDICT</div>'
            '<div style="font-size:14px;color:#eee;margin:6px 0;font-weight:bold;">{}</div>'
            '<div style="color:#aaa;font-size:11px;">{}/{} indicators agree</div>'
            '<div style="color:#ccc;font-size:12px;margin-top:8px;">{}</div>'
            '</div>'.format(
                _douglas["verdict"],
                max(_douglas["bull"], _douglas["bear"]),
                _douglas["total"], _douglas["advice"],
            ),
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown("**Multi-framework reasoning:**")
        for _ln in _reasons:
            st.markdown("• " + _ln)

    st.divider()
    _fa1, _fa2 = st.columns(2)
    with _fa1:
        with st.expander("📖 Jesse Livermore — Trend, Stage & Pivots"):
            lv = _livermore
            _tc = "#4CAF50" if lv["trend"]=="UPTREND" else "#ef5350" if lv["trend"]=="DOWNTREND" else "#FF9800"
            st.markdown(
                '<div style="background:#1e1e2e;padding:14px;border-radius:10px;">'
                '<b style="color:{c};">{t}</b> &nbsp;|&nbsp;<span style="color:#ccc;">{s}</span><br><br>'
                '<span style="color:#aaa;font-size:12px;">5d momentum: <b>{m:+.2f}%</b> '
                '| EMA-slow: <b>&#8377;{e:.0f}</b> | 20d range: <b>&#8377;{r:.0f}</b></span><br>'
                '<i style="color:#ddd;font-size:12px;">{tape}</i></div>'.format(
                    c=_tc, t=lv["trend"], s=lv["stage"],
                    m=lv["mom5_pct"], e=lv["ema_slow"], r=lv["range20"], tape=lv["tape"]),
                unsafe_allow_html=True,
            )
            if lv.get("pivot_hi"):
                st.markdown("**Resistance:** " + " | ".join("&#8377;{:,.0f}".format(p) for p in lv["pivot_hi"]))
            if lv.get("pivot_lo"):
                st.markdown("**Support:** "    + " | ".join("&#8377;{:,.0f}".format(p) for p in lv["pivot_lo"]))

        with st.expander("🏦 ICT / Smart Money — Institutional Footprints"):
            ic = _ict
            st.info(ic.get("narrative", ""))
            st.dataframe(pd.DataFrame({
                "Level": ["CE Wall (Resistance)", "PE Wall (Support)", "Max Pain Magnet"],
                "Strike": [ic.get("ce_wall",0), ic.get("pe_wall",0), ic.get("max_pain",0)],
                "Dist from Spot": ["&#8377;{:+.0f}".format(ic.get("dist_ce",0)),
                                   "&#8377;{:+.0f}".format(-ic.get("dist_pe",0)),
                                   "&#8377;{:+.0f}".format(ic.get("dist_mp",0))],
            }), hide_index=True, use_container_width=True)

    with _fa2:
        with st.expander("💰 Van Tharp — Position Sizing & R-Multiple"):
            vt = _tharp
            st.markdown(
                '<div style="background:#1e1e2e;padding:14px;border-radius:10px;">'
                '<b style="color:#FFD700;">1R = 1.5% of &#8377;{:,.0f} capital</b><br><br>'
                '<table style="width:100%;color:#ccc;font-size:13px;">'
                '<tr><td>ATM Premium</td><td><b>&#8377;{}</b></td></tr>'
                '<tr><td>Stop-Loss ({}%)</td><td style="color:#ef5350;"><b>&#8377;{}</b></td></tr>'
                '<tr><td>Target (2R)</td><td style="color:#26a69a;"><b>&#8377;{}</b></td></tr>'
                '<tr><td>Lot Size</td><td><b>{} units</b></td></tr>'
                '<tr><td>Max Lots</td><td style="color:#FFD700;"><b>{}</b></td></tr>'
                '<tr><td>Risk/Lot</td><td><b>&#8377;{:,}</b></td></tr>'
                '<tr><td>Expectancy</td><td><b>{:.2f}R</b></td></tr>'
                '</table></div>'.format(
                    vt["capital"], vt["premium_est"], int(vt["sl_pct"]),
                    vt["sl_price"], vt["target_2r"], vt["lot_size"],
                    vt["max_lots"], int(vt["risk_per_lot"]), vt["expectancy_r"]),
                unsafe_allow_html=True,
            )
            st.caption("Van Tharp: Size so 1 stop-out = 1.5% of capital. Never risk more than 2%.")

        with st.expander("📚 Historical Expiry Learning — Pattern Match"):
            hist = _history
            st.info(hist.get("summary", "No data."))
            _bc = "#4CAF50" if "BULLISH" in hist.get("pattern_bias","") else \
                  "#ef5350" if "BEARISH" in hist.get("pattern_bias","") else "#FF9800"
            st.markdown(
                '<div style="background:#1e1e2e;padding:10px;border-radius:8px;'
                'border-left:4px solid {c};">'
                '<b style="color:{c};">Today matches: {b}</b></div>'.format(
                    c=_bc, b=hist.get("pattern_bias","")),
                unsafe_allow_html=True,
            )
            if hist.get("lesson"):
                st.success("💡 " + hist["lesson"])
            if hist.get("patterns"):
                _hdf = pd.DataFrame(hist["patterns"])[
                    ["date","weekday","rsi","ema_trend","range_pct","close_chg","outcome"]]
                _hdf.columns = ["Date","Day","RSI","EMA","Range%","Chg%","Outcome"]
                def _c_out(v):
                    return "color:#4CAF50" if v=="BULLISH" else "color:#ef5350" if v=="BEARISH" else "color:#FF9800"
                st.dataframe(_hdf.style.map(_c_out, subset=["Outcome"]),
                             hide_index=True, use_container_width=True)

    with st.expander("📊 Mark Douglas — Full Confluence Grid"):
        _chk = pd.DataFrame(_douglas["checks"], columns=["Indicator","Signal"])
        _chk["Signal"] = _chk["Signal"].map({1:"✅ BULL", -1:"❌ BEAR", 0:"⚪ NEUTRAL"})
        st.dataframe(_chk, hide_index=True, use_container_width=True)
        st.markdown("**{} Bull | {} Bear | {} Neutral** of {} → **{}**".format(
            _douglas["bull"], _douglas["bear"], _douglas["neutral"],
            _douglas["total"], _douglas["verdict"]))

    st.divider()

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("#### Open Interest Distribution — Murphy Support/Resistance")
        f = go.Figure()
        # CE OI bars (resistance above spot)
        ce_colors = ["#b71c1c" if s > underlying else "#ef5350" for s in df["strike"]]
        pe_colors = ["#004d40" if s < underlying else "#26a69a" for s in df["strike"]]
        f.add_trace(go.Bar(x=df["strike"], y=df["ce_oi"]/1000, name="Call OI (Resistance)",
                           marker_color=ce_colors, opacity=0.9))
        f.add_trace(go.Bar(x=df["strike"], y=df["pe_oi"]/1000, name="Put OI (Support)",
                           marker_color=pe_colors, opacity=0.9))
        f.add_vline(x=underlying, line_dash="dash", line_color="white",
                    annotation_text="Spot {}".format(int(underlying)), annotation_font_color="white")
        f.add_vline(x=max_pain, line_dash="dot", line_color="#FFD700",
                    annotation_text="MaxPain {}".format(int(max_pain)), annotation_font_color="#FFD700")
        f.add_vline(x=sig["max_ce_resistance"], line_dash="dot", line_color="#ef5350",
                    annotation_text="Resist {}".format(int(sig["max_ce_resistance"])),
                    annotation_font_color="#ef5350", annotation_position="top left")
        f.add_vline(x=sig["max_pe_support"], line_dash="dot", line_color="#26a69a",
                    annotation_text="Support {}".format(int(sig["max_pe_support"])),
                    annotation_font_color="#26a69a")
        f.update_layout(barmode="overlay", height=320,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", y=1.1),
                        xaxis=dict(title="Strike", gridcolor="#333"),
                        yaxis=dict(title="OI (thousands)", gridcolor="#333"),
                        margin=dict(t=20, b=10))
        st.plotly_chart(f, use_container_width=True)

    with cc2:
        st.markdown("#### OI Change — Velocity & Direction (Net Flow)")
        # Net OI flow: PE build - CE build at each strike
        net_oi_flow = (df["pe_chg_oi"] - df["ce_chg_oi"]) / 1000
        flow_colors = ["#26a69a" if v > 0 else "#ef5350" for v in net_oi_flow]
        f2 = go.Figure()
        # Raw CE/PE change bars (lighter)
        f2.add_trace(go.Bar(x=df["strike"], y=df["ce_chg_oi"]/1000, name="CE Build",
                            marker_color="#ef5350", opacity=0.4))
        f2.add_trace(go.Bar(x=df["strike"], y=df["pe_chg_oi"]/1000, name="PE Build",
                            marker_color="#26a69a", opacity=0.4))
        # Net flow line (the KEY signal: positive = bullish PE building)
        f2.add_trace(go.Scatter(x=df["strike"], y=net_oi_flow, name="Net OI Flow (PEâˆ'CE)",
                                mode="lines+markers",
                                line=dict(color="#FFD700", width=2.5),
                                marker=dict(color=flow_colors, size=8)))
        f2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        f2.add_vline(x=underlying, line_dash="dash", line_color="white",
                     annotation_text="Spot {}".format(int(underlying)), annotation_font_color="white")
        f2.update_layout(barmode="group", height=320,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         legend=dict(orientation="h", y=1.1),
                         xaxis=dict(title="Strike", gridcolor="#333"),
                         yaxis=dict(title="Chg OI (K) — +ve = PE writing (bullish)", gridcolor="#333"),
                         margin=dict(t=20, b=10))
        st.plotly_chart(f2, use_container_width=True)
        # Net OI flow summary
        net_total = net_oi_flow.sum()
        flow_bias = "[BULL] PE writing dominates (Bullish)" if net_total > 0 else "[BEAR] CE writing dominates (Bearish)"
        st.caption("Net OI Flow: {:+.1f}K â†' {}".format(net_total, flow_bias))

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
            'PCR &lt;0.5=Bearish | 0.5"0.8=Neutral-Bearish<br>'
            'PCR 0.8"1.2=Neutral | 1.2"1.5=Bullish | &gt;1.5=Extremely Bullish'
            '</div></div>'.format(pcr_col, pcr, pcr_col, interp),
            unsafe_allow_html=True
        )

    st.divider()

    # ── Expiry proximity alert ────────────────────────────────────────────────
    h_left = ea.hours_to_expiry(expiry)
    atm_iv = float(df.iloc[(df["strike"] - underlying).abs().idxmin()]["ce_iv"])
    if atm_iv == 0:
        atm_iv = float(df["ce_iv"][df["ce_iv"] > 0].mean()) if (df["ce_iv"] > 0).any() else 15.0

    if 0 < h_left <= 6.5:  # expiry day
        st.markdown(
            '<div style="background:#3d0000;border:2px solid #ef5350;padding:12px;border-radius:10px;margin-bottom:10px;">'
            '<b style="color:#ef5350;">âš¡ EXPIRY DAY</b> — {:.1f} market hours left | '
            'Switch to <b>ðŸ"… Expiry Signals</b> tab for scalp targets</div>'.format(h_left),
            unsafe_allow_html=True
        )
    elif 0 < h_left <= 24:
        next_info = ea.next_expiry_info(
            meta.get("all_expiries", [expiry]), expiry, underlying, atm_iv)
        if "error" not in next_info:
            st.markdown(
                '<div style="background:#1a2a00;border:2px solid #8BC34A;padding:12px;border-radius:10px;margin-bottom:10px;">'
                '<b style="color:#8BC34A;">â° Expiry Tomorrow</b> — Next expiry: <b>{}</b> | '
                'Switch to <b>ðŸ"… Expiry Signals</b> for next-expiry analysis</div>'.format(
                    next_info.get("next_expiry", "")),
                unsafe_allow_html=True
            )

    st.markdown("### Prediction — {} Â· ATM IV: {}%".format(
        "Today's Range" if h_left <= 6.5 else "Tomorrow", round(atm_iv, 1)))

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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 2 — MARKET SCANNER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_scan:
    st.markdown("## ðŸ" Market Scanner")
    st.markdown(
        "Multi-instrument trending analysis using **Murphy's OI+Price 4-scenario**, "
        "**Natenberg IV Rank**, and **McMillan PCR** methodologies."
    )

    sc1, sc2 = st.columns([1, 3])
    with sc1:
        scan_stocks = st.checkbox("Include F&O Stocks", value=False)
        scan_demo_mode = (data_source == "Demo Mode")

    with sc2:
        st.markdown(
            "**Murphy (1999) OI Framework:** Rising price + Rising OI = Bullish (fresh longs) Â· "
            "Rising price + Falling OI = Short covering (weak) Â· "
            "Falling price + Rising OI = Bearish (fresh shorts)"
        )

    if st.button("ðŸ"„ Scan Now", type="primary", key="scan_btn"):
        with st.spinner("Scanning instrumentsâ€¦"):
            if scan_demo_mode or data_source != "Angel One (Live)":
                scan_results = sc.scan_demo(include_stocks=scan_stocks)
            else:
                scan_results = sc.scan_live(symbol, df, meta, sig, include_stocks=scan_stocks)
        st.session_state["scan_results"] = scan_results

    scan_results = st.session_state.get("scan_results")
    if not scan_results:
        st.info("Click **Scan Now** to analyse trending instruments.")
    else:
        # Summary conviction bar
        bullish  = [r for r in scan_results if r["signal"] == "BUY CALL"]
        bearish  = [r for r in scan_results if r["signal"] == "BUY PUT"]
        watching = [r for r in scan_results if r["signal"] == "WATCH"]

        sb1, sb2, sb3 = st.columns(3)
        sb1.metric("[BULL] Bullish",  len(bullish))
        sb2.metric("[BEAR] Bearish",  len(bearish))
        sb3.metric("[NEUT] Watch",    len(watching))
        st.divider()

        # Ranked table
        st.markdown("### Ranked by Conviction (strongest first)")
        for r in scan_results:
            sig_icon = "[BULL]" if r["signal"] == "BUY CALL" else "[BEAR]" if r["signal"] == "BUY PUT" else "[NEUT]"
            border   = "#4CAF50" if r["signal"] == "BUY CALL" else "#ef5350" if r["signal"] == "BUY PUT" else "#FF9800"

            with st.expander("{} **{}** — {} (conviction: {:+.0f})".format(
                    sig_icon, r["symbol"], r["signal"], r["conviction"]), expanded=False):

                ec1, ec2, ec3, ec4 = st.columns(4)
                ec1.metric("PCR",         "{:.2f}".format(r["pcr"]),
                            delta=r["pcr_label"])
                ec2.metric("CE OI Chg",   "{:+.1f}%".format(r["ce_oi_chg"]))
                ec3.metric("PE OI Chg",   "{:+.1f}%".format(r["pe_oi_chg"]))
                ec4.metric("Net OI Bias", "{:+.1f}".format(r["net_oi_bias"]),
                            delta="Bullish" if r["net_oi_bias"] > 0 else "Bearish")

                ec5, ec6, ec7 = st.columns(3)
                ec5.metric("ATM IV",      "{:.1f}%".format(r["atm_iv"]))
                ec6.metric("IV Rank",     "{:.0f}%".format(r["iv_rank"]),
                            delta="Cheap buy" if r["iv_rank"] < 30 else "Sell IV" if r["iv_rank"] > 70 else "Fair")
                ec7.metric("Vol Ratio",   "{:.1f}x".format(r["vol_ratio"]),
                            delta="High volume" if r["vol_ratio"] > 1.5 else "Normal")

                st.markdown("**Murphy OI Signal:** {} — {}".format(r["murphy_signal"], r["murphy_note"]))
                st.caption("Natenberg: {}".format(r["iv_note"]))

        st.divider()

        # Heatmap of convictions
        st.markdown("### Conviction Heatmap")
        scan_df = pd.DataFrame([{
            "Symbol":    r["symbol"],
            "Signal":    r["signal"],
            "Conviction":r["conviction"],
            "PCR":       r["pcr"],
            "CE OI Chg%":r["ce_oi_chg"],
            "PE OI Chg%":r["pe_oi_chg"],
            "IV Rank":   r["iv_rank"],
            "Vol Ratio": r["vol_ratio"],
            "Murphy":    r["murphy_signal"],
        } for r in scan_results])

        def _color_signal(val):
            if val == "BUY CALL": return "background-color:#1a3a00;color:#4CAF50"
            if val == "BUY PUT":  return "background-color:#3a0000;color:#ef5350"
            return "background-color:#2a2a00;color:#FF9800"

        def _color_conv(val):
            if val > 40:  return "color:#4CAF50;font-weight:bold"
            if val < -40: return "color:#ef5350;font-weight:bold"
            return "color:#FF9800"

        styled = (scan_df.style
                  .map(_color_signal, subset=["Signal"])
                  .map(_color_conv,   subset=["Conviction"])
                  .format({"PCR": "{:.2f}", "CE OI Chg%": "{:+.1f}", "PE OI Chg%": "{:+.1f}",
                            "IV Rank": "{:.0f}%", "Conviction": "{:+.0f}", "Vol Ratio": "{:.1f}x"}))
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.caption(
            "ðŸ"– Sources: Murphy J.J. (1999) Technical Analysis of Financial Markets Â· "
            "Natenberg S. (2015) Option Volatility & Pricing Â· McMillan L.G. (2012) Options as a Strategic Investment"
        )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 3 — EXPIRY SIGNALS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_exp:
    st.markdown("## ðŸ"… Expiry Signals")

    # Re-compute ATM IV for this tab
    atm_iv_exp = 0.0
    if not df.empty:
        try:
            idx_atm    = (df["strike"] - underlying).abs().idxmin()
            atm_iv_exp = float(df.iloc[idx_atm]["ce_iv"])
        except Exception:
            pass
    if atm_iv_exp == 0:
        atm_iv_exp = 15.0

    h_left_exp = ea.hours_to_expiry(expiry)

    # ── Header ────────────────────────────────────────────────────────────────
    if h_left_exp <= 0:
        st.error("âš ï¸ Current expiry {} has passed. Showing analysis for reference.".format(expiry))
    elif h_left_exp <= 6.5:
        st.markdown(
            '<div style="background:#2d0000;border:2px solid #ef5350;padding:15px;border-radius:12px;">'
            '<h3 style="color:#ef5350;margin:0;">âš¡ EXPIRY DAY — {:.1f} market hours left</h3>'
            '<p style="color:#ccc;margin:5px 0 0 0;">Augen (2009): Max Pain magnet effect strongest in last 2 hours. '
            'Gamma spikes near ATM — scalp targets 30"80 points on NIFTY.</p>'
            '</div>'.format(h_left_exp),
            unsafe_allow_html=True
        )
    elif h_left_exp <= 24:
        st.warning("â° Expiry **{}** is tomorrow. Check next expiry below.".format(expiry))
    else:
        st.info("Current expiry: **{}** — {:.1f} market hours remaining".format(expiry, h_left_exp))

    st.divider()

    # ── Expiry day scalp signal ────────────────────────────────────────────────
    exp_col, next_col = st.columns(2)

    with exp_col:
        st.markdown("### Expiry Scalp Signal (Augen Framework)")
        st.caption("Volume + OI change + Max Pain — 30"50 point targets")

        exp_sig = ea.expiry_scalp_signal(df, meta, sig, atm_iv=atm_iv_exp)

        sig_icon = "[BULL]" if exp_sig["direction"] == "BUY CALL" else \
                   "[BEAR]" if exp_sig["direction"] == "BUY PUT" else "[NEUT]"
        sig_bg   = {"BUY CALL": "#0a2a0a", "BUY PUT": "#2a0a0a"}.get(exp_sig["direction"], "#2a2a00")
        sig_col_ = {"BUY CALL": "#4CAF50", "BUY PUT": "#ef5350"}.get(exp_sig["direction"], "#FF9800")

        st.markdown(
            '<div style="background:{};border:2px solid {};padding:20px;border-radius:12px;text-align:center;">'
            '<div style="font-size:36px;">{}</div>'
            '<div style="font-size:24px;font-weight:bold;color:{};">{}</div>'
            '<div style="color:#ccc;margin-top:8px;">{}</div>'
            '</div>'.format(sig_bg, sig_col_, sig_icon,
                            sig_col_, exp_sig["direction"], exp_sig["action"]),
            unsafe_allow_html=True
        )

        st.markdown("**Signal Checks:**")
        for r in exp_sig["reasons"]:
            st.markdown(r)

        # Expected range
        er = exp_sig["expected_range"]
        st.markdown("---")
        st.markdown("**Expected Range (Natenberg IV-based)**")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Upper", "₹{:,.0f}".format(er["upper"]))
        rc2.metric("Move ±", "₹{:,.0f}".format(er["move_pts"]))
        rc3.metric("Lower", "₹{:,.0f}".format(er["lower"]))

        # Trade params
        st.markdown("---")
        st.markdown("**Expiry Scalp Parameters**")
        prem = exp_sig["atm_premium_est"]
        sl_p = round(prem * (1 - exp_sig["sl_pct"]), 1)
        tg_p = round(prem * (1 + exp_sig["tgt_pct"]), 1)
        st.dataframe(pd.DataFrame({
            "Parameter": ["ATM Premium (est.)", "Stop-Loss (âˆ'30%)", "Target (+50%)",
                          "Points Target", "Lot Size", "Best Hold"],
            "Value":     ["₹{:.1f}".format(prem),
                          "₹{:.1f}".format(sl_p),
                          "₹{:.1f}".format(tg_p),
                          "±{} pts (NIFTY)".format(exp_sig["pts_target"]),
                          "{} units".format(exp_sig["lot_size"]),
                          "30"60 min or 3:15 PM"]
        }), hide_index=True, use_container_width=True)

    with next_col:
        st.markdown("### Next Expiry Analysis")

        all_exp = meta.get("all_expiries", [expiry])
        nxt = ea.next_expiry_info(all_exp, expiry, underlying, atm_iv_exp)

        if "error" in nxt:
            st.info("Next expiry data not available (need â‰¥2 expiry dates in chain).")
        else:
            st.markdown("**Next Expiry: {}**".format(nxt["next_expiry"]))
            st.markdown("{} trading days away Â· ATM IV for next: {}%".format(
                nxt["days_to_next"], nxt["iv_for_next"]))

            nr = nxt["expected_range"]
            nc1, nc2, nc3 = st.columns(3)
            nc1.metric("Upper", "₹{:,.0f}".format(nr["upper"]))
            nc2.metric("Move ±", "₹{:,.0f}".format(nr["move_pts"]))
            nc3.metric("Lower", "₹{:,.0f}".format(nr["lower"]))

            st.markdown("---")
            st.markdown("**Strategy Recommendation (Natenberg)**")
            st.markdown(
                '<div style="background:#1e1e2e;padding:15px;border-radius:10px;">'
                '<b style="color:#26a69a;font-size:18px;">{}</b><br>'
                '<span style="color:#ccc;">{}</span>'
                '</div>'.format(nxt["strategy"], nxt["strategy_why"]),
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Augen Gamma Zones
        st.markdown("### Gamma Hot Zones (Augen)")
        st.caption("Strikes where OI creates strongest magnet effect — expiry day key levels")
        zones = ea.augen_gamma_zones(df, underlying, h_left_exp)
        if zones:
            z_df = pd.DataFrame(zones)[["strike","type","ce_oi","pe_oi","pull_score","dist_pts"]]
            z_df.columns = ["Strike","Type","CE OI","PE OI","Pull Score","Dist (pts)"]
            def _color_zone(val):
                return "color:#ef5350" if "RESIST" in str(val) else "color:#26a69a"
            st.dataframe(
                z_df.style.map(_color_zone, subset=["Type"])
                          .format({"CE OI": "{:,.0f}", "PE OI": "{:,.0f}", "Pull Score": "{:.1f}"}),
                hide_index=True, use_container_width=True
            )
        else:
            st.info("Load live option chain data to see gamma zones.")

    st.divider()

    # ── Theory section ────────────────────────────────────────────────────────
    st.markdown("### Expiry Day Trading Methodology")
    with st.expander("ðŸ"– Augen (2009) — Expiry Day Framework", expanded=False):
        st.markdown("""
**From "Trading Options at Expiration" by Jeff Augen:**

1. **Gamma Spike**: ATM options gain gamma exponentially on expiry day.
   - A 0.5% move in the underlying can double/halve an ATM option's price
   - Trade smaller size — premium moves are violent

2. **Max Pain Magnet**: In last 2 hours, underlying gravitates toward Max Pain
   - Below Max Pain â†' call writers buy back â†' price rises
   - Above Max Pain â†' put writers buy back â†' price falls

3. **Time-of-day patterns (NSE Expiry)**:
   - 9:20"10:30 AM: Gap fills and initial direction established
   - 12:00"1:30 PM: Lull — avoid trading, volume low
   - 2:00"3:15 PM: Max Pain pull strongest — last hour most reliable

4. **Target sizing for NIFTY**:
   - 5 min expiry scalp: ±15-25 pts
   - 15 min expiry trade: ±30-50 pts
   - Full session: ±60-100 pts
   - Set tight SL (25-30% of premium) — gamma can reverse quickly

5. **Strike selection on expiry day**:
   - ATM = 0.5 delta = best for directional trades
   - 1-OTM = cheaper but lower delta, harder to recover from wrong direction
        """)

    with st.expander("ðŸ"– Murphy (1999) — OI Analysis on Expiry", expanded=False):
        st.markdown("""
**From "Technical Analysis of Financial Markets" (Chapter 7 — Volume and OI):**

| OI Change | Price Change | Interpretation | Action |
|-----------|-------------|----------------|--------|
| Rising OI + Rising Price | â†' | Fresh longs entering = **Strong Bullish** | BUY CALL |
| Rising OI + Falling Price | â†' | Fresh shorts entering = **Strong Bearish** | BUY PUT |
| Falling OI + Rising Price | â†' | Short covering (weak move) = **Rally may fade** | CAUTION |
| Falling OI + Falling Price | â†' | Long liquidation (selling easing) = **Recovery near** | WATCH |

On expiry day, **net OI flow near ATM** (within ±2 strikes) is the key signal:
- PE writers adding OI at support â†' market expects to hold support â†' BUY CALL
- CE writers adding OI at resistance â†' market expects to hold resistance â†' BUY PUT
        """)

    st.caption("âš ï¸ Educational only — not financial advice. Options trading involves significant risk.")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 4 — BACKTEST & MODEL TRAINING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
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
                with st.spinner("Running backtestâ€¦"):
                    metrics, rdf = backtest.run_accuracy_backtest(candles)
                if not metrics:
                    st.error("Not enough data (need â‰¥12 candles).")
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
                "Win = 1.5Ã— premium, Loss = 0.4Ã— premium (approximate)."
            )
            lot_defaults = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 40,
                            "MIDCPNIFTY": 50, "SENSEX": 20}
            sim_lot  = st.number_input("Lot Size", value=lot_defaults.get(symbol.upper(), 75),
                                        min_value=1, max_value=500)
            sim_comm = st.number_input("Commission per trade (₹)", value=40,
                                        min_value=0, max_value=500)

            if st.button("Run P&L Simulation", type="primary", key="run_pnl"):
                with st.spinner("Simulating P&Lâ€¦"):
                    metrics, rdf = backtest.run_pnl_simulation(candles, int(sim_lot), int(sim_comm))
                if not metrics:
                    st.error("Not enough data (need â‰¥12 candles).")
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
                "Save today's signal â†' auto-checked against next trading day's close. "
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
                    fdf.style.map(_oc, subset=["outcome"]),
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
                with st.spinner("Calibrating — scanning 576 parameter combinationsâ€¦"):
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 3 — PAPER TRADE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
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
            closed_df.style.map(_pnl_color, subset=["P&L ₹"]),
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 4 — CHAT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
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
        "5-min":  "10-15 min hold Â· SL 20% Â· Target 30% Â· Best: 9:20-10:00 AM",
        "15-min": "45-60 min hold Â· SL 30% Â· Target 50% Â· Best: 9:30-11:30 AM",
        "30-min": "1.5-2 hr hold  Â· SL 35% Â· Target 60% Â· Best: 9:45 AM-12:00 PM",
        "1-hr":   "2-3 hr hold    Â· SL 40% Â· Target 65% Â· Best: 9:30-11:30 AM",
        "2-hr":   "1-day hold     Â· SL 45% Â· Target 80% Â· Use credit spreads",
        "daily":  "2-3 day hold   Â· SL 50% Â· Target 100% Â· Use next-week expiry",
    }
    st.caption("â± " + tf_info[selected_tf])

    # ── Primary action buttons ────────────────────────────────────────────────
    st.markdown("#### Start here:")
    pa1, pa2 = st.columns(2)
    if pa1.button(
        "ðŸŽ¯ Get Trade Decision",
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

    checklist_label = "ðŸ"‹ {} Checklist".format(tf_options[selected_tf])
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
        "Ask: trade decision Â· checklist Â· signal Â· pcr Â· IV Â· tomorrow Â· help"
    )
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinkingâ€¦"):
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


# ── Auto Refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()

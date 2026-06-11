import os
import json
import math
import time
import threading
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from nse_data import fetch_option_chain, parse_option_chain, generate_signal
from demo_data import generate_demo_option_chain
from stock_list import NSE_STOCKS, INDICES, search_stocks
import paper_trade as pt
import chat_bot
import backtest
import model_train
import scanner as sc
import expiry_analysis as ea
import smart_analysis as sa

APP_VERSION = "v2.5"

# ── OI Snapshot disk persistence & background collector ───────────────────────
_SNAP_FILE = os.path.join(os.path.dirname(__file__), ".oi_snapshots_cache.json")
_bg_lock   = threading.Lock()

def _load_snaps_from_disk():
    try:
        if os.path.exists(_SNAP_FILE):
            with open(_SNAP_FILE) as _f:
                _data = json.load(_f)
            _cutoff = datetime.now() - timedelta(hours=7)
            return [
                {**{k: v for k, v in _s.items() if k != "time"},
                 "time": datetime.fromisoformat(_s["time"])}
                for _s in _data
                if datetime.fromisoformat(_s["time"]) >= _cutoff
            ]
    except Exception:
        pass
    return []

def _save_snaps_to_disk(snaps):
    try:
        _data = [{**{k: v for k, v in _s.items() if k != "time"},
                  "time": _s["time"].isoformat()} for _s in snaps]
        with _bg_lock:
            with open(_SNAP_FILE, "w") as _f:
                json.dump(_data, _f)
    except Exception:
        pass

_BG_STARTED = {}

def _start_bg_collector():
    if "ok" in _BG_STARTED:
        return
    _BG_STARTED["ok"] = True

    def _worker():
        while True:
            try:
                for _bg_sym in ["NIFTY", "BANKNIFTY"]:
                    _bg_raw = fetch_option_chain(_bg_sym, True)
                    _bg_df, _bg_meta = parse_option_chain(_bg_raw, 20)
                    if not _bg_df.empty and "error" not in _bg_meta:
                        _bg_ts = datetime.now()
                        _bg_sn = {
                            "time":        _bg_ts,
                            "symbol":      _bg_sym,
                            "total_ce_oi": float(_bg_df["ce_oi"].sum()),
                            "total_pe_oi": float(_bg_df["pe_oi"].sum()),
                            "pcr":         float(_bg_meta.get("pcr", 1.0)),
                            "underlying":  float(_bg_meta.get("underlying", 0)),
                        }
                        _bg_existing = _load_snaps_from_disk()
                        _bg_sym_snaps = [x for x in _bg_existing if x.get("symbol", "") == _bg_sym]
                        if not _bg_sym_snaps or (_bg_ts - _bg_sym_snaps[-1]["time"]).total_seconds() >= 55:
                            _bg_existing.append(_bg_sn)
                            _bg_cutoff = _bg_ts - timedelta(hours=7)
                            _save_snaps_to_disk([x for x in _bg_existing if x["time"] >= _bg_cutoff])
            except Exception:
                pass
            time.sleep(60)

    threading.Thread(target=_worker, daemon=True, name="oi_bg_collector").start()

_start_bg_collector()  # start once per process; safe no-op on repeated imports

# yfinance fallback symbols for intraday data
INDEX_YF_SYMBOLS = {
    "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "NIFTY_MID_SELECT.NS", "SENSEX": "^BSESN",
}

st.set_page_config(
    page_title="NSE Options Intelligence {}".format(APP_VERSION),
    page_icon="chart_with_upwards_trend",
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
        index_options = ["{} - {}".format(k, v) for k, v in INDICES.items()]
        selected_idx  = st.selectbox("Select Index", index_options)
        symbol        = selected_idx.split(" - ")[0]
        is_index      = True
    else:
        search_query = st.text_input(
            "Search Stock", value="SBIN",
            placeholder="Type symbol or name: SBIN, HDFC, TATA..."
        ).upper().strip()
        suggestions = search_stocks(search_query)
        if suggestions:
            opts     = ["{} - {}".format(s, n) for s, n in suggestions]
            selected = st.selectbox("Matches", opts, index=0)
            symbol   = selected.split(" - ")[0]
        else:
            st.warning("No match. Using: " + search_query)
            symbol = search_query
        is_index = False

    num_strikes  = st.slider("Strikes around ATM", 10, 40, 10, step=2)
    auto_refresh = st.checkbox("Auto Refresh", value=True)
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
        st.info("Try during market hours (9:15 AM - 3:30 PM IST) on Indian internet.")
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

# ── Cached data fetchers ───────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _get_nse_data(sym, idx, strikes):
    raw      = fetch_option_chain(sym, idx)
    df, meta = parse_option_chain(raw, strikes)
    return df, meta

@st.cache_data(ttl=60)
def _get_angelone_data(sym, idx, strikes):
    from angelone_data import fetch_option_chain_angelone
    return fetch_option_chain_angelone(sym, idx, strikes)

@st.cache_data(ttl=60)
def _get_sensex_data(src, n):
    if src == "Angel One (Live)":
        try:
            from angelone_data import fetch_option_chain_angelone
            _d, _m = fetch_option_chain_angelone("SENSEX", True, n)
            if not _d.empty and "error" not in _m:
                _m["source"] = "Angel One (Live) - BSE"
                return _d, _m
        except Exception:
            pass
    elif src == "NSE Direct":
        _r = fetch_option_chain("SENSEX", True)
        if "error" not in _r:
            _d, _m = parse_option_chain(_r, n)
            if not _d.empty:
                return _d, _m
    _d, _m = generate_demo_option_chain("SENSEX")
    _m["source"] = "Demo (SENSEX options on BSE - not available via NSE)"
    return _d, _m

@st.cache_data(ttl=300)
def _get_candles(sym, idx, days, src):
    if src == "Angel One (Live)":
        try:
            from angelone_data import fetch_historical_candles
            c = fetch_historical_candles(sym, idx, days)
            if c is not None and not c.empty:
                return c
        except Exception:
            pass
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


def _get_intraday(sym, tf, src):
    """Fetch intraday candles: Angel One -> yfinance -> empty DataFrame."""
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


def _compute_tech(df: pd.DataFrame) -> pd.DataFrame:
    """VWAP, EMA9/20, MACD, RSI14 from OHLCV DataFrame."""
    if df.empty:
        return df
    d = df.copy().reset_index(drop=True)
    hlc3       = (d["high"] + d["low"] + d["close"]) / 3
    cum_vol    = d["volume"].cumsum().replace(0, np.nan)
    d["vwap"]  = (hlc3 * d["volume"]).cumsum() / cum_vol
    d["ema9"]  = d["close"].ewm(span=9,  adjust=False).mean()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    ema12      = d["close"].ewm(span=12, adjust=False).mean()
    ema26      = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"]  = ema12 - ema26
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()
    delta      = d["close"].diff()
    gain       = delta.where(delta > 0, 0.0).rolling(14, min_periods=1).mean()
    loss       = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=1).mean()
    rs         = gain / loss.replace(0, 1e-10)
    d["rsi14"] = 100 - 100 / (1 + rs)
    return d


# ── Data fetch ────────────────────────────────────────────────────────────────

df   = pd.DataFrame()
meta = {}

if data_source == "Demo Mode":
    df, meta = generate_demo_option_chain(symbol if is_index else "NIFTY")
    st.info("Demo Mode ON - simulated data. Switch Data Source for live data.")

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

# ── ATM IV (used across tabs) ─────────────────────────────────────────────────
_atm_idx_g = (df["strike"] - underlying).abs().idxmin()
atm_iv = float(df.iloc[_atm_idx_g]["ce_iv"]) if not df.empty else 0.0
if atm_iv == 0:
    atm_iv = float(df["ce_iv"][df["ce_iv"] > 0].mean()) if not df.empty and (df["ce_iv"] > 0).any() else 15.0

# ── OI Snapshot tracking (for history table) ──────────────────────────────────
_now_ts = datetime.now()
_snap = {
    "time":        _now_ts,
    "symbol":      symbol,
    "total_ce_oi": float(df["ce_oi"].sum()) if not df.empty else 0.0,
    "total_pe_oi": float(df["pe_oi"].sum()) if not df.empty else 0.0,
    "pcr":         float(pcr),
    "underlying":  float(underlying),
}
if "oi_snapshots" not in st.session_state:
    st.session_state["oi_snapshots"] = _load_snaps_from_disk()
_snaps = st.session_state["oi_snapshots"]
if not _snaps or (_now_ts - _snaps[-1]["time"]).total_seconds() >= 60:
    _snaps.append(_snap)
    cutoff = _now_ts - timedelta(hours=7)
    st.session_state["oi_snapshots"] = [s for s in _snaps if s["time"] >= cutoff]
    _save_snaps_to_disk(st.session_state["oi_snapshots"])

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_sensex, tab_scan, tab_exp, tab_bt, tab_pt, tab_chat = st.tabs(
    ["[Live] Dashboard", "[SENSEX] Live", "[Scan] Scanner", "[Exp] Expiry Signals",
     "[BT] Backtest", "[PT] Paper Trade", "[AI] Chat"]
)

# =============================================================================
# TAB 1 - LIVE DASHBOARD
# =============================================================================
with tab_live:
    st.markdown("## {} Option Chain - Expiry: **{}**".format(symbol, expiry))
    st.markdown("**Spot: Rs.{:,.2f}**  |  ATM: **{}**  |  Source: `{}`".format(
        underlying, meta["atm"], source_tag))

    # ── Overall Signal Summary Card (Book Checkpoints) ─────────────────────────
    _chk_list = []

    # Check 1: PCR (McMillan)
    if pcr > 1.3:
        _chk_list.append(("PCR {:.2f}".format(pcr), "BULLISH - PE writing dominant (put writers defending support)", "#4CAF50"))
    elif pcr < 0.7:
        _chk_list.append(("PCR {:.2f}".format(pcr), "BEARISH - CE writing dominant (call writers capping rally)", "#ef5350"))
    else:
        _chk_list.append(("PCR {:.2f}".format(pcr), "NEUTRAL - balanced options activity (0.7 - 1.3 range)", "#FF9800"))

    # Check 2: Max Pain pull (Augen)
    _pain_diff_c = max_pain - underlying
    if _pain_diff_c > underlying * 0.001:
        _chk_list.append(("Max Pain Rs.{:,.0f}".format(max_pain),
                           "UPWARD PULL - Rs.{:,.0f} above spot (Augen: magnet strongest last 2hr)".format(abs(_pain_diff_c)), "#4CAF50"))
    elif _pain_diff_c < -underlying * 0.001:
        _chk_list.append(("Max Pain Rs.{:,.0f}".format(max_pain),
                           "DOWNWARD PULL - Rs.{:,.0f} below spot".format(abs(_pain_diff_c)), "#ef5350"))
    else:
        _chk_list.append(("Max Pain Rs.{:,.0f}".format(max_pain), "AT SPOT - no directional pull", "#FF9800"))

    # Check 3: Near-ATM OI build (Murphy 4-scenario)
    if not df.empty:
        _atm_idx_c = int((df["strike"] - underlying).abs().idxmin())
        _near_c = df.iloc[max(0, _atm_idx_c - 2):min(len(df), _atm_idx_c + 3)]
        _near_ce_c = _near_c["ce_chg_oi"].sum()
        _near_pe_c = _near_c["pe_chg_oi"].sum()
        if _near_pe_c > _near_ce_c * 1.2:
            _chk_list.append(("ATM OI Build",
                               "BULLISH - PE {:+,.0f} > CE {:+,.0f} near ATM (put writers adding)".format(_near_pe_c, _near_ce_c), "#4CAF50"))
        elif _near_ce_c > _near_pe_c * 1.2:
            _chk_list.append(("ATM OI Build",
                               "BEARISH - CE {:+,.0f} > PE {:+,.0f} near ATM (call writers adding)".format(_near_ce_c, _near_pe_c), "#ef5350"))
        else:
            _chk_list.append(("ATM OI Build", "NEUTRAL - CE/PE OI balanced near ATM", "#FF9800"))
    else:
        _chk_list.append(("ATM OI Build", "No data available", "#FF9800"))

    # Check 4: IV level (Natenberg)
    if atm_iv < 15:
        _chk_list.append(("ATM IV {:.1f}%".format(atm_iv), "LOW IV - cheap to buy options (Natenberg: prefer buying)", "#4CAF50"))
    elif atm_iv > 25:
        _chk_list.append(("ATM IV {:.1f}%".format(atm_iv), "HIGH IV - expensive premiums (Natenberg: prefer selling)", "#ef5350"))
    else:
        _chk_list.append(("ATM IV {:.1f}%".format(atm_iv), "MODERATE IV - fair entry", "#FF9800"))

    # Check 5: 15-min OI flow direction from snapshots
    _sym_snaps15 = [s for s in st.session_state.get("oi_snapshots", [])
                    if s.get("symbol", symbol) == symbol]
    _tgt_15m = _now_ts - timedelta(minutes=15)
    _past_15m = [s for s in _sym_snaps15 if s["time"] <= _tgt_15m]
    if _past_15m:
        _old15 = _past_15m[-1]
        _ce_15 = _snap["total_ce_oi"] - _old15["total_ce_oi"]
        _pe_15 = _snap["total_pe_oi"] - _old15["total_pe_oi"]
        _net15 = _pe_15 - _ce_15
        if _net15 > 10000:
            _chk_list.append(("OI 15-min Flow",
                "PE {:+,.0f} / CE {:+,.0f} - PE writing pushing UP".format(int(_pe_15), int(_ce_15)), "#4CAF50"))
        elif _net15 < -10000:
            _chk_list.append(("OI 15-min Flow",
                "CE {:+,.0f} / PE {:+,.0f} - CE writing pushing DOWN".format(int(_ce_15), int(_pe_15)), "#ef5350"))
        else:
            _chk_list.append(("OI 15-min Flow",
                "Balanced CE {:+,.0f} / PE {:+,.0f}".format(int(_ce_15), int(_pe_15)), "#FF9800"))
    else:
        _chk_list.append(("OI 15-min Flow", "Collecting data - need 15 min of auto-refresh history", "#888888"))

    # Check 6: Overall signal (from generate_signal)
    _sig_c = "#4CAF50" if sig["signal"] == "BUY CALL" else "#ef5350" if sig["signal"] == "BUY PUT" else "#FF9800"
    _chk_list.append(("Signal", "{} ({}% confidence, score: {})".format(
        sig["signal"], sig["confidence"], sig.get("score", 0)), _sig_c))

    # Expiry proximity check
    _h_left_c = ea.hours_to_expiry(expiry)
    if 0 < _h_left_c <= 6.5:
        _chk_list.append(("Expiry", "TODAY - {:.1f} hrs left (Augen: gamma risk HIGH, scalp mode)".format(_h_left_c), "#ef5350"))
    elif 0 < _h_left_c <= 24:
        _chk_list.append(("Expiry", "TOMORROW - plan next expiry position (Natenberg: buy before IV crush)".format(), "#FF9800"))

    # Overall verdict
    _bull_c = sum(1 for _, _, c in _chk_list if c == "#4CAF50")
    _bear_c = sum(1 for _, _, c in _chk_list if c == "#ef5350")
    _neut_c = sum(1 for _, _, c in _chk_list if c == "#FF9800")
    if _bull_c >= 3 and _bull_c > _bear_c:
        _verdict, _vcolor = "BUY CALL", "#4CAF50"
    elif _bear_c >= 3 and _bear_c > _bull_c:
        _verdict, _vcolor = "BUY PUT", "#ef5350"
    elif _bull_c == 2 and _bull_c > _bear_c:
        _verdict, _vcolor = "LEAN BULLISH", "#8BC34A"
    elif _bear_c == 2 and _bear_c > _bull_c:
        _verdict, _vcolor = "LEAN BEARISH", "#FF7043"
    else:
        _verdict, _vcolor = "WAIT / NEUTRAL", "#FF9800"

    _rows_html = ""
    for _label, _desc, _color in _chk_list:
        _icon = "BULL" if _color == "#4CAF50" else "BEAR" if _color == "#ef5350" else "NEUT"
        _rows_html += (
            '<div style="display:flex;gap:10px;padding:4px 0;border-bottom:1px solid #2a2a3e;">'
            '<span style="color:{c};font-weight:bold;font-size:11px;min-width:44px;flex-shrink:0;">[{i}]</span>'
            '<span style="color:#bbb;font-size:12px;min-width:150px;flex-shrink:0;">{l}</span>'
            '<span style="color:{c};font-size:12px;">{d}</span>'
            '</div>'
        ).format(c=_color, i=_icon, l=_label, d=_desc)

    st.markdown(
        '<div style="background:#12122a;border:2px solid {vc};border-radius:12px;padding:14px;margin-bottom:14px;">'
        '<div style="font-size:11px;color:#888;letter-spacing:1px;margin-bottom:8px;">'
        'LIVE SIGNAL SUMMARY - Book Checkpoints (McMillan/Augen/Murphy/Natenberg)</div>'
        '{rows}'
        '<div style="margin-top:10px;text-align:center;padding:8px;background:#0d0d1e;border-radius:8px;">'
        '<span style="font-size:20px;font-weight:bold;color:{vc};">OVERALL: {v}</span>'
        '&nbsp;&nbsp;<span style="color:#888;font-size:12px;">{b} bullish / {br} bearish / {n} neutral</span>'
        '</div></div>'.format(
            vc=_vcolor, rows=_rows_html, v=_verdict,
            b=_bull_c, br=_bear_c, n=_neut_c
        ),
        unsafe_allow_html=True
    )

    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spot Price",    "Rs.{:,.1f}".format(underlying))
    c2.metric("PCR",           str(pcr), delta="Bullish" if pcr > 1.0 else "Bearish")
    c3.metric("Max Pain",      "Rs.{:,.0f}".format(max_pain))
    c4.metric("CE Resistance", "Rs.{:,.0f}".format(sig["max_ce_resistance"]), delta="Sell Wall")
    c5.metric("PE Support",    "Rs.{:,.0f}".format(sig["max_pe_support"]),    delta="Buy Wall")

    # ── OI Intelligence row ────────────────────────────────────────────────────
    oi1, oi2, oi3, oi4, oi5 = st.columns(5)
    call_sum_v = sig.get("call_sum", 0)
    put_sum_v  = sig.get("put_sum", 0)
    oi_diff_v  = sig.get("oi_difference", 0)
    itm_r      = sig.get("itm_ratio", 0)
    oi1.metric("Call Sum (ATM+-1)", "{:+.1f}K".format(call_sum_v),
               delta="CE writing up" if call_sum_v > 0 else "CE covering")
    oi2.metric("Put Sum (ATM+-1)", "{:+.1f}K".format(put_sum_v),
               delta="PE writing up" if put_sum_v > 0 else "PE covering")
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
            '<div style="background:{bg};border:3px solid {bc};padding:25px;border-radius:14px;text-align:center;">'
            '<div style="font-size:14px;color:#ccc;margin-bottom:6px;">TRADE SIGNAL</div>'
            '<div style="font-size:32px;font-weight:bold;color:{bc};">{sig}</div>'
            '<div style="font-size:16px;color:#ddd;margin-top:8px;">Confidence</div>'
            '<div style="font-size:42px;font-weight:bold;color:white;">{conf}%</div>'
            '</div>'.format(bg=bg, bc=border, sig=sig["signal"], conf=sig["confidence"]),
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
            icon = "[B]" if any(w in rl for w in ["bullish","support","upward","pe writing","put sum","itm ratio","covering"]) else \
                   "[S]" if any(w in rl for w in ["bearish","resistance","downward","ce writing","call sum"]) else "[N]"
            st.markdown(icon + " " + r)
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
                '<div style="background:#1e1e2e;padding:12px;border-radius:8px;border-left:4px solid {sc};">'
                '<b style="color:{sc};">{st}</b><br>'
                '<span style="color:#ccc;font-size:13px;">{sn}</span>'
                '</div>'.format(sc=strat_col, st=strat_type, sn=strat_note),
                unsafe_allow_html=True
            )

    st.divider()

    # ── Smart Trade Intelligence (Mark Douglas / Livermore / ICT / Van Tharp) ──
    st.markdown("### Smart Trade Intelligence")
    st.caption("Mark Douglas - Jesse Livermore - ICT/Smart Money - Van Tharp - Historical Expiry Learning")

    with st.spinner("Running multi-framework analysis..."):
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
            '<div style="font-size:14px;color:#eee;margin:6px 0;font-weight:bold;">{v}</div>'
            '<div style="color:#aaa;font-size:11px;">{ag}/{tot} indicators agree</div>'
            '<div style="color:#ccc;font-size:12px;margin-top:8px;">{adv}</div>'
            '</div>'.format(
                v=_douglas["verdict"],
                ag=max(_douglas["bull"], _douglas["bear"]),
                tot=_douglas["total"],
                adv=_douglas["advice"],
            ),
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown("**Multi-framework reasoning:**")
        for _ln in _reasons:
            st.markdown("- " + _ln)

    st.divider()
    _fa1, _fa2 = st.columns(2)
    with _fa1:
        with st.expander("Jesse Livermore - Trend, Stage and Pivots"):
            lv = _livermore
            _tc = "#4CAF50" if lv["trend"]=="UPTREND" else "#ef5350" if lv["trend"]=="DOWNTREND" else "#FF9800"
            st.markdown(
                '<div style="background:#1e1e2e;padding:14px;border-radius:10px;">'
                '<b style="color:{c};">{t}</b> &nbsp;|&nbsp;<span style="color:#ccc;">{s}</span><br><br>'
                '<span style="color:#aaa;font-size:12px;">5d momentum: <b>{m:+.2f}%</b> '
                '| EMA-slow: <b>Rs.{e:.0f}</b> | 20d range: <b>Rs.{r:.0f}</b></span><br>'
                '<i style="color:#ddd;font-size:12px;">{tape}</i></div>'.format(
                    c=_tc, t=lv["trend"], s=lv["stage"],
                    m=lv["mom5_pct"], e=lv["ema_slow"], r=lv["range20"], tape=lv["tape"]),
                unsafe_allow_html=True,
            )
            if lv.get("pivot_hi"):
                st.markdown("**Resistance:** " + " | ".join("Rs.{:,.0f}".format(p) for p in lv["pivot_hi"]))
            if lv.get("pivot_lo"):
                st.markdown("**Support:** " + " | ".join("Rs.{:,.0f}".format(p) for p in lv["pivot_lo"]))

        with st.expander("ICT / Smart Money - Institutional Footprints"):
            ic = _ict
            st.info(ic.get("narrative", ""))
            st.dataframe(pd.DataFrame({
                "Level": ["CE Wall (Resistance)", "PE Wall (Support)", "Max Pain Magnet"],
                "Strike": [ic.get("ce_wall",0), ic.get("pe_wall",0), ic.get("max_pain",0)],
                "Dist from Spot": ["Rs.{:+.0f}".format(ic.get("dist_ce",0)),
                                   "Rs.{:+.0f}".format(-ic.get("dist_pe",0)),
                                   "Rs.{:+.0f}".format(ic.get("dist_mp",0))],
            }), hide_index=True, use_container_width=True)

    with _fa2:
        with st.expander("Van Tharp - Position Sizing and R-Multiple"):
            vt = _tharp
            st.markdown(
                '<div style="background:#1e1e2e;padding:14px;border-radius:10px;">'
                '<b style="color:#FFD700;">1R = 1.5% of Rs.{cap:,.0f} capital</b><br><br>'
                '<table style="width:100%;color:#ccc;font-size:13px;">'
                '<tr><td>ATM Premium</td><td><b>Rs.{prem}</b></td></tr>'
                '<tr><td>Stop-Loss ({sl}%)</td><td style="color:#ef5350;"><b>Rs.{slp}</b></td></tr>'
                '<tr><td>Target (2R)</td><td style="color:#26a69a;"><b>Rs.{tgt}</b></td></tr>'
                '<tr><td>Lot Size</td><td><b>{ls} units</b></td></tr>'
                '<tr><td>Max Lots</td><td style="color:#FFD700;"><b>{ml}</b></td></tr>'
                '<tr><td>Risk/Lot</td><td><b>Rs.{rpl:,}</b></td></tr>'
                '<tr><td>Expectancy</td><td><b>{exp:.2f}R</b></td></tr>'
                '</table></div>'.format(
                    cap=vt["capital"], prem=vt["premium_est"], sl=int(vt["sl_pct"]),
                    slp=vt["sl_price"], tgt=vt["target_2r"], ls=vt["lot_size"],
                    ml=vt["max_lots"], rpl=int(vt["risk_per_lot"]), exp=vt["expectancy_r"]),
                unsafe_allow_html=True,
            )
            st.caption("Van Tharp: Size so 1 stop-out = 1.5% of capital. Never risk more than 2%.")

        with st.expander("Historical Expiry Learning - Pattern Match"):
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
                st.success("Tip: " + hist["lesson"])
            if hist.get("patterns"):
                _hdf = pd.DataFrame(hist["patterns"])[
                    ["date","weekday","rsi","ema_trend","range_pct","close_chg","outcome"]]
                _hdf.columns = ["Date","Day","RSI","EMA","Range%","Chg%","Outcome"]
                def _c_out(v):
                    return "color:#4CAF50" if v=="BULLISH" else "color:#ef5350" if v=="BEARISH" else "color:#FF9800"
                st.dataframe(_hdf.style.map(_c_out, subset=["Outcome"]),
                             hide_index=True, use_container_width=True)

    with st.expander("Mark Douglas - Full Confluence Grid"):
        _chk = pd.DataFrame(_douglas["checks"], columns=["Indicator","Signal"])
        _chk["Signal"] = _chk["Signal"].map({1:"BULL", -1:"BEAR", 0:"NEUTRAL"})
        st.dataframe(_chk, hide_index=True, use_container_width=True)
        st.markdown("**{} Bull | {} Bear | {} Neutral** of {} -> **{}**".format(
            _douglas["bull"], _douglas["bear"], _douglas["neutral"],
            _douglas["total"], _douglas["verdict"]))

    st.divider()

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("#### Open Interest Distribution - Murphy Support/Resistance")
        f = go.Figure()
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
        st.markdown("#### OI Change - Velocity & Direction (Net Flow)")
        net_oi_flow = (df["pe_chg_oi"] - df["ce_chg_oi"]) / 1000
        flow_colors = ["#26a69a" if v > 0 else "#ef5350" for v in net_oi_flow]
        f2 = go.Figure()
        f2.add_trace(go.Bar(x=df["strike"], y=df["ce_chg_oi"]/1000, name="CE Build",
                            marker_color="#ef5350", opacity=0.4))
        f2.add_trace(go.Bar(x=df["strike"], y=df["pe_chg_oi"]/1000, name="PE Build",
                            marker_color="#26a69a", opacity=0.4))
        f2.add_trace(go.Scatter(x=df["strike"], y=net_oi_flow, name="Net OI Flow (PE-CE)",
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
                         yaxis=dict(title="Chg OI (K) - +ve = PE writing (bullish)", gridcolor="#333"),
                         margin=dict(t=20, b=10))
        st.plotly_chart(f2, use_container_width=True)
        net_total = net_oi_flow.sum()
        flow_bias = "[B] PE writing dominates (Bullish)" if net_total > 0 else "[S] CE writing dominates (Bearish)"
        st.caption("Net OI Flow: {:+.1f}K -> {}".format(net_total, flow_bias))

    # ── OI Trend History ───────────────────────────────────────────────────────
    st.markdown("#### OI Trend History - Change per time window")
    _curr_ce_h = float(df["ce_oi"].sum()) if not df.empty else 0.0
    _curr_pe_h = float(df["pe_oi"].sum()) if not df.empty else 0.0
    _time_windows = [
        (1,   "1 min"),  (5,  "5 min"),  (10, "10 min"), (15, "15 min"), (30, "30 min"),
        (60,  "1 hr"),   (120, "2 hr"),  (180, "3 hr"),  (240, "4 hr"),  (300, "5 hr"), (360, "6 hr"),
    ]
    _hist_rows = []
    for _mins, _label in _time_windows:
        _target_t = _now_ts - timedelta(minutes=_mins)
        _past = [s for s in st.session_state.get("oi_snapshots", [])
                 if s.get("symbol", symbol) == symbol
                 if s["time"] <= _target_t]
        if _past:
            _old = _past[-1]
            _ce_chg = _curr_ce_h - _old["total_ce_oi"]
            _pe_chg = _curr_pe_h - _old["total_pe_oi"]
            _net    = _pe_chg - _ce_chg
            _pcr_chg = float(pcr) - _old.get("pcr", float(pcr))
            if _net > 10000:
                _bias = "PE building - Bullish"
            elif _net < -10000:
                _bias = "CE building - Bearish"
            else:
                _bias = "Balanced"
            _hist_rows.append({
                "Period":     _label,
                "CE OI Chg":  "{:+,.0f}".format(int(_ce_chg)),
                "PE OI Chg":  "{:+,.0f}".format(int(_pe_chg)),
                "Net Flow":   "{:+,.0f}".format(int(_net)),
                "PCR Chg":    "{:+.2f}".format(_pcr_chg),
                "Trend":      _bias,
            })
        else:
            _hist_rows.append({
                "Period":    _label,
                "CE OI Chg": "collecting...",
                "PE OI Chg": "collecting...",
                "Net Flow":  "collecting...",
                "PCR Chg":   "collecting...",
                "Trend":     "Need 1+ min of data",
            })

    _hist_df = pd.DataFrame(_hist_rows)

    def _color_hist_trend(val):
        if "Bullish" in str(val): return "color: #4CAF50"
        if "Bearish" in str(val): return "color: #ef5350"
        return "color: #FF9800"

    st.dataframe(
        _hist_df.style.map(_color_hist_trend, subset=["Trend"]),
        hide_index=True, use_container_width=True
    )
    st.caption(
        "OI snapshots collected every refresh (1+ min interval). Positive Net Flow = PE building (bullish). "
        "Negative = CE building (bearish). {} snapshots stored.".format(
            len(st.session_state.get("oi_snapshots", [])))
    )

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
        st.markdown("#### PCR + IV Skew Analysis")

        if   pcr > 1.5: interp, pcr_col = "EXTREMELY BULLISH",           "#00e676"
        elif pcr > 1.2: interp, pcr_col = "BULLISH - strong put writing", "#4CAF50"
        elif pcr > 0.8: interp, pcr_col = "NEUTRAL to BULLISH",           "#8BC34A"
        elif pcr > 0.5: interp, pcr_col = "NEUTRAL to BEARISH",           "#FF9800"
        else:           interp, pcr_col = "BEARISH - heavy call writing",  "#f44336"

        # IV Skew calculation
        _atm_idx_iv = int((df["strike"] - underlying).abs().idxmin()) if not df.empty else 0
        _atm_ce_iv  = float(df.iloc[_atm_idx_iv]["ce_iv"])  if not df.empty else 0.0
        _atm_pe_iv  = float(df.iloc[_atm_idx_iv]["pe_iv"])  if not df.empty else 0.0
        _iv_skew    = _atm_pe_iv - _atm_ce_iv  # positive = PE more expensive

        # IV trend: is CE side or PE side moving more?
        _ce_above = df[df["strike"] > underlying]["ce_iv"] if not df.empty else pd.Series(dtype=float)
        _pe_below = df[df["strike"] < underlying]["pe_iv"] if not df.empty else pd.Series(dtype=float)
        _ce_avg   = float(_ce_above[_ce_above > 0].mean()) if not _ce_above[_ce_above > 0].empty else _atm_ce_iv
        _pe_avg   = float(_pe_below[_pe_below > 0].mean()) if not _pe_below[_pe_below > 0].empty else _atm_pe_iv

        if _iv_skew > 1.5:
            _skew_txt  = "PE side higher by {:.1f}% - Put hedging demand (typical NSE pattern)".format(abs(_iv_skew))
            _skew_col  = "#26a69a"
        elif _iv_skew < -1.5:
            _skew_txt  = "CE side higher by {:.1f}% - Call buying dominant (bullish speculation)".format(abs(_iv_skew))
            _skew_col  = "#ef5350"
        else:
            _skew_txt  = "CE/PE IV roughly equal - No strong IV directional bias"
            _skew_col  = "#FF9800"

        _dominant_side = "PE side (puts)" if _ce_avg < _pe_avg else "CE side (calls)"

        st.markdown(
            '<div style="background:#1e1e2e;padding:16px;border-radius:12px;">'
            '<div style="font-size:16px;font-weight:bold;color:{pc};text-align:center;margin-bottom:4px;">{iv}</div>'
            '<div style="font-size:14px;color:{pc};text-align:center;margin-bottom:10px;">{interp}</div>'
            '<hr style="border-color:#333;margin:10px 0;">'
            '<table style="width:100%;font-size:13px;color:#ccc;">'
            '<tr><td>ATM CE IV</td><td style="text-align:right;color:#ef5350;font-weight:bold;">{atm_ce:.1f}%</td></tr>'
            '<tr><td>ATM PE IV</td><td style="text-align:right;color:#26a69a;font-weight:bold;">{atm_pe:.1f}%</td></tr>'
            '<tr><td>IV Skew (PE-CE)</td><td style="text-align:right;color:{sc};font-weight:bold;">{sk:+.1f}%</td></tr>'
            '<tr><td>Higher IV side</td><td style="text-align:right;color:{sc};">{ds}</td></tr>'
            '</table>'
            '<div style="margin-top:8px;color:{sc};font-size:12px;">{st}</div>'
            '<hr style="border-color:#333;margin:10px 0;">'
            '<div style="font-size:11px;color:#888;">'
            'PCR &lt;0.5=Bearish | 0.8-1.2=Neutral | &gt;1.5=Extremely Bullish'
            '</div></div>'.format(
                pc=pcr_col, iv=pcr, interp=interp,
                atm_ce=_atm_ce_iv, atm_pe=_atm_pe_iv,
                sc=_skew_col, sk=_iv_skew, ds=_dominant_side, st=_skew_txt
            ),
            unsafe_allow_html=True
        )

    st.divider()

    # ── Expiry proximity alert ────────────────────────────────────────────────
    h_left = ea.hours_to_expiry(expiry)

    if 0 < h_left <= 6.5:
        st.markdown(
            '<div style="background:#3d0000;border:2px solid #ef5350;padding:12px;border-radius:10px;margin-bottom:10px;">'
            '<b style="color:#ef5350;">EXPIRY DAY</b> - {:.1f} market hours left | '
            'Switch to <b>[Exp] Expiry Signals</b> tab for scalp targets</div>'.format(h_left),
            unsafe_allow_html=True
        )
    elif 0 < h_left <= 24:
        next_info = ea.next_expiry_info(
            meta.get("all_expiries", [expiry]), expiry, underlying, atm_iv)
        if "error" not in next_info:
            st.markdown(
                '<div style="background:#1a2a00;border:2px solid #8BC34A;padding:12px;border-radius:10px;margin-bottom:10px;">'
                '<b style="color:#8BC34A;">Expiry Tomorrow</b> - Next expiry: <b>{}</b> | '
                'Switch to <b>[Exp] Expiry Signals</b> for next-expiry analysis</div>'.format(
                    next_info.get("next_expiry", "")),
                unsafe_allow_html=True
            )

    st.markdown("### Prediction - {} - ATM IV: {}%".format(
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
            '<div style="color:#aaa;font-size:13px;">Expected Daily Range (IV={iv}%)</div>'
            '<div style="font-size:22px;font-weight:bold;color:#26a69a;margin:8px 0;">+/-Rs.{mv:,.0f}</div>'
            '<div style="color:#ccc;font-size:14px;">Upper: <b>Rs.{up:,.0f}</b></div>'
            '<div style="color:#ccc;font-size:14px;">Lower: <b>Rs.{lo:,.0f}</b></div>'
            '</div>'.format(iv=round(atm_iv, 1), mv=daily_move, up=upper_level, lo=lower_level),
            unsafe_allow_html=True
        )
    with tp2:
        st.markdown(
            '<div style="background:#1e1e2e;padding:18px;border-radius:12px;text-align:center;">'
            '<div style="color:#aaa;font-size:13px;">Directional Bias</div>'
            '<div style="font-size:32px;font-weight:bold;color:{dc};margin:8px 0;">{dir}</div>'
            '<div style="color:#ccc;font-size:13px;">Score: {sc}</div>'
            '<div style="color:#ccc;font-size:13px;">PCR: {pcr}</div>'
            '</div>'.format(dc=dir_color, dir=direction, sc=bias_score, pcr=pcr),
            unsafe_allow_html=True
        )
    with tp3:
        mp_dir  = "above" if pain_diff > 0 else "below"
        mp_pull = "Upward pull" if pain_diff > 0 else "Downward pull"
        st.markdown(
            '<div style="background:#1e1e2e;padding:18px;border-radius:12px;text-align:center;">'
            '<div style="color:#aaa;font-size:13px;">Max Pain Magnet</div>'
            '<div style="font-size:22px;font-weight:bold;color:#FFD700;margin:8px 0;">Rs.{mp:,.0f}</div>'
            '<div style="color:#ccc;font-size:13px;">Rs.{pd:,.0f} {dir} spot</div>'
            '<div style="color:#ccc;font-size:13px;">{pull}</div>'
            '</div>'.format(mp=max_pain, pd=abs(pain_diff), dir=mp_dir, pull=mp_pull),
            unsafe_allow_html=True
        )

    st.markdown(
        '<div style="background:#1e1e2e;padding:15px;border-radius:10px;margin-top:10px;">'
        '<b style="color:#aaa;">Key Levels</b><br>'
        '<span style="color:#ef5350;">Resistance: Rs.{res:,.0f} (max CE OI)  |  Upper: Rs.{up:,.0f}</span><br>'
        '<span style="color:#26a69a;">Support: Rs.{sup:,.0f} (max PE OI)  |  Lower: Rs.{lo:,.0f}</span><br>'
        '<span style="color:#FFD700;">Max Pain: Rs.{mp:,.0f}  |  Spot: Rs.{sp:,.0f}</span>'
        '</div>'.format(
            res=sig["max_ce_resistance"], up=upper_level,
            sup=sig["max_pe_support"],    lo=lower_level,
            mp=max_pain, sp=underlying
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


# =============================================================================
# TAB 2 - SENSEX LIVE DASHBOARD
# =============================================================================
with tab_sensex:
    st.markdown("## SENSEX Option Chain - Live")

    with st.spinner("Fetching SENSEX data..."):
        _sx_df, _sx_meta = _get_sensex_data(data_source, num_strikes)

    _sx_underlying = float(_sx_meta.get("underlying", 80000))
    _sx_expiry     = _sx_meta.get("expiry", "N/A")
    _sx_src        = _sx_meta.get("source", data_source)

    # Save SENSEX snapshot for OI history
    if not _sx_df.empty:
        _sx_sn = {
            "time":        _now_ts,
            "symbol":      "SENSEX",
            "total_ce_oi": float(_sx_df["ce_oi"].sum()),
            "total_pe_oi": float(_sx_df["pe_oi"].sum()),
            "pcr":         float(_sx_meta.get("pcr", 1.0)),
            "underlying":  _sx_underlying,
        }
        _all_snaps = st.session_state.get("oi_snapshots", [])
        _sx_prev   = [s for s in _all_snaps if s.get("symbol") == "SENSEX"]
        if not _sx_prev or (_now_ts - _sx_prev[-1]["time"]).total_seconds() >= 60:
            _all_snaps.append(_sx_sn)
            st.session_state["oi_snapshots"] = [
                s for s in _all_snaps if s["time"] >= _now_ts - timedelta(hours=7)
            ]
            _save_snaps_to_disk(st.session_state["oi_snapshots"])

    st.markdown("**Spot: Rs.{:,.2f}**  |  Expiry: **{}**  |  Source: `{}`".format(
        _sx_underlying, _sx_expiry, _sx_src))

    if _sx_df.empty:
        st.warning("SENSEX option chain not available via NSE (it trades on BSE). Showing demo data.")

    if not _sx_df.empty:
        _sx_sig = generate_signal(_sx_df, _sx_meta)
    else:
        _sx_demo_df, _sx_demo_meta = generate_demo_option_chain("SENSEX")
        _sx_sig = generate_signal(_sx_demo_df, _sx_demo_meta)

    _sx_pcr      = float(_sx_sig.get("pcr", 1.0))
    _sx_maxpain  = float(_sx_sig.get("max_pain", _sx_underlying))
    _sx_atm_idx  = int((_sx_df["strike"] - _sx_underlying).abs().idxmin()) if not _sx_df.empty else 0
    _sx_atm_iv   = float(_sx_df.iloc[_sx_atm_idx]["ce_iv"]) if not _sx_df.empty else 0.0
    if _sx_atm_iv == 0 and not _sx_df.empty:
        _sx_atm_iv = float(_sx_df["ce_iv"][_sx_df["ce_iv"] > 0].mean()) if (_sx_df["ce_iv"] > 0).any() else 15.0

    # Signal card
    sx1, sx2, sx3, sx4, sx5 = st.columns(5)
    sx1.metric("SENSEX Spot",   "Rs.{:,.0f}".format(_sx_underlying))
    sx2.metric("PCR",           str(_sx_pcr), delta="Bullish" if _sx_pcr > 1.0 else "Bearish")
    sx3.metric("Max Pain",      "Rs.{:,.0f}".format(_sx_maxpain))
    sx4.metric("CE Resistance", "Rs.{:,.0f}".format(_sx_sig.get("max_ce_resistance", _sx_underlying)))
    sx5.metric("PE Support",    "Rs.{:,.0f}".format(_sx_sig.get("max_pe_support", _sx_underlying)))

    st.divider()

    # Build checkpoints (same logic as NIFTY tab)
    _sx_chk = []
    if _sx_pcr > 1.3:
        _sx_chk.append(("PCR {:.2f}".format(_sx_pcr), "BULLISH - PE writing dominant", "#4CAF50"))
    elif _sx_pcr < 0.7:
        _sx_chk.append(("PCR {:.2f}".format(_sx_pcr), "BEARISH - CE writing dominant", "#ef5350"))
    else:
        _sx_chk.append(("PCR {:.2f}".format(_sx_pcr), "NEUTRAL (0.7-1.3 range)", "#FF9800"))

    _sx_pain_diff = _sx_maxpain - _sx_underlying
    if _sx_pain_diff > _sx_underlying * 0.001:
        _sx_chk.append(("Max Pain Rs.{:,.0f}".format(_sx_maxpain),
            "UPWARD PULL Rs.{:,.0f} above spot".format(abs(_sx_pain_diff)), "#4CAF50"))
    elif _sx_pain_diff < -_sx_underlying * 0.001:
        _sx_chk.append(("Max Pain Rs.{:,.0f}".format(_sx_maxpain),
            "DOWNWARD PULL Rs.{:,.0f} below spot".format(abs(_sx_pain_diff)), "#ef5350"))
    else:
        _sx_chk.append(("Max Pain Rs.{:,.0f}".format(_sx_maxpain), "AT SPOT - no directional pull", "#FF9800"))

    if not _sx_df.empty:
        _sx_near = _sx_df.iloc[max(0, _sx_atm_idx-2):min(len(_sx_df), _sx_atm_idx+3)]
        _sx_nce  = _sx_near["ce_chg_oi"].sum()
        _sx_npe  = _sx_near["pe_chg_oi"].sum()
        if _sx_npe > _sx_nce * 1.2:
            _sx_chk.append(("ATM OI Build",
                "BULLISH - PE {:+,.0f} > CE {:+,.0f} near ATM".format(_sx_npe, _sx_nce), "#4CAF50"))
        elif _sx_nce > _sx_npe * 1.2:
            _sx_chk.append(("ATM OI Build",
                "BEARISH - CE {:+,.0f} > PE {:+,.0f} near ATM".format(_sx_nce, _sx_npe), "#ef5350"))
        else:
            _sx_chk.append(("ATM OI Build", "NEUTRAL - CE/PE OI balanced near ATM", "#FF9800"))

    if _sx_atm_iv < 15:
        _sx_chk.append(("ATM IV {:.1f}%".format(_sx_atm_iv), "LOW IV - cheap to buy", "#4CAF50"))
    elif _sx_atm_iv > 25:
        _sx_chk.append(("ATM IV {:.1f}%".format(_sx_atm_iv), "HIGH IV - expensive premiums", "#ef5350"))
    else:
        _sx_chk.append(("ATM IV {:.1f}%".format(_sx_atm_iv), "MODERATE IV - fair entry", "#FF9800"))

    # 15-min OI flow for SENSEX
    _sx_snaps15 = [s for s in st.session_state.get("oi_snapshots", []) if s.get("symbol") == "SENSEX"]
    _sx_snap_curr_ce = float(_sx_df["ce_oi"].sum()) if not _sx_df.empty else 0.0
    _sx_snap_curr_pe = float(_sx_df["pe_oi"].sum()) if not _sx_df.empty else 0.0
    _sx_past15 = [s for s in _sx_snaps15 if s["time"] <= _now_ts - timedelta(minutes=15)]
    if _sx_past15:
        _sx_old15 = _sx_past15[-1]
        _sx_ce15  = _sx_snap_curr_ce - _sx_old15["total_ce_oi"]
        _sx_pe15  = _sx_snap_curr_pe - _sx_old15["total_pe_oi"]
        _sx_net15 = _sx_pe15 - _sx_ce15
        if _sx_net15 > 10000:
            _sx_chk.append(("OI 15-min Flow", "PE {:+,.0f} / CE {:+,.0f} - Bullish push".format(int(_sx_pe15), int(_sx_ce15)), "#4CAF50"))
        elif _sx_net15 < -10000:
            _sx_chk.append(("OI 15-min Flow", "CE {:+,.0f} / PE {:+,.0f} - Bearish push".format(int(_sx_ce15), int(_sx_pe15)), "#ef5350"))
        else:
            _sx_chk.append(("OI 15-min Flow", "Balanced CE {:+,.0f} / PE {:+,.0f}".format(int(_sx_ce15), int(_sx_pe15)), "#FF9800"))
    else:
        _sx_chk.append(("OI 15-min Flow", "Collecting (need 15 min of auto-refresh)", "#888888"))

    _sx_sig_c = "#4CAF50" if _sx_sig["signal"] == "BUY CALL" else "#ef5350" if _sx_sig["signal"] == "BUY PUT" else "#FF9800"
    _sx_chk.append(("Signal", "{} ({}% conf)".format(_sx_sig["signal"], _sx_sig["confidence"]), _sx_sig_c))

    _sx_bull = sum(1 for _, _, c in _sx_chk if c == "#4CAF50")
    _sx_bear = sum(1 for _, _, c in _sx_chk if c == "#ef5350")
    _sx_neut = sum(1 for _, _, c in _sx_chk if c == "#FF9800")
    if _sx_bull >= 3 and _sx_bull > _sx_bear:
        _sx_v, _sx_vc = "BUY CALL", "#4CAF50"
    elif _sx_bear >= 3 and _sx_bear > _sx_bull:
        _sx_v, _sx_vc = "BUY PUT", "#ef5350"
    elif _sx_bull == 2 and _sx_bull > _sx_bear:
        _sx_v, _sx_vc = "LEAN BULLISH", "#8BC34A"
    elif _sx_bear == 2 and _sx_bear > _sx_bull:
        _sx_v, _sx_vc = "LEAN BEARISH", "#FF7043"
    else:
        _sx_v, _sx_vc = "WAIT / NEUTRAL", "#FF9800"

    _sx_rows_html = ""
    for _sl, _sd, _sc in _sx_chk:
        _si = "BULL" if _sc == "#4CAF50" else "BEAR" if _sc == "#ef5350" else "NEUT"
        _sx_rows_html += (
            '<div style="display:flex;gap:10px;padding:4px 0;border-bottom:1px solid #2a2a3e;">'
            '<span style="color:{c};font-weight:bold;font-size:11px;min-width:44px;flex-shrink:0;">[{i}]</span>'
            '<span style="color:#bbb;font-size:12px;min-width:150px;flex-shrink:0;">{l}</span>'
            '<span style="color:{c};font-size:12px;">{d}</span>'
            '</div>'
        ).format(c=_sc, i=_si, l=_sl, d=_sd)

    st.markdown(
        '<div style="background:#12122a;border:2px solid {vc};border-radius:12px;padding:14px;margin-bottom:14px;">'
        '<div style="font-size:11px;color:#888;letter-spacing:1px;margin-bottom:8px;">'
        'SENSEX LIVE SIGNAL SUMMARY - Book Checkpoints</div>'
        '{rows}'
        '<div style="margin-top:10px;text-align:center;padding:8px;background:#0d0d1e;border-radius:8px;">'
        '<span style="font-size:20px;font-weight:bold;color:{vc};">OVERALL: {v}</span>'
        '&nbsp;&nbsp;<span style="color:#888;font-size:12px;">{b} bullish / {br} bearish / {n} neutral</span>'
        '</div></div>'.format(
            vc=_sx_vc, rows=_sx_rows_html, v=_sx_v,
            b=_sx_bull, br=_sx_bear, n=_sx_neut
        ),
        unsafe_allow_html=True
    )

    st.divider()

    # SENSEX Option chain table
    if not _sx_df.empty:
        sxc1, sxc2 = st.columns(2)
        with sxc1:
            st.markdown("#### Signal Gauge")
            _sx_bg  = {"green": "#1a7a1a", "red": "#8b1a1a", "orange": "#7a5c00"}.get(_sx_sig.get("color",""), "#333")
            _sx_bc  = {"green": "#4CAF50", "red": "#f44336", "orange": "#FF9800"}.get(_sx_sig.get("color",""), "#888")
            st.markdown(
                '<div style="background:{bg};border:3px solid {bc};padding:20px;border-radius:14px;text-align:center;">'
                '<div style="font-size:28px;font-weight:bold;color:{bc};">{sig}</div>'
                '<div style="font-size:36px;font-weight:bold;color:white;">{conf}%</div>'
                '</div>'.format(bg=_sx_bg, bc=_sx_bc, sig=_sx_sig["signal"], conf=_sx_sig["confidence"]),
                unsafe_allow_html=True
            )
        with sxc2:
            st.markdown("#### Key Levels")
            st.dataframe(pd.DataFrame({
                "Level": ["SENSEX Spot", "Max Pain", "CE Resistance", "PE Support"],
                "Price": [_sx_underlying, _sx_maxpain,
                          _sx_sig.get("max_ce_resistance", _sx_underlying),
                          _sx_sig.get("max_pe_support", _sx_underlying)],
            }), hide_index=True, use_container_width=True)

        st.divider()
        st.markdown("#### SENSEX OI Trend History")
        _sx_tw = [
            (1,"1 min"),(5,"5 min"),(10,"10 min"),(15,"15 min"),(30,"30 min"),
            (60,"1 hr"),(120,"2 hr"),(180,"3 hr"),(240,"4 hr"),(300,"5 hr"),(360,"6 hr"),
        ]
        _sx_hist_rows = []
        for _sm, _sl in _sx_tw:
            _sx_tgt = _now_ts - timedelta(minutes=_sm)
            _sx_past = [s for s in _sx_snaps15 if s["time"] <= _sx_tgt]
            if _sx_past:
                _sx_o = _sx_past[-1]
                _sx_cchg = _sx_snap_curr_ce - _sx_o["total_ce_oi"]
                _sx_pchg = _sx_snap_curr_pe - _sx_o["total_pe_oi"]
                _sx_net  = _sx_pchg - _sx_cchg
                _sx_pcr_c = _sx_pcr - _sx_o.get("pcr", _sx_pcr)
                _sx_bias = "PE building - Bullish" if _sx_net > 10000 else \
                           "CE building - Bearish" if _sx_net < -10000 else "Balanced"
                _sx_hist_rows.append({
                    "Period": _sl, "CE OI Chg": "{:+,.0f}".format(int(_sx_cchg)),
                    "PE OI Chg": "{:+,.0f}".format(int(_sx_pchg)),
                    "Net Flow": "{:+,.0f}".format(int(_sx_net)),
                    "PCR Chg": "{:+.2f}".format(_sx_pcr_c), "Trend": _sx_bias,
                })
            else:
                _sx_hist_rows.append({
                    "Period": _sl, "CE OI Chg": "collecting...", "PE OI Chg": "collecting...",
                    "Net Flow": "collecting...", "PCR Chg": "collecting...", "Trend": "Need data",
                })
        _sx_hdf = pd.DataFrame(_sx_hist_rows)
        st.dataframe(
            _sx_hdf.style.map(_color_hist_trend, subset=["Trend"]),
            hide_index=True, use_container_width=True
        )

        st.divider()
        st.markdown("#### SENSEX Option Chain (Near ATM)")
        _sx_disp_cols = ["strike", "ce_oi", "ce_chg_oi", "ce_ltp", "ce_iv",
                         "pe_iv", "pe_ltp", "pe_chg_oi", "pe_oi"]
        _sx_disp_cols = [c for c in _sx_disp_cols if c in _sx_df.columns]
        st.dataframe(_sx_df[_sx_disp_cols].reset_index(drop=True),
                     hide_index=True, use_container_width=True, height=350)

    st.caption("SENSEX options trade on BSE. Data may be demo/simulated when using NSE Direct source. "
               "Use Angel One (Live) for real BSE SENSEX data.")


# =============================================================================
# TAB 3 - MARKET SCANNER
# =============================================================================
with tab_scan:
    st.markdown("## Market Scanner")
    st.markdown(
        "Multi-instrument trending analysis using **Murphy OI+Price 4-scenario**, "
        "**Natenberg IV Rank**, and **McMillan PCR** methodologies."
    )

    sc1, sc2 = st.columns([1, 3])
    with sc1:
        scan_stocks = st.checkbox("Include F&O Stocks", value=False)
        scan_demo_mode = (data_source == "Demo Mode")

    with sc2:
        st.markdown(
            "**Murphy (1999) OI Framework:** Rising price + Rising OI = Bullish (fresh longs) · "
            "Rising price + Falling OI = Short covering (weak) · "
            "Falling price + Rising OI = Bearish (fresh shorts)"
        )

    if st.button("Scan Now", type="primary", key="scan_btn"):
        with st.spinner("Scanning instruments..."):
            if scan_demo_mode or data_source != "Angel One (Live)":
                scan_results = sc.scan_demo(include_stocks=scan_stocks)
            else:
                scan_results = sc.scan_live(symbol, df, meta, sig, include_stocks=scan_stocks)
        st.session_state["scan_results"] = scan_results

    scan_results = st.session_state.get("scan_results")
    if not scan_results:
        st.info("Click **Scan Now** to analyse trending instruments.")
    else:
        bullish  = [r for r in scan_results if r["signal"] == "BUY CALL"]
        bearish  = [r for r in scan_results if r["signal"] == "BUY PUT"]
        watching = [r for r in scan_results if r["signal"] == "WATCH"]

        sb1, sb2, sb3 = st.columns(3)
        sb1.metric("Bullish",  len(bullish))
        sb2.metric("Bearish",  len(bearish))
        sb3.metric("Watch",    len(watching))
        st.divider()

        st.markdown("### Ranked by Conviction (strongest first)")
        for r in scan_results:
            sig_icon = "[B]" if r["signal"] == "BUY CALL" else "[S]" if r["signal"] == "BUY PUT" else "[N]"

            with st.expander("{} **{}** - {} (conviction: {:+.0f})".format(
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

                st.markdown("**Murphy OI Signal:** {} - {}".format(r["murphy_signal"], r["murphy_note"]))
                st.caption("Natenberg: {}".format(r["iv_note"]))

        st.divider()

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
            "Sources: Murphy J.J. (1999) Technical Analysis of Financial Markets - "
            "Natenberg S. (2015) Option Volatility & Pricing - McMillan L.G. (2012) Options as a Strategic Investment"
        )


# =============================================================================
# TAB 3 - EXPIRY SIGNALS
# =============================================================================
with tab_exp:
    st.markdown("## Expiry Signals")

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

    if h_left_exp <= 0:
        st.error("Current expiry {} has passed. Showing analysis for reference.".format(expiry))
    elif h_left_exp <= 6.5:
        st.markdown(
            '<div style="background:#2d0000;border:2px solid #ef5350;padding:15px;border-radius:12px;">'
            '<h3 style="color:#ef5350;margin:0;">EXPIRY DAY - {:.1f} market hours left</h3>'
            '<p style="color:#ccc;margin:5px 0 0 0;">Augen (2009): Max Pain magnet effect strongest in last 2 hours. '
            'Gamma spikes near ATM - scalp targets 30-80 points on NIFTY.</p>'
            '</div>'.format(h_left_exp),
            unsafe_allow_html=True
        )
    elif h_left_exp <= 24:
        st.warning("Expiry **{}** is tomorrow. Check next expiry below.".format(expiry))
    else:
        st.info("Current expiry: **{}** - {:.1f} market hours remaining".format(expiry, h_left_exp))

    st.divider()

    exp_col, next_col = st.columns(2)

    with exp_col:
        st.markdown("### Expiry Scalp Signal (Augen Framework)")
        st.caption("Volume + OI change + Max Pain - 30-50 point targets")

        exp_sig = ea.expiry_scalp_signal(df, meta, sig, atm_iv=atm_iv_exp)

        sig_icon = "[B]" if exp_sig["direction"] == "BUY CALL" else \
                   "[S]" if exp_sig["direction"] == "BUY PUT" else "[N]"
        sig_bg   = {"BUY CALL": "#0a2a0a", "BUY PUT": "#2a0a0a"}.get(exp_sig["direction"], "#2a2a00")
        sig_col_ = {"BUY CALL": "#4CAF50", "BUY PUT": "#ef5350"}.get(exp_sig["direction"], "#FF9800")

        st.markdown(
            '<div style="background:{bg};border:2px solid {sc};padding:20px;border-radius:12px;text-align:center;">'
            '<div style="font-size:28px;font-weight:bold;color:{sc};">{icon} {dir}</div>'
            '<div style="color:#ccc;margin-top:8px;">{act}</div>'
            '</div>'.format(bg=sig_bg, sc=sig_col_, icon=sig_icon,
                            dir=exp_sig["direction"], act=exp_sig["action"]),
            unsafe_allow_html=True
        )

        st.markdown("**Signal Checks:**")
        for r in exp_sig["reasons"]:
            st.markdown(r)

        er = exp_sig["expected_range"]
        st.markdown("---")
        st.markdown("**Expected Range (Natenberg IV-based)**")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Upper", "Rs.{:,.0f}".format(er["upper"]))
        rc2.metric("Move +-", "Rs.{:,.0f}".format(er["move_pts"]))
        rc3.metric("Lower", "Rs.{:,.0f}".format(er["lower"]))

        st.markdown("---")
        st.markdown("**Expiry Scalp Parameters**")
        prem = exp_sig["atm_premium_est"]
        sl_p = round(prem * (1 - exp_sig["sl_pct"]), 1)
        tg_p = round(prem * (1 + exp_sig["tgt_pct"]), 1)
        st.dataframe(pd.DataFrame({
            "Parameter": ["ATM Premium (est.)", "Stop-Loss (-30%)", "Target (+50%)",
                          "Points Target", "Lot Size", "Best Hold"],
            "Value":     ["Rs.{:.1f}".format(prem),
                          "Rs.{:.1f}".format(sl_p),
                          "Rs.{:.1f}".format(tg_p),
                          "+/-{} pts (NIFTY)".format(exp_sig["pts_target"]),
                          "{} units".format(exp_sig["lot_size"]),
                          "30-60 min or 3:15 PM"]
        }), hide_index=True, use_container_width=True)

    with next_col:
        st.markdown("### Next Expiry Analysis")

        all_exp = meta.get("all_expiries", [expiry])
        nxt = ea.next_expiry_info(all_exp, expiry, underlying, atm_iv_exp)

        if "error" in nxt:
            st.info("Next expiry data not available (need 2+ expiry dates in chain).")
        else:
            st.markdown("**Next Expiry: {}**".format(nxt["next_expiry"]))
            st.markdown("{} trading days away - ATM IV for next: {}%".format(
                nxt["days_to_next"], nxt["iv_for_next"]))

            nr = nxt["expected_range"]
            nc1, nc2, nc3 = st.columns(3)
            nc1.metric("Upper", "Rs.{:,.0f}".format(nr["upper"]))
            nc2.metric("Move +-", "Rs.{:,.0f}".format(nr["move_pts"]))
            nc3.metric("Lower", "Rs.{:,.0f}".format(nr["lower"]))

            st.markdown("---")
            st.markdown("**Strategy Recommendation (Natenberg)**")
            st.markdown(
                '<div style="background:#1e1e2e;padding:15px;border-radius:10px;">'
                '<b style="color:#26a69a;font-size:18px;">{strat}</b><br>'
                '<span style="color:#ccc;">{why}</span>'
                '</div>'.format(strat=nxt["strategy"], why=nxt["strategy_why"]),
                unsafe_allow_html=True
            )

        st.markdown("---")

        st.markdown("### Gamma Hot Zones (Augen)")
        st.caption("Strikes where OI creates strongest magnet effect - expiry day key levels")
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

    st.markdown("### Expiry Day Trading Methodology")
    with st.expander("Augen (2009) - Expiry Day Framework", expanded=False):
        st.markdown("""
**From "Trading Options at Expiration" by Jeff Augen:**

1. **Gamma Spike**: ATM options gain gamma exponentially on expiry day.
   - A 0.5% move in the underlying can double/halve an ATM option price
   - Trade smaller size - premium moves are violent

2. **Max Pain Magnet**: In last 2 hours, underlying gravitates toward Max Pain
   - Below Max Pain = call writers buy back = price rises
   - Above Max Pain = put writers buy back = price falls

3. **Time-of-day patterns (NSE Expiry)**:
   - 9:20-10:30 AM: Gap fills and initial direction established
   - 12:00-1:30 PM: Lull - avoid trading, volume low
   - 2:00-3:15 PM: Max Pain pull strongest - last hour most reliable

4. **Target sizing for NIFTY**:
   - 5 min expiry scalp: +-15-25 pts
   - 15 min expiry trade: +-30-50 pts
   - Full session: +-60-100 pts
   - Set tight SL (25-30% of premium) - gamma can reverse quickly

5. **Strike selection on expiry day**:
   - ATM = 0.5 delta = best for directional trades
   - 1-OTM = cheaper but lower delta, harder to recover
        """)

    with st.expander("Murphy (1999) - OI Analysis on Expiry", expanded=False):
        st.markdown("""
**From "Technical Analysis of Financial Markets" (Chapter 7 - Volume and OI):**

| OI Change | Price Change | Interpretation | Action |
|-----------|-------------|----------------|--------|
| Rising OI + Rising Price | -> | Fresh longs entering = **Strong Bullish** | BUY CALL |
| Rising OI + Falling Price | -> | Fresh shorts entering = **Strong Bearish** | BUY PUT |
| Falling OI + Rising Price | -> | Short covering (weak move) = **Rally may fade** | CAUTION |
| Falling OI + Falling Price | -> | Long liquidation (selling easing) = **Recovery near** | WATCH |

On expiry day, **net OI flow near ATM** (within +-2 strikes) is the key signal:
- PE writers adding OI at support = market expects to hold support = BUY CALL
- CE writers adding OI at resistance = market expects to hold resistance = BUY PUT
        """)

    st.caption("Educational only - not financial advice. Options trading involves significant risk.")


# =============================================================================
# TAB 4 - BACKTEST & MODEL TRAINING
# =============================================================================
with tab_bt:
    st.markdown("## Backtest & Model Training")
    st.markdown("Analyse historical signal accuracy and P&L using index OHLCV data.")

    left_bt, right_bt = st.columns([1, 3])

    with left_bt:
        bt_mode = st.radio(
            "Mode",
            ["A - Signal Accuracy", "B - P&L Simulation", "C - Forward Tracker", "Train Model"],
        )
        bt_days = st.slider("Historical Days", 20, 90, 30, step=5)
        if data_source == "Angel One (Live)":
            st.success("Using Angel One historical candles")
        else:
            st.info("Using synthetic candles\n(select Angel One for real data)")

    with right_bt:
        candles = _get_candles(symbol, is_index, bt_days, data_source)

        if bt_mode.startswith("A"):
            st.markdown("### Signal Accuracy Backtest")
            st.markdown(
                "Uses **RSI-14 + EMA-9/EMA-21 crossover + volume ratio** consensus "
                "on {} daily candles. Signals fire only when 2+ indicators agree - "
                "higher selectivity means fewer trades but better quality signals. "
                "**Note:** Synthetic demo data is a random walk; near-50% win rate "
                "on synthetic data is expected. Use Angel One for real results.".format(bt_days)
            )
            if st.button("Run Accuracy Backtest", type="primary", key="run_acc"):
                with st.spinner("Running backtest..."):
                    metrics, rdf = backtest.run_accuracy_backtest(candles)
                if not metrics:
                    st.error("Not enough data (need 12+ candles).")
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

        elif bt_mode.startswith("B"):
            st.markdown("### P&L Simulation (Rs.)")
            st.markdown(
                "Buys 1 lot ATM option at estimated premium (0.8% of index price). "
                "Win = 1.5x premium, Loss = 0.4x premium (approximate)."
            )
            lot_defaults = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 40,
                            "MIDCPNIFTY": 50, "SENSEX": 20}
            sim_lot  = st.number_input("Lot Size", value=lot_defaults.get(symbol.upper(), 75),
                                        min_value=1, max_value=500)
            sim_comm = st.number_input("Commission per trade (Rs.)", value=40,
                                        min_value=0, max_value=500)

            if st.button("Run P&L Simulation", type="primary", key="run_pnl"):
                with st.spinner("Simulating P&L..."):
                    metrics, rdf = backtest.run_pnl_simulation(candles, int(sim_lot), int(sim_comm))
                if not metrics:
                    st.error("Not enough data (need 12+ candles).")
                else:
                    pnl_color = "#26a69a" if metrics["total_pnl"] >= 0 else "#ef5350"
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total P&L", "Rs.{:,}".format(metrics["total_pnl"]),
                               delta="Profit" if metrics["total_pnl"] >= 0 else "Loss")
                    m2.metric("Win Rate",   "{}%".format(metrics["win_rate"]))
                    m3.metric("Max Drawdown", "Rs.{:,}".format(metrics["max_drawdown"]))
                    m4.metric("Sharpe",     str(metrics["sharpe_ratio"]))

                    m5, m6, m7 = st.columns(3)
                    m5.metric("Best Trade",  "Rs.{:,}".format(metrics["best_trade"]))
                    m6.metric("Worst Trade", "Rs.{:,}".format(metrics["worst_trade"]))
                    m7.metric("Avg P&L",     "Rs.{:,}".format(int(metrics["avg_pnl"])))

                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Scatter(
                        x=rdf["date"], y=rdf["equity"],
                        mode="lines", fill="tozeroy",
                        line=dict(color=pnl_color, width=2),
                        name="Cumulative P&L (Rs.)"
                    ))
                    fig_pnl.update_layout(
                        title="Cumulative P&L Curve",
                        height=280,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(gridcolor="#333", tickangle=-45),
                        yaxis=dict(title="Rs.", gridcolor="#333"),
                        margin=dict(t=40, b=30)
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)
                    st.dataframe(rdf[["date","close","signal","premium","pnl","result","equity"]],
                                 use_container_width=True, height=280)

        elif bt_mode.startswith("C"):
            st.markdown("### Forward Paper Signal Tracker")
            st.markdown(
                "Save today's signal -> auto-checked against next trading day's close. "
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
                    st.success("Saved: {} {} @ Rs.{:,.0f}".format(
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
                        st.info("{} signals tracked - awaiting resolution".format(total))
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

        else:
            st.markdown("### Model Weight Calibration")
            st.markdown(
                "Grid-searches over RSI thresholds, volume confirmation, and minimum "
                "indicator agreement (54 combinations) to find params that maximise "
                "signal accuracy. Works best with **Angel One** live candle data."
            )

            if st.button("Run Grid Search (54 combinations)", type="primary", key="run_calib"):
                with st.spinner("Calibrating - scanning parameter combinations..."):
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


# =============================================================================
# TAB 5 - PAPER TRADE
# =============================================================================
with tab_pt:
    pt.init_portfolio(st.session_state)
    summary = pt.get_summary(st.session_state, df, underlying)

    st.markdown("## Paper Trading - Virtual Rs.1,00,000 Capital")
    st.caption("All trades are simulated. No real money involved. "
               "Trades persist within a session and are saved locally.")

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Available Capital", "Rs.{:,.0f}".format(summary["capital"]))
    pc2.metric("Realised P&L",
               "Rs.{:,.0f}".format(summary["realized_pnl"]),
               delta="+" + str(summary["realized_pnl"]) if summary["realized_pnl"] >= 0
                     else str(summary["realized_pnl"]))
    pc3.metric("Unrealised P&L",    "Rs.{:,.0f}".format(summary["unrealized_pnl"]))
    pc4.metric("Total Return",
               "{}%".format(summary["total_return_pct"]),
               delta="{:+,.0f}".format(summary["total_pnl"]))

    st.divider()

    st.markdown("### Place New Paper Trade")

    with st.form("place_trade_form"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            pt_sym  = st.selectbox("Symbol", list(INDICES.keys()), index=0)
            pt_type = st.radio("Type", ["BUY CALL", "BUY PUT"])
        with fc2:
            def_strike = float(int(meta.get("atm", underlying) / 50) * 50)
            pt_strike  = st.number_input("Strike Price Rs.", value=def_strike,
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
            pt_entry = st.number_input("Entry Premium Rs./unit", value=round(sugg, 2),
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
            "Entry Rs.":  t["entry_price"],
            "Cost Rs.":   round(t["entry_price"] * t["lots"] * t["lot_size"], 2),
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
                close_exit = st.number_input("Exit Premium Rs./unit", value=1.0,
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

    closed_trades = summary["closed_trades"]
    st.markdown("### Closed Trades ({})".format(len(closed_trades)))

    if closed_trades:
        closed_df = pd.DataFrame([{
            "ID":       t["id"],
            "Symbol":   t["symbol"],
            "Type":     t["type"],
            "Strike":   t["strike"],
            "Lots":     t["lots"],
            "Entry Rs.": t["entry_price"],
            "Exit Rs.":  t["exit_price"],
            "P&L Rs.":   t["pnl"],
            "Exited":   t["exit_time"],
        } for t in closed_trades])

        def _pnl_color(val):
            if val > 0:   return "color: #4CAF50"
            if val < 0:   return "color: #f44336"
            return "color: #888"

        st.dataframe(
            closed_df.style.map(_pnl_color, subset=["P&L Rs."]),
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

    st.divider()
    st.markdown("### Reset Portfolio")
    if "pt_confirm_reset" not in st.session_state:
        st.session_state.pt_confirm_reset = False

    if not st.session_state.pt_confirm_reset:
        if st.button("Reset Portfolio to Rs.1,00,000", type="secondary"):
            st.session_state.pt_confirm_reset = True
            st.rerun()
    else:
        st.warning("This will clear ALL trades and reset capital. This cannot be undone.")
        yes_col, no_col = st.columns(2)
        if yes_col.button("Yes, Reset Everything", type="primary"):
            pt.reset_portfolio(st.session_state)
            st.session_state.pt_confirm_reset = False
            st.success("Portfolio reset to Rs.1,00,000!")
            st.rerun()
        if no_col.button("Cancel"):
            st.session_state.pt_confirm_reset = False
            st.rerun()


# =============================================================================
# TAB 6 - CHAT
# =============================================================================
with tab_chat:
    st.markdown("## Chat with the Data")
    st.markdown(
        "Discuss the trade before you place it - ask for a **Trade Decision** or run "
        "the **Scalping Checklist**, then drill into any detail."
    )

    try:
        _api_key_present = bool(st.secrets.get("anthropic_api_key", ""))
    except Exception:
        _api_key_present = False

    if _api_key_present:
        st.success("Claude AI enabled - detailed context-aware answers active")
    else:
        st.info("Keyword mode active - add `anthropic_api_key` to Streamlit secrets for AI answers")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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
        index=3,
        horizontal=True,
        key="chat_tf",
        label_visibility="collapsed",
    )

    tf_info = {
        "5-min":  "10-15 min hold - SL 20% - Target 30% - Best: 9:20-10:00 AM",
        "15-min": "45-60 min hold - SL 30% - Target 50% - Best: 9:30-11:30 AM",
        "30-min": "1.5-2 hr hold  - SL 35% - Target 60% - Best: 9:45 AM-12:00 PM",
        "1-hr":   "2-3 hr hold    - SL 40% - Target 65% - Best: 9:30-11:30 AM",
        "2-hr":   "1-day hold     - SL 45% - Target 80% - Use credit spreads",
        "daily":  "2-3 day hold   - SL 50% - Target 100% - Use next-week expiry",
    }
    st.caption("Timer: " + tf_info[selected_tf])

    st.markdown("#### Start here:")
    pa1, pa2 = st.columns(2)
    if pa1.button(
        "Get Trade Decision",
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

    checklist_label = "{} Checklist".format(tf_options[selected_tf])
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

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(
        "Ask: trade decision - checklist - signal - pcr - IV - tomorrow - oi - help"
    )
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chat_bot.answer(user_input, df, meta, sig,
                                         timeframe=selected_tf)
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.markdown("---")
    st.markdown("**Quick questions:**")
    quick_qs = [
        ("Signal?",     "What is the current trade signal?"),
        ("PCR?",        "What is the put call ratio?"),
        ("Tomorrow?",   "What is tomorrow's prediction?"),
        ("Levels?",     "What are the support and resistance levels?"),
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

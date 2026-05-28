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
from stock_list import NSE_STOCKS, INDICES, search_stocks
import paper_trade as pt
import chat_bot
import backtest
import model_train

st.set_page_config(
    page_title="NSE Options Intelligence",
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

    st.caption("Updated: " + datetime.now().strftime("%H:%M:%S"))

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
def _get_angelone_data(sym, idx, strikes):
    from angelone_data import fetch_option_chain_angelone
    return fetch_option_chain_angelone(sym, idx, strikes)

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

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_bt, tab_pt, tab_chat = st.tabs(
    ["📊 Live Dashboard", "📈 Backtest", "💼 Paper Trade", "💬 Chat"]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
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
            icon = "🟢" if any(w in rl for w in ["bullish","support","upward","pe writing"]) else \
                   "🔴" if any(w in rl for w in ["bearish","resistance","downward","ce writing"]) else "🟡"
            st.markdown(icon + " " + r)
        st.markdown("---")
        st.markdown("#### Key Levels")
        st.dataframe(pd.DataFrame({
            "Level": ["Spot Price","Max Pain","CE Resistance (OI)","PE Support (OI)"],
            "Price": [underlying, max_pain, sig["max_ce_resistance"], sig["max_pe_support"]],
        }), hide_index=True, use_container_width=True)

    st.divider()

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
                "Simulates SMA-crossover + momentum signals on {} daily candles "
                "and checks if next-day direction was correct.".format(bt_days)
            )
            if st.button("Run Accuracy Backtest", type="primary", key="run_acc"):
                with st.spinner("Running backtest…"):
                    metrics, rdf = backtest.run_accuracy_backtest(candles)
                if not metrics:
                    st.error("Not enough data (need ≥12 candles).")
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Win Rate",      "{}%".format(metrics["win_rate"]))
                    m2.metric("Call Accuracy", "{}%".format(metrics["call_accuracy"]))
                    m3.metric("Put Accuracy",  "{}%".format(metrics["put_accuracy"]))
                    m4.metric("Total Traded",  str(metrics["total_traded"]))

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
                "Grid-search over scoring parameters to find weights that maximise "
                "signal accuracy on historical candles."
            )

            if st.button("Run Grid Search (576 combinations)", type="primary", key="run_calib"):
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
        a = chat_bot.answer(q, df, meta, sig)
        st.session_state.chat_history.append({"role": "assistant", "content": a})
        st.rerun()

    if pa2.button(
        "📋 Scalping Checklist (1-hr)",
        use_container_width=True,
        type="secondary",
        key="pa_scalp",
        help="Dynamic pass/fail checklist for intraday 1-hour scalp trades",
    ):
        q = "scalping checklist"
        st.session_state.chat_history.append({"role": "user", "content": q})
        a = chat_bot.answer(q, df, meta, sig)
        st.session_state.chat_history.append({"role": "assistant", "content": a})
        st.rerun()

    st.divider()

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(
        "Ask: trade decision · scalping checklist · signal · pcr · IV · tomorrow · help"
    )
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = chat_bot.answer(user_input, df, meta, sig)
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
            answer = chat_bot.answer(question, df, meta, sig)
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

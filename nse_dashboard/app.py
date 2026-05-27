import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math
import time
from datetime import datetime
from nse_data import fetch_option_chain, parse_option_chain, calculate_pcr, calculate_max_pain, generate_signal
from demo_data import generate_demo_option_chain
from stock_list import NSE_STOCKS, INDICES, search_stocks

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

    num_strikes      = st.slider("Strikes around ATM", 10, 40, 20, step=2)
    auto_refresh     = st.checkbox("Auto Refresh", value=False)
    refresh_interval = st.selectbox("Refresh every (sec)", [30, 60, 120], index=1)

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
        st.info("Try running the app during market hours (9:15 AM – 3:30 PM IST) on Indian internet.")
    elif not raw:
        st.error("NSE returned empty response.")
    else:
        keys = list(raw.keys())
        st.success("NSE connection OK! Top-level keys: " + str(keys))
        if "records" in raw:
            rec = raw["records"]
            st.write("underlyingValue:", rec.get("underlyingValue"))
            st.write("expiryDates:", rec.get("expiryDates", [])[:3])
            st.write("record count:", len(rec.get("data", [])))
            if rec.get("data"):
                st.write("First record sample:", rec["data"][0])
    st.stop()

# ── Data fetch ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _get_nse_data(sym, idx, strikes):
    raw      = fetch_option_chain(sym, idx)
    df, meta = parse_option_chain(raw, strikes)
    return df, meta

@st.cache_data(ttl=60)
def _get_angelone_data(sym, idx, strikes):
    from angelone_data import fetch_option_chain_angelone
    return fetch_option_chain_angelone(sym, idx, strikes)

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
            "1. Go to [nseindia.com/option-chain](https://www.nseindia.com/option-chain)\n"
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
    with st.spinner("Fetching {} from NSE (may need Indian IP + market hours)...".format(symbol)):
        df, meta = _get_nse_data(symbol, is_index, num_strikes)
    if "error" in meta:
        st.error("NSE fetch failed: " + meta["error"])
        st.warning(
            "NSE requires Indian internet + market hours. \n\n"
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

# ── Signal ────────────────────────────────────────────────────────────────────
sig        = generate_signal(df, meta)
underlying = meta["underlying"]
expiry     = meta["expiry"]
pcr        = sig["pcr"]
max_pain   = sig["max_pain"]
source_tag = meta.get("source", data_source)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## {} Option Chain  —  Expiry: **{}**".format(symbol, expiry))
st.markdown("**Spot: ₹{:,.2f}**  |  ATM: **{}**  |  Source: `{}`".format(
    underlying, meta["atm"], source_tag))
st.divider()

# ── Metrics ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spot Price",    "₹{:,.1f}".format(underlying))
c2.metric("PCR",           str(pcr), delta="Bullish" if pcr > 1.0 else "Bearish")
c3.metric("Max Pain",      "₹{:,.0f}".format(max_pain))
c4.metric("CE Resistance", "₹{:,.0f}".format(sig["max_ce_resistance"]), delta="Sell Wall")
c5.metric("PE Support",    "₹{:,.0f}".format(sig["max_pe_support"]),    delta="Buy Wall")
st.divider()

# ── Signal panel ──────────────────────────────────────────────────────────────
sig_col, reason_col = st.columns([1, 2])

with sig_col:
    bg     = {"green": "#1a7a1a", "red": "#8b1a1a", "orange": "#7a5c00"}.get(sig["color"], "#333")
    border = {"green": "#4CAF50", "red": "#f44336", "orange": "#FF9800"}.get(sig["color"], "#888")
    st.markdown(
        '<div style="background:{}; border:3px solid {}; padding:25px; border-radius:14px; text-align:center;">'
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

# ── Charts ────────────────────────────────────────────────────────────────────
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

# ── Tomorrow's Prediction ─────────────────────────────────────────────────────
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
        '<div style="color:#aaa;font-size:13px;">Expected Daily Range (ATM IV = {}%)</div>'
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
    '<span style="color:#ef5350;">Resistance: ₹{:,.0f} (max CE OI)  |  Upper range: ₹{:,.0f}</span><br>'
    '<span style="color:#26a69a;">Support: ₹{:,.0f} (max PE OI)  |  Lower range: ₹{:,.0f}</span><br>'
    '<span style="color:#FFD700;">Max Pain: ₹{:,.0f}  |  Spot: ₹{:,.0f}</span>'
    '</div>'.format(
        sig["max_ce_resistance"], upper_level,
        sig["max_pe_support"],    lower_level,
        max_pain, underlying
    ),
    unsafe_allow_html=True
)

st.divider()

# ── Option Chain Table ────────────────────────────────────────────────────────
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

if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()

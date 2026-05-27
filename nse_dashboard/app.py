import streamlit as st
import plotly.graph_objects as go
import pandas as pd
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

st.markdown("""
<style>
.stMetric > div { font-size: 18px; }
div[data-testid="stSelectbox"] { margin-bottom: 0px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("NSE Options Intelligence")
    st.caption("Powered by Yahoo Finance (free, live)")
    st.divider()

    instrument_type = st.radio("Instrument Type", ["Stock", "Index"])

    if instrument_type == "Stock":
        # Search box with autocomplete hint
        search_query = st.text_input(
            "Search Stock (type symbol or name)",
            value="SBIN",
            placeholder="e.g. SBIN, RELIANCE, TCS...",
        ).upper().strip()

        # Show matching suggestions
        suggestions = search_stocks(search_query)
        if suggestions:
            options     = ["{} — {}".format(s, n) for s, n in suggestions]
            selected    = st.selectbox("Select from matches", options, index=0)
            symbol      = selected.split(" — ")[0]
        else:
            st.warning("No match found. Using: " + search_query)
            symbol = search_query

        is_index = False
        st.caption("NSE symbol: **" + symbol + ".NS**")

    else:
        index_options = ["{} — {}".format(k, v) for k, v in INDICES.items()]
        selected_idx  = st.selectbox("Select Index", index_options)
        symbol        = selected_idx.split(" — ")[0]
        is_index      = True

    st.divider()
    num_strikes      = st.slider("Strikes around ATM", 10, 40, 20, step=2)
    auto_refresh     = st.checkbox("Auto Refresh", value=False)
    refresh_interval = st.selectbox("Refresh every (seconds)", [30, 60, 120], index=1)

    st.divider()
    demo_mode = st.toggle("Demo Mode", value=False,
                          help="Simulated data — no internet needed")
    st.divider()
    if st.button("Refresh Now", use_container_width=True, type="primary"):
        st.cache_data.clear()
    st.caption("Last updated: " + datetime.now().strftime("%H:%M:%S"))

# ── Data fetch ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_live_data(sym, idx, strikes):
    raw      = fetch_option_chain(sym, idx)
    df, meta = parse_option_chain(raw, strikes)
    return df, meta

if demo_mode:
    df, meta = generate_demo_option_chain("NIFTY")
    st.info("Demo Mode ON — showing simulated data.")
else:
    with st.spinner("Fetching {} option chain via Yahoo Finance...".format(symbol)):
        df, meta = get_live_data(symbol, is_index, num_strikes)
    if "error" in meta:
        st.error("Fetch failed: " + meta["error"])
        st.info("Try a different stock symbol, or enable **Demo Mode** in the sidebar.")
        st.stop()
    if df.empty:
        st.warning("No option chain data returned. Try a different stock or enable Demo Mode.")
        st.stop()

# ── Signal ────────────────────────────────────────────────────────────────────
sig        = generate_signal(df, meta)
underlying = meta["underlying"]
expiry     = meta["expiry"]
pcr        = sig["pcr"]
max_pain   = sig["max_pain"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## {} Option Chain  —  Expiry: **{}**".format(symbol, expiry))
st.markdown("**Spot Price: Rs.{:,.2f}**  |  ATM Strike: **{}**".format(underlying, meta["atm"]))
st.divider()

# ── Metrics ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spot Price",     "Rs.{:,.1f}".format(underlying))
c2.metric("PCR",            str(pcr),   delta="Bullish" if pcr > 1.0 else "Bearish")
c3.metric("Max Pain",       "Rs.{:,.0f}".format(max_pain))
c4.metric("CE Resistance",  "Rs.{:,.0f}".format(sig["max_ce_resistance"]), delta="Sell Wall")
c5.metric("PE Support",     "Rs.{:,.0f}".format(sig["max_pe_support"]),    delta="Buy Wall")
st.divider()

# ── Signal panel ──────────────────────────────────────────────────────────────
sig_col, reason_col = st.columns([1, 2])

with sig_col:
    bg     = {"green": "#1a7a1a", "red": "#8b1a1a", "orange": "#7a5c00"}.get(sig["color"], "#333")
    border = {"green": "#4CAF50", "red": "#f44336", "orange": "#FF9800"}.get(sig["color"], "#888")
    st.markdown(
        '<div style="background:{}; border:3px solid {}; padding:25px; border-radius:14px; text-align:center;">'
        '<div style="font-size:14px; color:#ccc; margin-bottom:6px;">TRADE SIGNAL</div>'
        '<div style="font-size:32px; font-weight:bold; color:{};">{}</div>'
        '<div style="font-size:16px; color:#ddd; margin-top:8px;">Confidence</div>'
        '<div style="font-size:42px; font-weight:bold; color:white;">{}%</div>'
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
        r_lower = r.lower()
        if any(w in r_lower for w in ["bullish", "support", "upward", "pe writing"]):
            icon = "🟢"
        elif any(w in r_lower for w in ["bearish", "resistance", "downward", "ce writing"]):
            icon = "🔴"
        else:
            icon = "🟡"
        st.markdown(icon + " " + r)
    st.markdown("---")
    st.markdown("#### Key Levels")
    st.dataframe(pd.DataFrame({
        "Level": ["Spot Price", "Max Pain", "CE Resistance (OI)", "PE Support (OI)"],
        "Price": [underlying, max_pain, sig["max_ce_resistance"], sig["max_pe_support"]],
    }), hide_index=True, use_container_width=True)

st.divider()

# ── OI Charts ─────────────────────────────────────────────────────────────────
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
    st.markdown("#### Implied Volatility Smile")
    iv_df = df[(df["ce_iv"] > 0) | (df["pe_iv"] > 0)]
    fi = go.Figure()
    if not iv_df.empty:
        fi.add_trace(go.Scatter(x=iv_df["strike"], y=iv_df["ce_iv"], mode="lines+markers",
                                name="CE IV", line=dict(color="#ef5350", width=2)))
        fi.add_trace(go.Scatter(x=iv_df["strike"], y=iv_df["pe_iv"], mode="lines+markers",
                                name="PE IV", line=dict(color="#26a69a", width=2)))
        fi.add_vline(x=underlying, line_dash="dash", line_color="white")
    fi.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     legend=dict(orientation="h", y=1.1),
                     xaxis=dict(title="Strike", gridcolor="#333"),
                     yaxis=dict(title="IV %", gridcolor="#333"),
                     margin=dict(t=20, b=10))
    st.plotly_chart(fi, use_container_width=True)

# ── PCR card ──────────────────────────────────────────────────────────────────
pc1, pc2 = st.columns(2)
with pc1:
    st.markdown("#### PCR Interpretation")
    if   pcr > 1.5: interp, pcr_col = "EXTREMELY BULLISH",          "#00e676"
    elif pcr > 1.2: interp, pcr_col = "BULLISH - strong put writing","#4CAF50"
    elif pcr > 0.8: interp, pcr_col = "NEUTRAL to BULLISH",          "#8BC34A"
    elif pcr > 0.5: interp, pcr_col = "NEUTRAL to BEARISH",          "#FF9800"
    else:           interp, pcr_col = "BEARISH - heavy call writing", "#f44336"
    st.markdown(
        '<div style="background:#1e1e2e; padding:20px; border-radius:12px;">'
        '<div style="font-size:48px; font-weight:bold; color:{}; text-align:center;">{}</div>'
        '<div style="font-size:16px; color:{}; text-align:center; margin-top:8px;">{}</div>'
        '<hr style="border-color:#333; margin:15px 0;">'
        '<div style="font-size:13px; color:#aaa;">'
        'PCR &lt; 0.5 = Bearish &nbsp;|&nbsp; 0.5-0.8 = Neutral-Bearish<br>'
        'PCR 0.8-1.2 = Neutral &nbsp;|&nbsp; 1.2-1.5 = Bullish<br>'
        'PCR &gt; 1.5 = Extremely Bullish'
        '</div></div>'.format(pcr_col, pcr, pcr_col, interp),
        unsafe_allow_html=True
    )

with pc2:
    # Expiry selector info
    st.markdown("#### Available Expiries")
    for i, exp in enumerate(meta.get("all_expiries", [expiry])[:6]):
        tag = " ← current" if i == 0 else ""
        st.markdown("- **{}**{}".format(exp, tag))

st.divider()

# ── Option Chain Table ────────────────────────────────────────────────────────
st.markdown("#### Full Option Chain Table")
tdf     = df.copy()
atm_idx = (tdf["strike"] - underlying).abs().idxmin()
tdf = tdf.rename(columns={
    "ce_oi": "CE OI", "ce_chg_oi": "CE Chg OI", "ce_volume": "CE Vol",
    "ce_iv": "CE IV%", "ce_ltp": "CE LTP", "strike": "STRIKE",
    "pe_ltp": "PE LTP", "pe_iv": "PE IV%", "pe_volume": "PE Vol",
    "pe_chg_oi": "PE Chg OI", "pe_oi": "PE OI",
})
cols = ["CE OI","CE Chg OI","CE Vol","CE IV%","CE LTP","STRIKE","PE LTP","PE IV%","PE Vol","PE Chg OI","PE OI"]
tdf  = tdf[cols]

def hl_atm(row):
    if row.name == atm_idx:
        return ["background-color:#2d2d00; font-weight:bold"] * len(row)
    return [""] * len(row)

fmt = {
    "CE OI":"{:,.0f}","CE Chg OI":"{:,.0f}","CE Vol":"{:,.0f}",
    "CE IV%":"{:.1f}","CE LTP":"{:.2f}",
    "PE LTP":"{:.2f}","PE IV%":"{:.1f}",
    "PE Vol":"{:,.0f}","PE Chg OI":"{:,.0f}","PE OI":"{:,.0f}",
}
st.dataframe(tdf.style.apply(hl_atm, axis=1).format(fmt), use_container_width=True, height=400)

st.markdown("---")
st.caption("Data: Yahoo Finance (yfinance)  |  For educational purposes only. Not financial advice.")

if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()

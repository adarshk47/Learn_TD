import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime
from nse_data import fetch_option_chain, parse_option_chain, calculate_pcr, calculate_max_pain, generate_signal
from demo_data import generate_demo_option_chain

st.set_page_config(
    page_title="NSE Options Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stMetric > div { font-size: 18px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("NSE Options Intelligence")
    st.caption("Real-time option chain analysis")
    st.divider()

    instrument_type = st.radio("Instrument Type", ["Index", "Stock"])
    if instrument_type == "Index":
        symbol = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"])
        is_index = True
    else:
        symbol = st.text_input("Stock Symbol (e.g. RELIANCE, TCS)", value="RELIANCE").upper()
        is_index = False

    num_strikes = st.slider("Strikes to show (around ATM)", 10, 40, 20, step=2)
    auto_refresh = st.checkbox("Auto Refresh", value=False)
    refresh_interval = st.selectbox("Refresh every (seconds)", [15, 30, 60, 120], index=1)

    st.divider()
    demo_mode = st.toggle("Demo Mode (simulated data)", value=False,
                          help="Use when NSE is closed or you're testing")
    st.divider()
    if st.button("Refresh Now", use_container_width=True, type="primary"):
        st.cache_data.clear()
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


@st.cache_data(ttl=30)
def get_live_data(sym, idx, strikes):
    raw = fetch_option_chain(sym, idx)
    df, meta = parse_option_chain(raw, strikes)
    return df, meta


if demo_mode:
    df, meta = generate_demo_option_chain(symbol if is_index else "NIFTY")
    st.info("Demo Mode ON — showing simulated data. Toggle off during market hours for live NSE data.")
else:
    with st.spinner(f"Fetching {symbol} option chain from NSE..."):
        df, meta = get_live_data(symbol, is_index, num_strikes)
    if "error" in meta:
        st.error(f"NSE fetch failed: {meta['error']}")
        st.warning("Enable **Demo Mode** in the sidebar to preview the dashboard.")
        st.stop()
    if df.empty:
        st.warning("No data returned. NSE may be closed. Enable Demo Mode to preview.")
        st.stop()

sig = generate_signal(df, meta)
underlying = meta["underlying"]
expiry = meta["expiry"]
pcr = sig["pcr"]
max_pain = sig["max_pain"]

st.markdown(f"## {symbol} Option Chain  —  Expiry: **{expiry}**")
st.markdown(f"**Spot Price: ₹{underlying:,.2f}**  |  ATM Strike: **{meta['atm']}**")
st.divider()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Spot Price", f"₹{underlying:,.1f}")
with col2:
    st.metric("Put-Call Ratio", f"{pcr}", delta="Bullish" if pcr > 1.0 else "Bearish")
with col3:
    st.metric("Max Pain", f"₹{max_pain:,.0f}")
with col4:
    st.metric("CE Resistance", f"₹{sig['max_ce_resistance']:,.0f}", delta="Sell Wall")
with col5:
    st.metric("PE Support", f"₹{sig['max_pe_support']:,.0f}", delta="Buy Wall")

st.divider()

sig_col, reason_col = st.columns([1, 2])
with sig_col:
    bg_color = {"green": "#1a7a1a", "red": "#8b1a1a", "orange": "#7a5c00"}.get(sig["color"], "#333")
    border_color = {"green": "#4CAF50", "red": "#f44336", "orange": "#FF9800"}.get(sig["color"], "#888")
    st.markdown(f"""
    <div style="background:{bg_color}; border: 3px solid {border_color};
                padding:25px; border-radius:14px; text-align:center;">
        <div style="font-size:14px; color:#ccc; margin-bottom:6px;">TRADE SIGNAL</div>
        <div style="font-size:32px; font-weight:bold; color:{border_color};">{sig['signal']}</div>
        <div style="font-size:16px; color:#ddd; margin-top:8px;">Confidence</div>
        <div style="font-size:42px; font-weight:bold; color:white;">{sig['confidence']}%</div>
    </div>
    """, unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=sig["confidence"],
        title={"text": "Signal Strength", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": border_color},
            "steps": [{"range": [0, 40], "color": "#3d0000"},
                      {"range": [40, 65], "color": "#3d3d00"},
                      {"range": [65, 100], "color": "#003d00"}],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": sig["confidence"]},
        },
        number={"suffix": "%"},
    ))
    fig_gauge.update_layout(height=200, margin=dict(t=30, b=0, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)

with reason_col:
    st.markdown("#### Analysis Breakdown")
    for r in sig["reasons"]:
        icon = "🟢" if any(w in r.lower() for w in ["bullish", "support", "upward", "pe writing"]) else \
               "🔴" if any(w in r.lower() for w in ["bearish", "resistance", "downward", "ce writing"]) else "🟡"
        st.markdown(f"{icon} {r}")
    st.markdown("---")
    st.markdown("#### Key Levels")
    levels_df = pd.DataFrame({
        "Level": ["Spot Price", "Max Pain", "CE Resistance (OI)", "PE Support (OI)"],
        "Price": [underlying, max_pain, sig['max_ce_resistance'], sig['max_pe_support']],
    })
    st.dataframe(levels_df, hide_index=True, use_container_width=True)

st.divider()

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("#### Open Interest Distribution")
    fig_oi = go.Figure()
    fig_oi.add_trace(go.Bar(x=df["strike"], y=df["ce_oi"] / 1000, name="Call OI", marker_color="#ef5350", opacity=0.85))
    fig_oi.add_trace(go.Bar(x=df["strike"], y=df["pe_oi"] / 1000, name="Put OI", marker_color="#26a69a", opacity=0.85))
    fig_oi.add_vline(x=underlying, line_dash="dash", line_color="white",
                     annotation_text=f"Spot {underlying:.0f}", annotation_font_color="white")
    fig_oi.add_vline(x=max_pain, line_dash="dot", line_color="yellow",
                     annotation_text=f"MaxPain {max_pain:.0f}", annotation_font_color="yellow")
    fig_oi.update_layout(barmode="group", height=320,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         legend=dict(orientation="h", y=1.1),
                         xaxis=dict(title="Strike Price", gridcolor="#333"),
                         yaxis=dict(title="OI (thousands)", gridcolor="#333"),
                         margin=dict(t=20, b=10))
    st.plotly_chart(fig_oi, use_container_width=True)

with chart_col2:
    st.markdown("#### Change in OI (Today)")
    fig_chg = go.Figure()
    fig_chg.add_trace(go.Bar(x=df["strike"], y=df["ce_chg_oi"] / 1000, name="CE Change OI", marker_color="#ef5350", opacity=0.85))
    fig_chg.add_trace(go.Bar(x=df["strike"], y=df["pe_chg_oi"] / 1000, name="PE Change OI", marker_color="#26a69a", opacity=0.85))
    fig_chg.add_vline(x=underlying, line_dash="dash", line_color="white",
                      annotation_text=f"Spot {underlying:.0f}", annotation_font_color="white")
    fig_chg.update_layout(barmode="group", height=320,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", y=1.1),
                          xaxis=dict(title="Strike Price", gridcolor="#333"),
                          yaxis=dict(title="Change OI (thousands)", gridcolor="#333"),
                          margin=dict(t=20, b=10))
    st.plotly_chart(fig_chg, use_container_width=True)

iv_col1, iv_col2 = st.columns(2)
with iv_col1:
    st.markdown("#### Implied Volatility Smile")
    iv_df = df[(df["ce_iv"] > 0) | (df["pe_iv"] > 0)]
    if not iv_df.empty:
        fig_iv = go.Figure()
        fig_iv.add_trace(go.Scatter(x=iv_df["strike"], y=iv_df["ce_iv"], mode="lines+markers",
                                    name="CE IV", line=dict(color="#ef5350", width=2)))
        fig_iv.add_trace(go.Scatter(x=iv_df["strike"], y=iv_df["pe_iv"], mode="lines+markers",
                                    name="PE IV", line=dict(color="#26a69a", width=2)))
        fig_iv.add_vline(x=underlying, line_dash="dash", line_color="white")
        fig_iv.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             legend=dict(orientation="h", y=1.1),
                             xaxis=dict(title="Strike", gridcolor="#333"),
                             yaxis=dict(title="IV %", gridcolor="#333"),
                             margin=dict(t=20, b=10))
        st.plotly_chart(fig_iv, use_container_width=True)

with iv_col2:
    st.markdown("#### PCR Interpretation")
    if pcr > 1.5:
        interpretation, pcr_color = "EXTREMELY BULLISH", "#00e676"
    elif pcr > 1.2:
        interpretation, pcr_color = "BULLISH - Strong put writing", "#4CAF50"
    elif pcr > 0.8:
        interpretation, pcr_color = "NEUTRAL to BULLISH", "#8BC34A"
    elif pcr > 0.5:
        interpretation, pcr_color = "NEUTRAL to BEARISH", "#FF9800"
    else:
        interpretation, pcr_color = "BEARISH - Heavy call writing", "#f44336"
    st.markdown(f"""
    <div style="background:#1e1e2e; padding:20px; border-radius:12px; margin-top:10px;">
        <div style="font-size:48px; font-weight:bold; color:{pcr_color}; text-align:center;">{pcr}</div>
        <div style="font-size:16px; color:{pcr_color}; text-align:center; margin-top:8px;">{interpretation}</div>
        <hr style="border-color:#333; margin:15px 0;">
        <div style="font-size:13px; color:#aaa;">
            📌 PCR &lt; 0.5 → Bearish<br>
            📌 PCR 0.5–0.8 → Neutral-Bearish<br>
            📌 PCR 0.8–1.2 → Neutral<br>
            📌 PCR 1.2–1.5 → Bullish<br>
            📌 PCR &gt; 1.5 → Extremely Bullish
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("#### Full Option Chain Table")
display_df = df.copy()
atm_idx = (display_df["strike"] - underlying).abs().idxmin()
display_df = display_df.rename(columns={
    "ce_oi": "CE OI", "ce_chg_oi": "CE Chg OI", "ce_volume": "CE Vol",
    "ce_iv": "CE IV%", "ce_ltp": "CE LTP", "strike": "STRIKE",
    "pe_ltp": "PE LTP", "pe_iv": "PE IV%", "pe_volume": "PE Vol",
    "pe_chg_oi": "PE Chg OI", "pe_oi": "PE OI",
})
display_cols = ["CE OI", "CE Chg OI", "CE Vol", "CE IV%", "CE LTP",
                "STRIKE",
                "PE LTP", "PE IV%", "PE Vol", "PE Chg OI", "PE OI"]
display_df = display_df[display_cols]

def highlight_atm(row):
    if row.name == atm_idx:
        return ["background-color: #2d2d00; font-weight: bold"] * len(row)
    return [""] * len(row)

styled = display_df.style.apply(highlight_atm, axis=1).format({
    "CE OI": "{:,.0f}", "CE Chg OI": "{:,.0f}", "CE Vol": "{:,.0f}",
    "CE IV%": "{:.1f}", "CE LTP": "{:.2f}",
    "PE LTP": "{:.2f}", "PE IV%": "{:.1f}",
    "PE Vol": "{:,.0f}", "PE Chg OI": "{:,.0f}", "PE OI": "{:,.0f}",
})
st.dataframe(styled, use_container_width=True, height=400)

st.markdown("---")
st.caption("Data source: NSE India  |  For educational purposes only. Not financial advice.")

if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()
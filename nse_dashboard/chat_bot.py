"""
NSE Options Intelligence Chat Bot.

Answers natural-language questions about the current option chain data.

Two modes:
  1. Keyword-based  — always works, no external API required.
  2. Claude API     — used when `anthropic_api_key` is present in st.secrets
                      (falls back to keyword mode on any error).
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _fmt(val: float, decimals: int = 2) -> str:
    if decimals == 0:
        return "{:,.0f}".format(val)
    return "{:,.{}f}".format(val, decimals)


def _top_oi_strikes(df: pd.DataFrame, col: str, n: int = 3) -> list:
    if df.empty or col not in df.columns:
        return []
    top = df.nlargest(n, col)[["strike", col]]
    return [(row["strike"], row[col]) for _, row in top.iterrows()]


def _atm_iv(df: pd.DataFrame, underlying: float) -> tuple:
    if df.empty:
        return 0.0, 0.0
    try:
        idx = (df["strike"] - underlying).abs().idxmin()
        row = df.iloc[idx]
        return _safe_float(row.get("ce_iv", 0)), _safe_float(row.get("pe_iv", 0))
    except Exception:
        return 0.0, 0.0


def _atm_ltp(df: pd.DataFrame, underlying: float) -> tuple:
    """Return (ce_ltp, pe_ltp) for the ATM strike."""
    if df.empty:
        return 0.0, 0.0
    try:
        idx = (df["strike"] - underlying).abs().idxmin()
        row = df.iloc[idx]
        return _safe_float(row.get("ce_ltp", 0)), _safe_float(row.get("pe_ltp", 0))
    except Exception:
        return 0.0, 0.0


# ── Trade Decision ────────────────────────────────────────────────────────────

def _trade_decision(df: pd.DataFrame, meta: dict, sig: dict) -> str:
    """
    Comprehensive trade decision synthesising all available option chain data.
    Gives a clear BUY CALL / BUY PUT / AVOID recommendation with entry, SL, target.
    """
    underlying = _safe_float(meta.get("underlying", 0))
    expiry     = meta.get("expiry", "N/A")
    atm        = _safe_float(meta.get("atm", underlying))
    pcr        = _safe_float(sig.get("pcr", 0))
    max_pain   = _safe_float(sig.get("max_pain", 0))
    signal     = sig.get("signal", "N/A")
    confidence = _safe_float(sig.get("confidence", 0))
    score      = _safe_float(sig.get("score", 0))
    reasons    = sig.get("reasons", [])
    ce_res     = _safe_float(sig.get("max_ce_resistance", 0))
    pe_sup     = _safe_float(sig.get("max_pe_support", 0))
    symbol     = meta.get("symbol", "INDEX")

    ce_iv, pe_iv = _atm_iv(df, underlying)
    avg_iv       = (ce_iv + pe_iv) / 2 if (ce_iv + pe_iv) > 0 else 15.0
    ce_ltp, pe_ltp = _atm_ltp(df, underlying)

    # ── Signal header ─────────────────────────────────────────────────────────
    if signal == "BUY CALL":
        sig_icon   = "🟢"
        action     = "BUY CALL"
        ltp        = ce_ltp if ce_ltp > 0 else underlying * 0.008
        hedge      = "Consider buying 1-OTM PE as hedge if holding > 30 min"
    elif signal == "BUY PUT":
        sig_icon   = "🔴"
        action     = "BUY PUT"
        ltp        = pe_ltp if pe_ltp > 0 else underlying * 0.008
        hedge      = "Consider buying 1-OTM CE as hedge if holding > 30 min"
    else:
        sig_icon   = "🟡"
        action     = "AVOID / WAIT"
        ltp        = max(ce_ltp, pe_ltp, underlying * 0.006)
        hedge      = "No directional trade recommended right now"

    sl_pct  = 0.40
    tgt_pct = 0.55
    sl_price  = round(ltp * (1 - sl_pct), 1) if ltp > 0 else 0
    tgt1      = round(ltp * (1 + tgt_pct), 1) if ltp > 0 else 0
    tgt2      = round(ltp * (1 + tgt_pct + 0.20), 1) if ltp > 0 else 0

    # ── Individual checks ──────────────────────────────────────────────────────
    chk_pcr     = "✅" if (signal == "BUY CALL" and pcr > 1.1) or (signal == "BUY PUT" and pcr < 0.9) else \
                  "⚠️" if 0.9 <= pcr <= 1.1 else "❌"
    pain_diff   = max_pain - underlying
    chk_pain    = "✅" if (signal == "BUY CALL" and pain_diff > 0) or \
                         (signal == "BUY PUT"  and pain_diff < 0) else \
                  "⚠️" if abs(pain_diff) / max(underlying, 1) < 0.003 else "❌"
    chk_iv      = "✅" if avg_iv < 14 else "⚠️" if avg_iv < 22 else "❌"

    room_to_res = (ce_res - underlying) / max(underlying, 1) * 100
    room_to_sup = (underlying - pe_sup) / max(underlying, 1) * 100
    chk_res     = "✅" if room_to_res > 0.5 else "⚠️" if room_to_res > 0.2 else "❌"
    chk_sup     = "✅" if room_to_sup > 0.3 else "⚠️" if room_to_sup > 0.1 else "❌"
    chk_conf    = "✅" if confidence >= 65 else "⚠️" if confidence >= 50 else "❌"

    # ── OI Intelligence checks (VarunS2002 approach) ──────────────────────────
    call_sum_v = _safe_float(sig.get("call_sum", 0))
    put_sum_v  = _safe_float(sig.get("put_sum", 0))
    itm_ratio  = _safe_float(sig.get("itm_ratio", 0))
    chk_oi_sum = "✅" if (signal == "BUY CALL" and put_sum_v > call_sum_v) or \
                        (signal == "BUY PUT"  and call_sum_v > put_sum_v) else \
                 "⚠️" if abs(put_sum_v - call_sum_v) < 0.5 else "❌"
    chk_itm    = "✅" if (signal == "BUY CALL" and itm_ratio > 1.5) or \
                        (signal == "BUY PUT"  and itm_ratio < 0.67 and itm_ratio > 0) else \
                 "⚠️" if 0.67 <= itm_ratio <= 1.5 else "❌"

    iv_note  = "Low IV — cheap premiums ✅" if avg_iv < 14 else \
               "Moderate IV — fair entry ⚠️" if avg_iv < 22 else \
               "High IV — expensive, prefer selling ❌"

    pain_dir = "above" if pain_diff > 0 else "below"
    pain_pull = "upward" if pain_diff > 0 else "downward"

    # ── Overall verdict ───────────────────────────────────────────────────────
    all_checks = [chk_pcr, chk_pain, chk_iv, chk_res, chk_sup, chk_conf, chk_oi_sum, chk_itm]
    good = sum(1 for c in all_checks if c == "✅")
    warn = sum(1 for c in all_checks if c == "⚠️")
    if good >= 6:
        verdict = "🟢 **GO — conditions favour this trade**"
    elif good >= 4 and warn <= 2:
        verdict = "🟡 **CAUTION — mixed signals, size down**"
    else:
        verdict = "🔴 **WAIT — too many checks failing**"

    strategy_type = sig.get("strategy_type", "ATM Option Buy")
    strategy_note = sig.get("strategy_note", "")

    lines = [
        "## 🎯 Trade Decision — {}".format(symbol),
        "",
        "{} **Signal: {} ({}% confidence)**".format(sig_icon, action, int(confidence)),
        "",
        "---",
        "### Signal Checks",
        "{} **PCR: {}** → {}".format(
            chk_pcr, _fmt(pcr),
            "Bullish (>1.1 put writing)" if pcr > 1.1 else
            "Bearish (<0.9 call writing)" if pcr < 0.9 else "Neutral (0.9–1.1)"
        ),
        "{} **Max Pain: ₹{}** — ₹{:,.0f} {} spot ({} pull at expiry)".format(
            chk_pain, _fmt(max_pain, 0), abs(pain_diff), pain_dir, pain_pull
        ),
        "{} **Confidence: {}%** ({})".format(
            chk_conf, int(confidence),
            "Strong" if confidence >= 65 else "Moderate" if confidence >= 50 else "Weak"
        ),
        "{} **IV: {:.1f}%** — {}".format(chk_iv, avg_iv, iv_note),
        "{} **Resistance room: {:.1f}%** to ₹{} CE wall".format(chk_res, room_to_res, int(ce_res)),
        "{} **Support room: {:.1f}%** above ₹{} PE floor".format(chk_sup, room_to_sup, int(pe_sup)),
        "{} **OI 3-strike sum** — Put Sum {:+.1f}K vs Call Sum {:+.1f}K".format(
            chk_oi_sum, put_sum_v, call_sum_v),
        "{} **ITM Ratio: {:.2f}x** — {}".format(
            chk_itm, itm_ratio,
            "Strong put writing (bullish)" if itm_ratio > 1.5 else
            "Call writing dominates (bearish)" if 0 < itm_ratio < 0.67 else "Neutral"),
        "",
        "**Overall: {} — {}/{} checks pass**".format(verdict, good, 8),
        "",
        "---",
        "### Trade Setup" + (" (ATM {})".format("CE" if signal == "BUY CALL" else "PE") if signal != "AVOID / WAIT" else ""),
        "| | |",
        "|---|---|",
        "| Spot | ₹{} |".format(_fmt(underlying)),
        "| ATM Strike | ₹{} |".format(_fmt(atm, 0)),
        "| Est. Premium | ₹{:.1f} |".format(ltp) if ltp > 0 else "| Est. Premium | — |",
        "| Stop-loss | ₹{:.1f} (−40%) |".format(sl_price) if sl_price > 0 else "| Stop-loss | — |",
        "| Target 1 | ₹{:.1f} (+55%) |".format(tgt1) if tgt1 > 0 else "| Target 1 | — |",
        "| Target 2 | ₹{:.1f} (+75%) |".format(tgt2) if tgt2 > 0 else "| Target 2 | — |",
        "| Expiry | {} |".format(expiry),
        "",
        "💡 {}".format(hedge),
        "",
        "---",
        "### Recommended Strategy",
        "**{}** — {}".format(strategy_type, strategy_note),
        "",
        "| Strategy | Hist. Win Rate | When to use |",
        "|---|---|---|",
        "| ATM Option Buy | 74.8% | Confidence ≥ 70%, clear directional signal |",
        "| Credit Spread | 79.2% | Confidence 60–70%, moderate conviction |",
        "| Iron Condor | 83.4% | Neutral/range-bound, low conviction |",
        "| Long Straddle | 58.0% | High IV spike, big move expected |",
        "",
        "---",
        "### Reasons Behind Signal",
    ]
    for r in reasons:
        lines.append("• " + r)

    lines += [
        "",
        "---",
        "⏰ **Best intraday windows:** 9:20–10:30 AM · 1:30–3:00 PM IST",
        "📏 **Position sizing:** 1 lot per ₹50,000 deployed capital",
        "⚠️ Educational only — not financial advice. Always use a stop-loss.",
    ]
    return "\n".join(lines)


# ── Timeframe configuration ───────────────────────────────────────────────────

# Research-backed parameters per timeframe.
# Sources: NSE option chain methodology (VarunS2002), Zerodha Varsity intraday
# guide, and standard RSI/EMA practitioner thresholds for Indian indices.
_TF_CONFIG = {
    "5-min": {
        "label":       "5-min Scalp",
        "sl_pct":      0.20,   # tight stop — 20% of premium
        "tgt_pct":     0.30,   # quick target — 30% of premium
        "rr":          "1 : 1.5",
        "max_hold":    "10–15 min",
        "best_time":   "9:20–10:00 AM  |  2:15–3:00 PM",
        "rsi_bull":    65,     # stronger momentum needed for very short-term
        "rsi_bear":    35,
        "vol_mult":    "2×+ average",
        "chart_ema":   "9-EMA on 5-min",
        "chart_extra": "VWAP — trade only in VWAP direction",
        "manual_checks": [
            "5-min candle closed strongly above/below VWAP",
            "5-min RSI > 65 (calls) or < 35 (puts) — extreme reading required",
            "Volume on signal candle > 2× the 5-bar average",
            "No major level (round number, PDH/PDL) within 30 points",
            "Time window: 9:20–10:00 AM or 2:15–3:00 PM only",
            "Bid-ask spread at ATM < ₹3 (5-min scalps need very tight spreads)",
        ],
        "tips": [
            "Only trade 5-min scalps during the first 45 min and last 45 min of the session",
            "3-candle confirmation rule: wait for 3 consecutive candles above 9-EMA before entry",
            "Exit at any sign of reversal — do not wait for target if price stalls",
            "One bad trade can wipe 3 winning scalps — strict 20% SL is non-negotiable",
        ],
    },
    "15-min": {
        "label":       "15-min Intraday",
        "sl_pct":      0.30,
        "tgt_pct":     0.50,
        "rr":          "1 : 1.7",
        "max_hold":    "45–60 min",
        "best_time":   "9:30–11:30 AM  |  1:30–3:00 PM",
        "rsi_bull":    60,
        "rsi_bear":    40,
        "vol_mult":    "1.5× average",
        "chart_ema":   "20-EMA on 15-min",
        "chart_extra": "SuperTrend (10, 3) direction on 15-min",
        "manual_checks": [
            "15-min close above 20-EMA (calls) or below 20-EMA (puts)",
            "RSI(14) on 15-min: > 60 bullish entry  |  < 40 bearish entry",
            "SuperTrend(10,3) green (calls) or red (puts) on 15-min chart",
            "Volume on signal candle > 1.5× the 5-bar average",
            "No major event (RBI, earnings, FII data) in next 1 hour",
            "1-hr chart in same direction (higher-timeframe alignment)",
        ],
        "tips": [
            "Best entry: 15-min candle that re-tests 20-EMA after a breakout",
            "Avoid trading between 12:00–1:30 PM (low volume, choppy)",
            "If 1-hr trend opposes 15-min signal, skip — trade only with the big trend",
            "CE/PE LTP should move ≥ ₹5 within 2 candles after entry, else exit early",
        ],
    },
    "30-min": {
        "label":       "30-min Swing",
        "sl_pct":      0.35,
        "tgt_pct":     0.60,
        "rr":          "1 : 1.7",
        "max_hold":    "1.5–2 hours",
        "best_time":   "9:45 AM–12:00 PM  |  1:30–3:15 PM",
        "rsi_bull":    58,
        "rsi_bear":    42,
        "vol_mult":    "1.3× average",
        "chart_ema":   "20-EMA on 30-min",
        "chart_extra": "Bollinger Band mid-line (20-period)",
        "manual_checks": [
            "30-min close above 20-EMA (calls) or below (puts)",
            "RSI(14) on 30-min: 55–70 range for calls  |  30–45 for puts",
            "OI chain: PCR direction aligned with signal",
            "Max Pain within 0.5% of spot (neutral) OR clearly on your side",
            "1-hr trend aligned (higher-timeframe confirmation)",
            "Volume above 30-min average by at least 1.3×",
        ],
        "tips": [
            "30-min trades balance between scalp speed and sufficient reward",
            "Use option chain OI to confirm: put writers at support = calls OK",
            "Watch for divergence between 15-min RSI and price (early reversal signal)",
            "If holding into lunch (12–1:30 PM), consider partial exit — volume dries up",
        ],
    },
    "1-hr": {
        "label":       "1-hr Intraday",
        "sl_pct":      0.40,
        "tgt_pct":     0.65,
        "rr":          "1 : 1.6",
        "max_hold":    "2–3 hours",
        "best_time":   "9:30–11:30 AM  |  1:30–3:00 PM",
        "rsi_bull":    55,
        "rsi_bear":    45,
        "vol_mult":    "1.2× average",
        "chart_ema":   "20-EMA on 1-hr  +  50-EMA trend",
        "chart_extra": "MACD(12,26,9) signal crossover on 1-hr",
        "manual_checks": [
            "1-hr close above 20-EMA (calls) or below 20-EMA (puts)",
            "1-hr RSI(14) > 55 for calls  |  < 45 for puts",
            "MACD histogram positive (calls) or negative (puts) on 1-hr",
            "Option chain: PCR > 1.1 for calls  |  < 0.9 for puts",
            "Max Pain above spot (calls) or below spot (puts)",
            "ATM OI change: PE writing at support (calls) | CE writing at resistance (puts)",
            "No major economic data release in next 2 hours",
        ],
        "tips": [
            "1-hr is the most reliable intraday timeframe — skip 5-min noise",
            "Always confirm with the option chain: if OI doesn't agree, wait",
            "Entry at 9:30 AM after the 9:15–9:25 gap fills is often the best",
            "Book 50% at Target 1 (+40%), trail remainder with 20-EMA as guide",
        ],
    },
    "2-hr": {
        "label":       "2-hr Positional",
        "sl_pct":      0.45,
        "tgt_pct":     0.80,
        "rr":          "1 : 1.8",
        "max_hold":    "1 trading day",
        "best_time":   "Morning session (before 12:00 PM)",
        "rsi_bull":    55,
        "rsi_bear":    45,
        "vol_mult":    "1.1× average",
        "chart_ema":   "50-EMA on 1-hr  +  20-EMA on daily",
        "chart_extra": "Daily Supertrend direction",
        "manual_checks": [
            "Daily candle above 20-EMA — medium-term trend is up (calls)",
            "1-hr RSI(14) > 55 and not overbought (< 70) for call entry",
            "SuperTrend on daily chart: green for calls, red for puts",
            "Option chain PCR strongly bullish (> 1.2) or bearish (< 0.8)",
            "Max Pain pull clearly directional (> 0.5% away from spot)",
            "FII/DII data (NSE website) — net buyers align with your direction",
            "VIX: below 16 = calm market, better for directional trades",
        ],
        "tips": [
            "2-hr positional trades carry overnight/session risk — use defined risk spreads",
            "Consider Bull Put Spread (calls) or Bear Call Spread (puts) instead of naked options",
            "Check SGX Nifty or Dow futures for macro alignment",
            "Exit before the last 30 min if target not hit — theta decay accelerates",
        ],
    },
    "daily": {
        "label":       "Daily Swing",
        "sl_pct":      0.50,
        "tgt_pct":     1.00,
        "rr":          "1 : 2.0",
        "max_hold":    "2–3 trading days",
        "best_time":   "Any — use daily close data",
        "rsi_bull":    55,
        "rsi_bear":    45,
        "vol_mult":    "Above 5-day average",
        "chart_ema":   "20-EMA on daily  +  50-EMA direction",
        "chart_extra": "Weekly chart trend alignment",
        "manual_checks": [
            "Daily close above 20-EMA AND 50-EMA (calls) or below both (puts)",
            "Daily RSI(14) > 55 and trending up (calls) | < 45 trending down (puts)",
            "Weekly chart in same direction — don't fight the weekly trend",
            "FII net buying (calls) or selling (puts) consistently this week",
            "VIX < 20 for buying calls/puts; very high VIX = prefer selling strategies",
            "Check Max Pain for next expiry — build into the expiry pull",
            "Option chain: large OI buildup at nearby support / resistance",
            "No major event (Union Budget, RBI, US Fed) in next 2–3 days",
        ],
        "tips": [
            "For daily swing, use next-week expiry or monthly expiry options (more time)",
            "Prefer Bull Put Spread / Bear Call Spread for defined risk swing trades",
            "Historical win rate: Credit Spreads 79%, Naked Options 58% on swing timeframe",
            "Close before any high-impact news event — tail risk kills swing P&L",
            "Trail stop-loss using 20-EMA on daily chart once in profit by 30%+",
        ],
    },
}


# ── Scalping / Intraday Checklist ─────────────────────────────────────────────

def _scalping_checklist(df: pd.DataFrame, meta: dict, sig: dict,
                        timeframe: str = "1-hr") -> str:
    """
    Dynamic pass/fail checklist for intraday/positional option trades.
    Timeframe-aware: adjusts SL, target, manual checks per timeframe.
    """
    tf  = _TF_CONFIG.get(timeframe, _TF_CONFIG["1-hr"])
    underlying = _safe_float(meta.get("underlying", 0))
    atm        = _safe_float(meta.get("atm", underlying))
    pcr        = _safe_float(sig.get("pcr", 0))
    max_pain   = _safe_float(sig.get("max_pain", 0))
    signal     = sig.get("signal", "N/A")
    confidence = _safe_float(sig.get("confidence", 0))
    ce_res     = _safe_float(sig.get("max_ce_resistance", 0))
    pe_sup     = _safe_float(sig.get("max_pe_support", 0))
    symbol     = meta.get("symbol", "INDEX")

    ce_iv, pe_iv = _atm_iv(df, underlying)
    avg_iv       = (ce_iv + pe_iv) / 2 if (ce_iv + pe_iv) > 0 else 15.0
    ce_ltp, pe_ltp = _atm_ltp(df, underlying)

    # OI buildup near ATM
    if not df.empty:
        atm_idx  = (df["strike"] - underlying).abs().idxmin()
        near     = df.iloc[max(0, atm_idx - 3): atm_idx + 4]
        net_chg  = near["pe_chg_oi"].sum() - near["ce_chg_oi"].sum()
        atm_oi_bias = "PE writing (bullish)" if net_chg > 0 else \
                      "CE writing (bearish)" if net_chg < 0 else "Neutral"
        atm_oi_ok = "✅" if (signal == "BUY CALL" and net_chg > 0) or \
                            (signal == "BUY PUT"  and net_chg < 0) else "⚠️"
    else:
        atm_oi_bias = "N/A"
        atm_oi_ok   = "⚠️"

    # ATM liquidity
    atm_ce_oi = atm_pe_oi = 0.0
    if not df.empty:
        try:
            idx = (df["strike"] - underlying).abs().idxmin()
            atm_ce_oi = _safe_float(df.iloc[idx].get("ce_oi", 0))
            atm_pe_oi = _safe_float(df.iloc[idx].get("pe_oi", 0))
        except Exception:
            pass
    liq_oi  = atm_ce_oi if signal == "BUY CALL" else atm_pe_oi
    chk_liq = "✅" if liq_oi > 50000 else "⚠️" if liq_oi > 10000 else "❌"
    liq_note = "{:,.0f} lots at ATM".format(liq_oi)

    room_to_res = (ce_res - underlying) / max(underlying, 1) * 100
    room_to_sup = (underlying - pe_sup) / max(underlying, 1) * 100
    pain_diff   = max_pain - underlying

    # Confidence threshold is tighter for shorter timeframes
    conf_thresh = 70 if timeframe in ("5-min", "15-min") else 65
    chk_sig  = "✅" if confidence >= conf_thresh else "⚠️" if confidence >= 50 else "❌"

    # PCR check
    chk_pcr  = "✅" if (signal == "BUY CALL" and pcr > 1.1) or \
                       (signal == "BUY PUT"  and pcr < 0.9) else \
               "⚠️" if 0.9 <= pcr <= 1.1 else "❌"
    chk_pain = "✅" if (signal == "BUY CALL" and pain_diff > 0) or \
                       (signal == "BUY PUT"  and pain_diff < 0) else \
               "⚠️" if abs(pain_diff) / max(underlying, 1) < 0.003 else "❌"
    chk_iv   = "✅" if avg_iv < 14 else "⚠️" if avg_iv < 22 else "❌"
    chk_res  = "✅" if room_to_res > 0.5 else "⚠️" if room_to_res > 0.2 else "❌"
    chk_sup  = "✅" if room_to_sup > 0.3 else "⚠️" if room_to_sup > 0.1 else "❌"

    auto_checks = [chk_sig, chk_pcr, chk_pain, chk_iv, chk_res, chk_sup, atm_oi_ok, chk_liq]
    passed      = sum(1 for c in auto_checks if c == "✅")
    total       = len(auto_checks)

    if passed >= 6:
        verdict = "🟢 **GREEN LIGHT** — {}/{} auto-checks pass".format(passed, total)
    elif passed >= 4:
        verdict = "🟡 **CAUTION** — {}/{} pass, review warnings".format(passed, total)
    else:
        verdict = "🔴 **WAIT** — only {}/{} pass, skip this setup".format(passed, total)

    # Trade instrument
    if signal == "BUY CALL":
        instr = "ATM CE → ₹{} CE  (LTP ≈ ₹{:.1f})".format(int(atm), ce_ltp) if ce_ltp > 0 else "ATM CE"
    elif signal == "BUY PUT":
        instr = "ATM PE → ₹{} PE  (LTP ≈ ₹{:.1f})".format(int(atm), pe_ltp) if pe_ltp > 0 else "ATM PE"
    else:
        instr = "No directional trade — WAIT"

    ref_ltp = (ce_ltp if signal == "BUY CALL" else pe_ltp) if signal != "AVOID / WAIT" else 0
    sl_ltp  = round(ref_ltp * (1 - tf["sl_pct"]),  1) if ref_ltp > 0 else 0
    tgt_ltp = round(ref_ltp * (1 + tf["tgt_pct"]), 1) if ref_ltp > 0 else 0

    lines = [
        "## 📋 {} Checklist — {}".format(tf["label"], symbol),
        "",
        verdict,
        "",
        "---",
        "### Auto Checks (from live option chain data)",
        "",
        "**Direction & Signal**",
        "{} Signal: **{}** · Confidence {}% (need ≥{}%)".format(
            chk_sig, signal, int(confidence), conf_thresh),
        "{} PCR: **{}** → {}".format(
            chk_pcr, _fmt(pcr),
            "Bullish" if pcr > 1.1 else "Bearish" if pcr < 0.9 else "Neutral"
        ),
        "{} Max Pain: **₹{}** {} spot → {} pull".format(
            chk_pain, _fmt(max_pain, 0),
            "above" if pain_diff > 0 else "below",
            "upward" if pain_diff > 0 else "downward"
        ),
        "{} ATM OI buildup: **{}**".format(atm_oi_ok, atm_oi_bias),
        "",
        "**Options Pricing**",
        "{} IV: **{:.1f}%** — {}".format(
            chk_iv, avg_iv,
            "cheap, good to buy" if avg_iv < 14 else
            "fair value" if avg_iv < 22 else "expensive, prefer selling strategy"
        ),
        "{} Liquidity: **{}**".format(chk_liq, liq_note),
        "",
        "**Risk / Room**",
        "{} Resistance: ₹{} CE wall — **{:.1f}% headroom**".format(chk_res, int(ce_res), room_to_res),
        "{} Support: ₹{} PE floor — **{:.1f}% below**".format(chk_sup, int(pe_sup), room_to_sup),
        "",
        "---",
        "### Manual Checks (verify on your chart)",
        "",
    ]
    for chk in tf["manual_checks"]:
        lines.append("⬜ " + chk)

    lines += [
        "",
        "---",
        "### Trade Parameters — {}".format(tf["label"]),
        "",
        "| | |",
        "|---|---|",
        "| Instrument | {} |".format(instr),
        "| Stop-loss | ₹{:.1f} (−{}% of premium) |".format(sl_ltp, int(tf["sl_pct"]*100))
            if sl_ltp > 0 else "| Stop-loss | {}% of your entry premium |".format(int(tf["sl_pct"]*100)),
        "| Target | ₹{:.1f} (+{}%) |".format(tgt_ltp, int(tf["tgt_pct"]*100))
            if tgt_ltp > 0 else "| Target | {}% gain on premium |".format(int(tf["tgt_pct"]*100)),
        "| R:R | {} |".format(tf["rr"]),
        "| Max hold | {} |".format(tf["max_hold"]),
        "| Best entry window | {} |".format(tf["best_time"]),
        "| Chart EMA | {} |".format(tf["chart_ema"]),
        "| Extra indicator | {} |".format(tf["chart_extra"]),
        "| Lot rule | 1 lot per ₹50,000 capital |",
        "| Volume needed | {} |".format(tf["vol_mult"]),
        "",
        "---",
        "### Key Tips — {}".format(tf["label"]),
        "",
    ]
    for tip in tf["tips"]:
        lines.append("• " + tip)

    lines += [
        "",
        "---",
        "### General Options Scalping Rules (all timeframes)",
        "",
        "• Prefer ATM strikes — tightest spreads, best delta (0.5)",
        "• Avoid last 30 min before weekly expiry — gamma explosion risk",
        "• Rising OI + rising LTP = fresh long buildup (strong signal)",
        "• Falling OI + rising LTP = short covering (weaker, may reverse)",
        "• Never average down on a losing options position",
        "• Exit at SL without hesitation — 1 loss = 2 wins wiped at 40% SL",
        "",
        "⚠️ Educational only — not financial advice.",
    ]
    return "\n".join(lines)


# ── Keyword-based answers ─────────────────────────────────────────────────────

def _keyword_answer(query: str, df: pd.DataFrame, meta: dict, sig: dict,
                    timeframe: str = "1-hr") -> str:
    q = query.lower().strip()

    underlying = _safe_float(meta.get("underlying", 0))
    expiry     = meta.get("expiry", "N/A")
    atm        = _safe_float(meta.get("atm", underlying))
    pcr        = _safe_float(sig.get("pcr", 0))
    max_pain   = _safe_float(sig.get("max_pain", 0))
    signal     = sig.get("signal", "N/A")
    confidence = _safe_float(sig.get("confidence", 0))
    score      = _safe_float(sig.get("score", 0))
    reasons    = sig.get("reasons", [])
    ce_res     = _safe_float(sig.get("max_ce_resistance", 0))
    pe_sup     = _safe_float(sig.get("max_pe_support", 0))

    # ── Trade decision ────────────────────────────────────────────────────────
    if any(w in q for w in [
        "trade decision", "should i buy", "should i sell", "should i trade",
        "trade recommendation", "entry", "go long", "go short",
        "buy or sell", "call or put", "decision",
    ]):
        return _trade_decision(df, meta, sig)

    # ── Scalping / intraday checklist ─────────────────────────────────────────
    if any(w in q for w in [
        "scalp", "scalping", "intraday", "checklist",
        "what to check", "before trade", "pre trade", "pre-trade",
        "5 min", "5min", "15 min", "15min", "30 min", "30min",
        "1 hour", "one hour", "2 hour", "two hour", "daily swing",
    ]):
        # Detect requested timeframe from query
        # Detect requested timeframe from query; fallback to caller-supplied default
        tf = timeframe
        if any(w in q for w in ["5 min", "5min", "5-min"]):
            tf = "5-min"
        elif any(w in q for w in ["15 min", "15min", "15-min"]):
            tf = "15-min"
        elif any(w in q for w in ["30 min", "30min", "30-min"]):
            tf = "30-min"
        elif any(w in q for w in ["2 hour", "two hour", "2hr", "2-hr"]):
            tf = "2-hr"
        elif any(w in q for w in ["daily", "swing", "positional"]):
            tf = "daily"
        elif any(w in q for w in ["1 hour", "one hour", "1hr", "1-hr"]):
            tf = "1-hr"
        return _scalping_checklist(df, meta, sig, timeframe=tf)

    # ── Signal / trade ────────────────────────────────────────────────────────
    if any(w in q for w in ["signal", "buy", "sell", "trade", "recommend"]):
        color_map = {"green": "BULLISH", "red": "BEARISH", "orange": "NEUTRAL"}
        direction = color_map.get(sig.get("color", ""), "NEUTRAL")
        lines = [
            "**Current Signal: {} ({})**".format(signal, direction),
            "Confidence: {}%  |  Score: {}".format(int(confidence), int(score)),
            "",
            "**Reasons:**",
        ]
        for r in reasons:
            lines.append("• " + r)
        return "\n".join(lines)

    # ── PCR ───────────────────────────────────────────────────────────────────
    if any(w in q for w in ["pcr", "put call", "put-call", "put/call"]):
        if pcr > 1.5:
            interp = "EXTREMELY BULLISH — very heavy put writing by institutions"
        elif pcr > 1.2:
            interp = "BULLISH — strong put writing; market makers expect support"
        elif pcr > 0.8:
            interp = "NEUTRAL to BULLISH — balanced options activity"
        elif pcr > 0.5:
            interp = "NEUTRAL to BEARISH — slightly elevated call writing"
        else:
            interp = "BEARISH — heavy call writing; institutions hedging downside"

        return (
            "**Put-Call Ratio (PCR): {}**\n\n"
            "Interpretation: {}\n\n"
            "PCR is calculated as Total Put OI ÷ Total Call OI.\n"
            "• PCR > 1.2 → Bullish (puts outnumber calls)\n"
            "• PCR 0.8–1.2 → Neutral\n"
            "• PCR < 0.8 → Bearish (calls outnumber puts)"
        ).format(_fmt(pcr), interp)

    # ── Max Pain ──────────────────────────────────────────────────────────────
    if any(w in q for w in ["max pain", "maxpain", "pain", "expiry"]):
        diff     = max_pain - underlying
        dist_pct = abs(diff) / underlying * 100 if underlying else 0
        direction = "above" if diff > 0 else "below"
        pull      = "upward pull toward max pain" if diff > 0 else "downward pull toward max pain"

        return (
            "**Max Pain: ₹{}**\n\n"
            "Max Pain is ₹{} ({:.2f}%) {} current spot (₹{}).\n\n"
            "Theory: At expiry, underlying tends to gravitate toward the strike\n"
            "where the maximum number of option contracts expire worthless, causing\n"
            "maximum loss for option buyers. This suggests {}.\n\n"
            "Expiry: {}"
        ).format(
            _fmt(max_pain, 0), _fmt(abs(diff), 0), dist_pct, direction,
            _fmt(underlying, 0), pull, expiry,
        )

    # ── CE Resistance ─────────────────────────────────────────────────────────
    if any(w in q for w in ["resistance", "ce oi", "call oi", "ce writing", "call writing"]):
        top_ce = _top_oi_strikes(df, "ce_oi", 3)
        lines  = [
            "**CE (Call) Resistance Levels:**",
            "",
            "Max CE OI (strongest resistance): ₹{}".format(_fmt(ce_res, 0)),
            "",
            "Top 3 CE OI strikes:",
        ]
        for strike, oi in top_ce:
            marker = " ← KEY RESISTANCE" if strike == ce_res else ""
            lines.append("  • Strike ₹{}: OI {:,.0f} lots{}".format(int(strike), oi, marker))
        lines += [
            "",
            "High Call OI at a strike = strong resistance. Market makers are short\n"
            "calls there and will hedge by selling the underlying if price approaches.",
        ]
        return "\n".join(lines)

    # ── PE Support ────────────────────────────────────────────────────────────
    if any(w in q for w in ["support", "pe oi", "put oi", "pe writing", "put writing"]):
        top_pe = _top_oi_strikes(df, "pe_oi", 3)
        lines  = [
            "**PE (Put) Support Levels:**",
            "",
            "Max PE OI (strongest support): ₹{}".format(_fmt(pe_sup, 0)),
            "",
            "Top 3 PE OI strikes:",
        ]
        for strike, oi in top_pe:
            marker = " ← KEY SUPPORT" if strike == pe_sup else ""
            lines.append("  • Strike ₹{}: OI {:,.0f} lots{}".format(int(strike), oi, marker))
        lines += [
            "",
            "High Put OI at a strike = strong support. Put writers will buy the\n"
            "underlying to hedge if price approaches, creating a floor effect.",
        ]
        return "\n".join(lines)

    # ── IV / Volatility ───────────────────────────────────────────────────────
    if any(w in q for w in ["iv", "implied vol", "volatility", "vix", "premium"]):
        ce_iv, pe_iv = _atm_iv(df, underlying)
        avg_iv = (ce_iv + pe_iv) / 2 if (ce_iv + pe_iv) > 0 else 0

        daily_move = underlying * (avg_iv / 100) * math.sqrt(1 / 252) if avg_iv > 0 else 0

        level = "LOW (cheap premiums)" if avg_iv < 12 else \
                "MODERATE" if avg_iv < 20 else \
                "HIGH (expensive premiums)"

        return (
            "**Implied Volatility (ATM)**\n\n"
            "ATM Strike: ₹{}\n"
            "CE IV: {:.1f}%  |  PE IV: {:.1f}%  |  Average: {:.1f}%\n\n"
            "IV Level: {}\n\n"
            "Expected daily move (1σ): ±₹{:,.0f}\n\n"
            "High IV → options are expensive; prefer selling strategies.\n"
            "Low IV → options are cheap; prefer buying strategies."
        ).format(_fmt(atm, 0), ce_iv, pe_iv, avg_iv, level, daily_move)

    # ── Tomorrow prediction ───────────────────────────────────────────────────
    if any(w in q for w in ["tomorrow", "predict", "next day", "forecast", "direction"]):
        ce_iv, pe_iv = _atm_iv(df, underlying)
        avg_iv       = (ce_iv + pe_iv) / 2 if (ce_iv + pe_iv) > 0 else 15.0
        daily_move   = underlying * (avg_iv / 100) * math.sqrt(1 / 252)

        if score > 20:
            bias = "BULLISH — lean long / buy calls"
        elif score < -20:
            bias = "BEARISH — lean short / buy puts"
        else:
            bias = "NEUTRAL — avoid directional bets"

        pain_dir = "above" if max_pain > underlying else "below"

        return (
            "**Tomorrow's Prediction for {}**\n\n"
            "Directional Bias: {}\n"
            "Signal Score: {}\n\n"
            "Expected Range:\n"
            "  Upper: ₹{} (+₹{:,.0f})\n"
            "  Lower: ₹{} (-₹{:,.0f})\n\n"
            "Max Pain (₹{}) is {} spot → {} pull at expiry\n\n"
            "Key Levels:\n"
            "  Resistance: ₹{} (max CE OI)\n"
            "  Support: ₹{} (max PE OI)\n\n"
            "Expiry: {}\n\n"
            "⚠️ For educational use only. Not financial advice."
        ).format(
            meta.get("symbol", "INDEX"),
            bias, int(score),
            _fmt(underlying + daily_move, 0), daily_move,
            _fmt(underlying - daily_move, 0), daily_move,
            _fmt(max_pain, 0), pain_dir,
            "upward" if max_pain > underlying else "downward",
            _fmt(ce_res, 0), _fmt(pe_sup, 0),
            expiry,
        )

    # ── Spot / price / level ──────────────────────────────────────────────────
    if any(w in q for w in ["spot", "price", "level", "index", "underlying", "ltp", "current"]):
        return (
            "**Current Market Snapshot**\n\n"
            "Spot Price: ₹{}\n"
            "ATM Strike: ₹{}\n"
            "Expiry: {}\n\n"
            "CE Resistance: ₹{} (max call OI)\n"
            "PE Support:    ₹{} (max put OI)\n"
            "Max Pain:      ₹{}"
        ).format(
            _fmt(underlying), _fmt(atm, 0), expiry,
            _fmt(ce_res, 0), _fmt(pe_sup, 0), _fmt(max_pain, 0),
        )

    # ── Open Interest ─────────────────────────────────────────────────────────
    if any(w in q for w in ["oi", "open interest", "open-interest", "buildup", "unwinding"]):
        top_ce = _top_oi_strikes(df, "ce_oi", 3)
        top_pe = _top_oi_strikes(df, "pe_oi", 3)

        lines = ["**Open Interest Summary**", ""]
        lines.append("Top 3 CE OI (Resistance Zones):")
        for s, o in top_ce:
            lines.append("  ₹{}: {:,.0f} contracts".format(int(s), o))

        lines.append("")
        lines.append("Top 3 PE OI (Support Zones):")
        for s, o in top_pe:
            lines.append("  ₹{}: {:,.0f} contracts".format(int(s), o))

        if not df.empty and "ce_chg_oi" in df.columns:
            net_chg = df["pe_chg_oi"].sum() - df["ce_chg_oi"].sum()
            oi_bias = "Bullish (net PUT writing)" if net_chg > 0 else "Bearish (net CALL writing)"
            lines += ["", "Net Change in OI: {:,.0f} → {}".format(int(abs(net_chg)), oi_bias)]

        return "\n".join(lines)

    # ── Help ──────────────────────────────────────────────────────────────────
    if any(w in q for w in ["help", "what can", "topics", "commands", "capabilities"]):
        return (
            "**NSE Options Bot — Available Topics**\n\n"
            "Ask me about any of these:\n\n"
            "🎯 **trade decision** / should I buy / entry\n"
            "   → Full BUY/AVOID recommendation with entry, SL, target\n\n"
            "📋 **checklist** / intraday / scalping + timeframe\n"
            "   → **5-min checklist** · **15-min checklist** · **30-min checklist**\n"
            "   → **1-hour checklist** (default) · **2-hour checklist** · **daily checklist**\n"
            "   → Each has timeframe-specific RSI levels, SL%, hold time, tips\n\n"
            "📊 **signal** / trade / buy / sell\n"
            "   → Current signal with reasons\n\n"
            "📉 **pcr** / put call ratio\n"
            "   → PCR value and bullish/bearish interpretation\n\n"
            "🎯 **max pain** / expiry / pain\n"
            "   → Max pain level and distance from spot\n\n"
            "🔴 **resistance** / CE OI / call OI\n"
            "   → Key call resistance levels\n\n"
            "🟢 **support** / PE OI / put OI\n"
            "   → Key put support levels\n\n"
            "⚡ **IV** / implied volatility / premium\n"
            "   → ATM IV and expected daily move\n\n"
            "🔮 **tomorrow** / predict / forecast\n"
            "   → Tomorrow's prediction and key levels\n\n"
            "💹 **spot** / price / level / index\n"
            "   → Current spot, ATM, and key levels\n\n"
            "📋 **OI** / open interest / buildup\n"
            "   → Top OI strikes for CE and PE"
        )

    # ── Default fallback ──────────────────────────────────────────────────────
    return (
        "**Current Snapshot for {}**\n\n"
        "Spot: ₹{}  |  ATM: ₹{}  |  Expiry: {}\n"
        "Signal: {} ({}% confidence)\n"
        "PCR: {}  |  Max Pain: ₹{}\n"
        "CE Resistance: ₹{}  |  PE Support: ₹{}\n\n"
        "Ask: **trade decision** (full analysis) · **scalping checklist** · "
        "**signal** · **pcr** · **max pain** · **IV** · **tomorrow**\n"
        "Type **help** for all topics."
    ).format(
        meta.get("symbol", "INDEX"),
        _fmt(underlying), _fmt(atm, 0), expiry,
        signal, int(confidence),
        _fmt(pcr), _fmt(max_pain, 0),
        _fmt(ce_res, 0), _fmt(pe_sup, 0),
    )


# ── Claude API answer ─────────────────────────────────────────────────────────

def _build_context(df: pd.DataFrame, meta: dict, sig: dict) -> str:
    underlying = _safe_float(meta.get("underlying", 0))
    expiry     = meta.get("expiry", "N/A")
    atm        = _safe_float(meta.get("atm", underlying))
    pcr        = _safe_float(sig.get("pcr", 0))
    max_pain   = _safe_float(sig.get("max_pain", 0))
    signal     = sig.get("signal", "N/A")
    confidence = _safe_float(sig.get("confidence", 0))
    score      = _safe_float(sig.get("score", 0))
    ce_res     = _safe_float(sig.get("max_ce_resistance", 0))
    pe_sup     = _safe_float(sig.get("max_pe_support", 0))
    reasons    = sig.get("reasons", [])

    ce_iv, pe_iv = _atm_iv(df, underlying)
    ce_ltp, pe_ltp = _atm_ltp(df, underlying)

    top_ce = _top_oi_strikes(df, "ce_oi", 3)
    top_pe = _top_oi_strikes(df, "pe_oi", 3)

    # Near-ATM OI change for context
    near_ce_chg = near_pe_chg = 0.0
    if not df.empty:
        try:
            _idx = int((df["strike"] - underlying).abs().idxmin())
            _near = df.iloc[max(0, _idx - 2):min(len(df), _idx + 3)]
            near_ce_chg = float(_near["ce_chg_oi"].sum())
            near_pe_chg = float(_near["pe_chg_oi"].sum())
        except Exception:
            pass

    total_ce_chg = float(df["ce_chg_oi"].sum()) if not df.empty else 0.0
    total_pe_chg = float(df["pe_chg_oi"].sum()) if not df.empty else 0.0
    net_oi_flow  = total_pe_chg - total_ce_chg
    iv_skew      = pe_iv - ce_iv

    ctx_lines = [
        "=== NSE OPTIONS LIVE DATA ===",
        "Symbol: {}  Expiry: {}  Spot: Rs.{}  ATM: Rs.{}".format(
            meta.get("symbol", "NIFTY"), expiry, _fmt(underlying), _fmt(atm, 0)
        ),
        "SIGNAL: {} | Confidence: {}% | Score: {}".format(signal, int(confidence), int(score)),
        "PCR: {} ({}) | MaxPain: Rs.{} ({}) | CE_Res: Rs.{} | PE_Sup: Rs.{}".format(
            _fmt(pcr),
            "BULLISH" if pcr > 1.2 else "BEARISH" if pcr < 0.8 else "NEUTRAL",
            _fmt(max_pain, 0),
            "Rs.{:,.0f} ABOVE spot".format(max_pain - underlying) if max_pain > underlying
                else "Rs.{:,.0f} BELOW spot".format(underlying - max_pain),
            _fmt(ce_res, 0), _fmt(pe_sup, 0)
        ),
        "ATM IV: CE {:.1f}% | PE {:.1f}% | Skew (PE-CE): {:+.1f}% | ATM LTP: CE Rs.{:.1f} PE Rs.{:.1f}".format(
            ce_iv, pe_iv, iv_skew, ce_ltp, pe_ltp
        ),
        "OI Flow total: CE chg {:+,.0f} | PE chg {:+,.0f} | Net {:+,.0f} ({})".format(
            total_ce_chg, total_pe_chg, net_oi_flow,
            "PE building BULLISH" if net_oi_flow > 0 else "CE building BEARISH"
        ),
        "OI Flow near ATM: CE {:+,.0f} | PE {:+,.0f} | {}".format(
            near_ce_chg, near_pe_chg,
            "PE writing at ATM (bullish)" if near_pe_chg > near_ce_chg else "CE writing at ATM (bearish)"
        ),
        "Signal reasons: " + "; ".join(reasons),
        "Top CE OI (resistance): " + ", ".join(
            "Rs.{}={:,.0f}".format(int(s), o) for s, o in top_ce
        ),
        "Top PE OI (support): " + ", ".join(
            "Rs.{}={:,.0f}".format(int(s), o) for s, o in top_pe
        ),
        "Strategy: {} | {}".format(
            sig.get("strategy_type", ""), sig.get("strategy_note", "")
        ),
    ]
    return "\n".join(ctx_lines)


def _claude_answer(
    query: str,
    df: pd.DataFrame,
    meta: dict,
    sig: dict,
    api_key: str,
) -> str:
    try:
        import anthropic

        context = _build_context(df, meta, sig)

        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            system=(
                "You are an expert NSE options markets analyst. "
                "Use ONLY the provided live option chain data to answer questions — do not guess or use generic answers. "
                "Always cite the specific numbers from the data (PCR, Max Pain, OI levels, IV%). "
                "For trade decision / entry questions: give a clear BUY CALL / BUY PUT / AVOID verdict, "
                "state the specific entry strike, estimated premium, stop-loss (Rs. value), and target (Rs. value). "
                "For checklist questions: list each check with PASS/FAIL based on actual data values. "
                "For OI / PCR / IV questions: explain what the specific number means for this instrument today. "
                "For prediction questions: use PCR, Max Pain distance from spot, and OI buildup to forecast direction. "
                "Be detailed — the user wants analysis, not one-liners. "
                "Minimum 5-8 lines of analysis for any trade-related question. "
                "Use Rs. for Indian Rupees (not the symbol). "
                "End every response with a one-line bottom-line recommendation. "
                "Always add: 'Educational purposes only - not financial advice.'"
            ),
            messages=[
                {
                    "role": "user",
                    "content": "Live NSE option chain data:\n{}\n\nQuestion: {}".format(
                        context, query
                    ),
                }
            ],
        )

        response_text = message.content[0].text if message.content else ""
        if not response_text:
            raise ValueError("Empty response from Claude API")

        return response_text

    except ImportError:
        return _keyword_answer(query, df, meta, sig)
    except Exception:
        return _keyword_answer(query, df, meta, sig)


# ── Public API ────────────────────────────────────────────────────────────────

def answer(query: str, df: pd.DataFrame, meta: dict, sig: dict,
           timeframe: str = "1-hr") -> str:
    """
    Main entry point. Tries Claude API first if anthropic_api_key is in
    st.secrets, otherwise uses keyword-based matching.
    timeframe: one of "5-min", "15-min", "30-min", "1-hr", "2-hr", "daily"
    """
    if not query or not query.strip():
        return "Please ask a question. Type **help** for topics, or ask for a **trade decision**."

    if "symbol" not in meta:
        meta = dict(meta)
        meta["symbol"] = "INDEX"

    api_key: str | None = None
    try:
        import streamlit as st
        api_key = st.secrets.get("anthropic_api_key", None)
    except Exception:
        pass

    if api_key:
        return _claude_answer(query, df, meta, sig, api_key)

    return _keyword_answer(query, df, meta, sig, timeframe=timeframe)

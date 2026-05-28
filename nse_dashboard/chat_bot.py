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


# ── Scalping Checklist ────────────────────────────────────────────────────────

def _scalping_checklist(df: pd.DataFrame, meta: dict, sig: dict) -> str:
    """
    Dynamic pass/fail checklist for intraday 1-hour scalping trades.
    Checks what can be derived from option chain data; flags manual checks.
    """
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

    # ATM liquidity (OI > 50k is decent)
    atm_ce_oi = 0.0
    atm_pe_oi = 0.0
    if not df.empty:
        try:
            idx = (df["strike"] - underlying).abs().idxmin()
            atm_ce_oi = _safe_float(df.iloc[idx].get("ce_oi", 0))
            atm_pe_oi = _safe_float(df.iloc[idx].get("pe_oi", 0))
        except Exception:
            pass
    liq_oi = atm_ce_oi if signal == "BUY CALL" else atm_pe_oi
    chk_liq = "✅" if liq_oi > 50000 else "⚠️" if liq_oi > 10000 else "❌"
    liq_note = "{:,.0f} lots at ATM".format(liq_oi)

    room_to_res = (ce_res - underlying) / max(underlying, 1) * 100
    room_to_sup = (underlying - pe_sup) / max(underlying, 1) * 100
    pain_diff   = max_pain - underlying

    chk_sig  = "✅" if confidence >= 65 else "⚠️" if confidence >= 50 else "❌"
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

    # Preferred instrument for scalp
    if signal == "BUY CALL":
        instr = "ATM CE → ₹{} CE  (LTP ≈ ₹{:.1f})".format(int(atm), ce_ltp) if ce_ltp > 0 else "ATM CE"
    elif signal == "BUY PUT":
        instr = "ATM PE → ₹{} PE  (LTP ≈ ₹{:.1f})".format(int(atm), pe_ltp) if pe_ltp > 0 else "ATM PE"
    else:
        instr = "No directional trade — WAIT"

    ref_ltp = (ce_ltp if signal == "BUY CALL" else pe_ltp) if signal != "AVOID / WAIT" else 0
    sl_ltp  = round(ref_ltp * 0.60, 1) if ref_ltp > 0 else 0
    tgt_ltp = round(ref_ltp * 1.55, 1) if ref_ltp > 0 else 0

    lines = [
        "## 📋 Scalping Checklist — {} (1-Hour Intraday)".format(symbol),
        "",
        verdict,
        "",
        "---",
        "### Auto Checks (from live data)",
        "",
        "**Direction & Signal**",
        "{} Signal: **{}** · Confidence {}%".format(chk_sig, signal, int(confidence)),
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
            "fair value" if avg_iv < 22 else "expensive, prefer selling"
        ),
        "{} Liquidity: **{}**".format(chk_liq, liq_note),
        "",
        "**Risk / Room**",
        "{} Resistance: ₹{} CE wall — **{:.1f}% headroom**".format(chk_res, int(ce_res), room_to_res),
        "{} Support: ₹{} PE floor — **{:.1f}% below**".format(chk_sup, int(pe_sup), room_to_sup),
        "",
        "---",
        "### Manual Checks (do these on your chart)",
        "",
        "⬜ **15-min chart:** Close above 20-EMA? (calls) / Below? (puts)",
        "⬜ **15-min RSI:** > 55 for calls · < 45 for puts",
        "⬜ **Volume:** Current candle volume > 3-bar average?",
        "⬜ **1-hr trend:** Higher highs & higher lows (calls) / Lower lows (puts)?",
        "⬜ **No events:** Any RBI, GDP, earnings, or major news next 2 hours?",
        "⬜ **Time window:** Are you in 9:20–10:30 AM or 1:30–3:00 PM IST?",
        "",
        "---",
        "### Trade Parameters",
        "",
        "| | |",
        "|---|---|",
        "| Instrument | {} |".format(instr),
        "| Stop-loss | ₹{:.1f} (−40% of premium) |".format(sl_ltp) if sl_ltp > 0 else "| Stop-loss | 40% of your entry premium |",
        "| Target | ₹{:.1f} (+55%) |".format(tgt_ltp) if tgt_ltp > 0 else "| Target | 55% gain on premium |",
        "| Max hold | 1 hour / 3:15 PM |",
        "| Lot rule | 1 lot per ₹50,000 capital |",
        "| R:R | 1 : 1.4 (40% SL, 55% target) |",
        "",
        "---",
        "### What to check for scalping (theory)",
        "",
        "**Price action:**",
        "• Breakout of previous 15-min high (for calls) or low (for puts)",
        "• Strong close candle — not a doji or inside bar",
        "",
        "**Options-specific:**",
        "• Prefer ATM strikes — tightest spreads, best delta",
        "• Avoid 30+ mins before expiry — gamma explosion risk",
        "• Watch bid-ask spread: > ₹5 gap at ATM = avoid (low liquidity)",
        "• Rising OI + rising LTP = fresh buildup (strong signal)",
        "• Falling OI + rising LTP = short covering (weaker signal)",
        "",
        "**Risk management:**",
        "• Never risk > 2% of capital on one scalp",
        "• Exit at SL without hesitation — no averaging down in options",
        "• Bank partial profit at Target 1, trail the rest",
        "",
        "⚠️ Educational only — not financial advice.",
    ]
    return "\n".join(lines)


# ── Keyword-based answers ─────────────────────────────────────────────────────

def _keyword_answer(query: str, df: pd.DataFrame, meta: dict, sig: dict) -> str:
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
        "scalp", "scalping", "intraday", "1 hour", "one hour", "checklist",
        "what to check", "before trade", "pre trade", "pre-trade",
    ]):
        return _scalping_checklist(df, meta, sig)

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
            "📋 **scalping** / intraday / checklist\n"
            "   → Dynamic pass/fail checklist for 1-hour intraday trades\n\n"
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

    ctx_lines = [
        "=== NSE OPTIONS LIVE DATA ===",
        "Symbol: {}  Expiry: {}  Spot: {}  ATM: {}".format(
            meta.get("symbol", "NIFTY"), expiry,
            _fmt(underlying), _fmt(atm, 0)
        ),
        "Signal: {} | Confidence: {}% | Score: {}".format(signal, int(confidence), int(score)),
        "PCR: {} | MaxPain: {} | CE_Res: {} | PE_Sup: {}".format(
            _fmt(pcr), _fmt(max_pain, 0), _fmt(ce_res, 0), _fmt(pe_sup, 0)
        ),
        "ATM IV — CE: {:.1f}% PE: {:.1f}%  |  ATM LTP — CE: ₹{:.1f} PE: ₹{:.1f}".format(
            ce_iv, pe_iv, ce_ltp, pe_ltp
        ),
        "Reasons: " + "; ".join(reasons),
        "Top CE OI strikes: " + ", ".join(
            "₹{}={:,.0f}".format(int(s), o) for s, o in top_ce
        ),
        "Top PE OI strikes: " + ", ".join(
            "₹{}={:,.0f}".format(int(s), o) for s, o in top_pe
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
            max_tokens=600,
            system=(
                "You are an NSE options markets expert assistant. "
                "Answer questions using only the provided option chain data context. "
                "Be concise, factual and educational. "
                "For trade decision / entry questions, give a clear BUY CALL / BUY PUT / AVOID "
                "recommendation with entry, stop-loss, and target levels. "
                "For scalping / intraday questions, provide a structured checklist. "
                "Always remind the user this is for educational purposes only and not financial advice. "
                "Use ₹ symbol for Indian Rupees."
            ),
            messages=[
                {
                    "role": "user",
                    "content": "Option chain context:\n{}\n\nQuestion: {}".format(
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

def answer(query: str, df: pd.DataFrame, meta: dict, sig: dict) -> str:
    """
    Main entry point. Tries Claude API first if anthropic_api_key is in
    st.secrets, otherwise uses keyword-based matching.
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

    return _keyword_answer(query, df, meta, sig)

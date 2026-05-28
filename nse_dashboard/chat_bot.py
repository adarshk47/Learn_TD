"""
NSE Options Intelligence Chat Bot.

Answers natural-language questions about the current option chain data.

Two modes:
  1. Keyword-based  — always works, no external API required.
  2. Claude API     — used when `anthropic_api_key` is present in st.secrets
                      (falls back to keyword mode on any error).
"""

from __future__ import annotations

import re
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
    """Format a number with thousand separators."""
    if decimals == 0:
        return "{:,.0f}".format(val)
    return "{:,.{}f}".format(val, decimals)


def _top_oi_strikes(df: pd.DataFrame, col: str, n: int = 3) -> list:
    """Return top-n strikes sorted by OI column descending."""
    if df.empty or col not in df.columns:
        return []
    top = df.nlargest(n, col)[["strike", col]]
    return [(row["strike"], row[col]) for _, row in top.iterrows()]


def _atm_iv(df: pd.DataFrame, underlying: float) -> tuple:
    """Return (ce_iv, pe_iv) for the ATM strike."""
    if df.empty:
        return 0.0, 0.0
    try:
        idx = (df["strike"] - underlying).abs().idxmin()
        row = df.iloc[idx]
        return _safe_float(row.get("ce_iv", 0)), _safe_float(row.get("pe_iv", 0))
    except Exception:
        return 0.0, 0.0


# ── Keyword-based answers ─────────────────────────────────────────────────────

def _keyword_answer(query: str, df: pd.DataFrame, meta: dict, sig: dict) -> str:
    """
    Match query against keyword topics and return a formatted string answer.
    """
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

        # Expected daily move
        import math
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
        import math
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
        "I can answer questions about: **signal, pcr, max pain, resistance, "
        "support, IV/volatility, tomorrow's prediction, open interest**.\n"
        "Type **help** for the full topic list."
    ).format(
        meta.get("symbol", "INDEX"),
        _fmt(underlying), _fmt(atm, 0), expiry,
        signal, int(confidence),
        _fmt(pcr), _fmt(max_pain, 0),
        _fmt(ce_res, 0), _fmt(pe_sup, 0),
    )


# ── Claude API answer ─────────────────────────────────────────────────────────

def _build_context(df: pd.DataFrame, meta: dict, sig: dict) -> str:
    """Build a compact context string from live option chain data."""
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
        "ATM IV — CE: {:.1f}% PE: {:.1f}%".format(ce_iv, pe_iv),
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
    """
    Call the Anthropic Claude API (claude-haiku-4-5-20251001) and return the response.

    Falls back to keyword answer if the API call fails.
    """
    try:
        import anthropic  # type: ignore

        context = _build_context(df, meta, sig)

        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=(
                "You are an NSE options markets expert assistant. "
                "Answer questions using only the provided option chain data context. "
                "Be concise, factual and educational. "
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
        # anthropic package not installed → keyword fallback
        return _keyword_answer(query, df, meta, sig)
    except Exception:
        # Any API error → keyword fallback
        return _keyword_answer(query, df, meta, sig)


# ── Public API ────────────────────────────────────────────────────────────────

def answer(query: str, df: pd.DataFrame, meta: dict, sig: dict) -> str:
    """
    Main entry point for the chat bot.

    Tries Claude API first if `anthropic_api_key` is available in st.secrets,
    otherwise uses keyword-based matching.

    Parameters
    ----------
    query : str         — user's natural language question
    df    : pd.DataFrame — current option chain DataFrame
    meta  : dict         — metadata from parse_option_chain (underlying, expiry, atm …)
    sig   : dict         — signal dict from generate_signal (pcr, max_pain, signal …)

    Returns
    -------
    str — markdown-formatted answer
    """
    if not query or not query.strip():
        return "Please ask a question about the option chain data. Type **help** for topics."

    # Enrich meta with symbol if not already set
    if "symbol" not in meta:
        meta = dict(meta)
        meta["symbol"] = "INDEX"

    # Try Streamlit secrets for API key
    api_key: str | None = None
    try:
        import streamlit as st  # type: ignore
        api_key = st.secrets.get("anthropic_api_key", None)
    except Exception:
        pass

    if api_key:
        return _claude_answer(query, df, meta, sig, api_key)

    return _keyword_answer(query, df, meta, sig)

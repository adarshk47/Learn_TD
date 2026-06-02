"""
smart_analysis.py — Multi-framework trade intelligence engine.

Draws from the core principles of legendary traders / books:
  1. Mark Douglas  — Trading in the Zone (probability mindset, setup consistency)
  2. Jesse Livermore — Reminiscences of a Stock Operator (trend, tape, pivots)
  3. ICT / Smart Money — Institutional OI footprints, liquidity levels
  4. Stan Weinstein — Secrets for Profiting (4-stage market structure)
  5. Van Tharp   — Trade Your Way to Financial Freedom (R-multiples, expectancy)
  6. Historical Expiry Learning — pattern-match today vs last N expiry days
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(v, default=0.0):
    try:
        x = float(v)
        return default if math.isnan(x) else x
    except Exception:
        return default


def _market_hours_left():
    """Return actual NSE market hours remaining today (0.0 after close)."""
    from datetime import time as _t
    now   = datetime.now()
    close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    open_ = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    if now.time() < _t(9, 15):
        return 6.25
    if now.time() > _t(15, 30):
        return 0.0
    return round((close - now).total_seconds() / 3600, 2)


# ── 1. Mark Douglas — Probability / Confluence Scorer ───────────────────────

def mark_douglas_score(oi_sig, tech_df, pcr, dte):
    """
    "The market is a probability game. Each trade is just one of a series."
    Counts how many independent signals agree on the same direction.
    Returns (bull_count, bear_count, neutral_count, total, verdict, advice).
    """
    checks = []   # (name, +1 bull / -1 bear / 0 neutral)

    # ── Option chain signals ──────────────────────────────────────────────────
    score = oi_sig.get("score", 0)
    checks.append(("OI Net Score",   1 if score > 15 else -1 if score < -15 else 0))
    checks.append(("PCR",            1 if pcr > 1.1  else -1 if pcr < 0.8  else 0))
    checks.append(("Max Pain Pull",  1 if oi_sig.get("max_pain", 0) > oi_sig.get("underlying_proxy", 0)
                                     else -1 if oi_sig.get("max_pain", 0) < oi_sig.get("underlying_proxy", 0)
                                     else 0))
    checks.append(("ITM Ratio",      1 if oi_sig.get("itm_ratio", 1) > 1.3
                                     else -1 if oi_sig.get("itm_ratio", 1) < 0.7 else 0))

    # ── Technical signals (from compute_technicals output) ───────────────────
    if tech_df is not None and not tech_df.empty and len(tech_df) >= 5:
        r = tech_df.iloc[-1]

        rsi  = _safe(r.get("rsi", 50))
        checks.append(("RSI",     1 if rsi > 60 else -1 if rsi < 40 else 0))

        macd = _safe(r.get("macd", 0))
        msig = _safe(r.get("macd_sig", 0))
        checks.append(("MACD",    1 if macd > msig else -1))

        e9, e20 = _safe(r.get("ema9", 0)), _safe(r.get("ema20", 1))
        checks.append(("EMA9/20", 1 if e9 > e20 else -1))

        close, vwap = _safe(r.get("close", 0)), _safe(r.get("vwap", 0))
        if vwap > 0:
            checks.append(("VWAP",  1 if close > vwap else -1))

        if "pvt" in tech_df.columns and len(tech_df) >= 4:
            pvt_slope = float(tech_df["pvt"].iloc[-1] - tech_df["pvt"].iloc[-4])
            checks.append(("PVT",   1 if pvt_slope > 0 else -1))

    bull  = sum(1 for _, v in checks if v > 0)
    bear  = sum(1 for _, v in checks if v < 0)
    neut  = sum(1 for _, v in checks if v == 0)
    total = len(checks)

    if bull >= math.ceil(total * 0.75):
        verdict = "HIGH-PROBABILITY LONG"
        confidence = min(96, 60 + bull * 5)
    elif bear >= math.ceil(total * 0.75):
        verdict = "HIGH-PROBABILITY SHORT"
        confidence = min(96, 60 + bear * 5)
    elif bull > bear:
        verdict = "MODERATE BULLISH"
        confidence = 50 + (bull - bear) * 6
    elif bear > bull:
        verdict = "MODERATE BEARISH"
        confidence = 50 + (bear - bull) * 6
    else:
        verdict = "NO EDGE — WAIT"
        confidence = 30

    # DTE penalty
    if dte == 0:
        advice = "Expiry day: take only high-confluence scalps. Cut immediately if wrong."
    elif dte == 1:
        advice = "1 DTE: premium decay accelerates after 12 PM — prefer selling or tiny buy."
    elif dte <= 3:
        advice = "Short expiry: ATM directional trade with defined stop."
    else:
        advice = "Healthy DTE: trend trade or spread. Let the trade breathe."

    return {
        "checks":      checks,
        "bull":        bull,
        "bear":        bear,
        "neutral":     neut,
        "total":       total,
        "verdict":     verdict,
        "confidence":  confidence,
        "advice":      advice,
    }


# ── 2. Jesse Livermore — Trend & Pivot Analysis ──────────────────────────────

def livermore_analysis(candles, underlying, symbol):
    """
    "The big money is not in the buying and selling, but in the waiting."
    — Identifies the dominant trend, key pivots, and whether 'the tape' confirms.
    """
    if candles is None or candles.empty or len(candles) < 10:
        return {"trend": "UNKNOWN", "stage": "?", "pivots": [], "tape": "No data"}

    c = candles.copy().sort_values("datetime").reset_index(drop=True)

    # 20-day high/low range
    hi20 = c["high"].tail(20).max()
    lo20 = c["low"].tail(20).min()
    rng  = hi20 - lo20

    # 5-day momentum
    mom5 = float(c["close"].iloc[-1] - c["close"].iloc[-6]) if len(c) >= 6 else 0
    mom_pct = mom5 / float(c["close"].iloc[-6]) * 100 if len(c) >= 6 else 0

    # EMA 50 via daily candles (approximate with EMA21 if < 50 bars)
    span = min(50, len(c))
    ema_slow = float(c["close"].ewm(span=span, adjust=False).mean().iloc[-1])
    close_now = float(c["close"].iloc[-1])

    # Key pivot levels (last 2 swing highs / lows)
    highs = c["high"].rolling(5, center=True).max()
    lows  = c["low"].rolling(5,  center=True).min()
    pivot_hi = sorted(set(round(v, 0) for v in c[c["high"] == highs]["high"].tail(5)), reverse=True)[:2]
    pivot_lo = sorted(set(round(v, 0) for v in c[c["low"]  == lows]["low"].tail(5)))[:2]

    # Trend classification
    if close_now > ema_slow and mom_pct > 0.3:
        trend = "UPTREND"
        tape  = "Price above EMA, positive momentum — Livermore: go with the trend."
    elif close_now < ema_slow and mom_pct < -0.3:
        trend = "DOWNTREND"
        tape  = "Price below EMA, negative momentum — Livermore: don't fight the tape."
    else:
        trend = "SIDEWAYS"
        tape  = "Consolidation — Livermore: sit on hands, wait for the line of least resistance."

    # 4-stage (Weinstein overlay)
    c["ema30"] = c["close"].ewm(span=min(30, len(c)), adjust=False).mean()
    slope30 = float(c["ema30"].diff().tail(5).mean())
    if slope30 > 0 and close_now > float(c["ema30"].iloc[-1]):
        stage = "Stage 2 — Advancing (LONG bias)"
    elif slope30 < 0 and close_now < float(c["ema30"].iloc[-1]):
        stage = "Stage 4 — Declining (SHORT bias)"
    elif slope30 >= 0:
        stage = "Stage 1 — Basing (wait for breakout)"
    else:
        stage = "Stage 3 — Topping (tighten stops)"

    return {
        "trend":      trend,
        "stage":      stage,
        "tape":       tape,
        "pivot_hi":   pivot_hi,
        "pivot_lo":   pivot_lo,
        "mom5_pct":   round(mom_pct, 2),
        "ema_slow":   round(ema_slow, 1),
        "close":      round(close_now, 1),
        "range20":    round(rng, 1),
        "hi20":       round(hi20, 1),
        "lo20":       round(lo20, 1),
    }


# ── 3. ICT / Smart Money — Institutional Footprints ─────────────────────────

def ict_analysis(df_chain, underlying, sig):
    """
    ICT concept: big players show their hand in the OI.
    OI walls = institutional sell/buy zones. PCR tells bank positioning.
    """
    if df_chain.empty:
        return {}

    ce_wall = sig.get("max_ce_resistance", 0)
    pe_wall = sig.get("max_pe_support",    0)
    mp      = sig.get("max_pain", underlying)
    pcr     = sig.get("pcr", 1.0)

    # Distance to institutional levels
    dist_ce = round(ce_wall - underlying, 0)
    dist_pe = round(underlying - pe_wall, 0)
    dist_mp = round(mp - underlying, 0)

    # Liquidity zones: where there is concentrated OI → stops cluster there
    top3_ce = df_chain.nlargest(3, "ce_oi")[["strike", "ce_oi"]].to_dict("records")
    top3_pe = df_chain.nlargest(3, "pe_oi")[["strike", "pe_oi"]].to_dict("records")

    # ICT narrative
    if pcr > 1.2 and dist_mp > 0:
        narrative = ("Smart money is long (high PCR). Max Pain is ₹{} above spot — "
                     "institutions will defend ₹{} PE wall. "
                     "Liquidity hunts above ₹{} before potential reversal.").format(
                         int(abs(dist_mp)), int(pe_wall), int(ce_wall))
    elif pcr < 0.8 and dist_mp < 0:
        narrative = ("Smart money is short (low PCR). Max Pain is ₹{} below spot — "
                     "institutions will defend ₹{} CE wall. "
                     "Watch for stop-hunt below ₹{}.").format(
                         int(abs(dist_mp)), int(ce_wall), int(pe_wall))
    else:
        narrative = ("Balanced positioning. Market likely to oscillate between ₹{} support "
                     "and ₹{} resistance. Max Pain magnet at ₹{}.").format(
                         int(pe_wall), int(ce_wall), int(mp))

    return {
        "ce_wall":   ce_wall,
        "pe_wall":   pe_wall,
        "max_pain":  mp,
        "dist_ce":   dist_ce,
        "dist_pe":   dist_pe,
        "dist_mp":   dist_mp,
        "top3_ce":   top3_ce,
        "top3_pe":   top3_pe,
        "narrative": narrative,
        "pcr":       pcr,
    }


# ── 4. Van Tharp — R-Multiple & Position Sizing ──────────────────────────────

def van_tharp_sizing(underlying, premium_est, dte, symbol, capital=100000):
    """
    Van Tharp: risk no more than 1R per trade. Size position so 1R = 1-2% of capital.
    R = risk per unit = SL amount = premium × SL%.
    """
    lot_sizes  = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 40,
                  "MIDCPNIFTY": 50, "SENSEX": 20}
    lot        = lot_sizes.get(symbol.upper(), 75)

    # SL% by DTE
    sl_pct = 0.15 if dte == 0 else 0.20 if dte == 1 else 0.30 if dte <= 3 else 0.40

    risk_per_unit = premium_est * sl_pct
    risk_per_lot  = risk_per_unit * lot

    # 1R = 1.5% of capital
    one_r         = capital * 0.015
    max_lots      = max(1, int(one_r / risk_per_lot)) if risk_per_lot > 0 else 1
    cost_1lot     = premium_est * lot

    # Expectancy (assuming 55% win at 2R, 45% loss at 1R)
    win_pct, rr = 0.55, 2.0
    expectancy  = (win_pct * rr) - ((1 - win_pct) * 1.0)   # in R

    return {
        "lot_size":       lot,
        "premium_est":    round(premium_est, 1),
        "sl_pct":         round(sl_pct * 100, 0),
        "risk_per_lot":   round(risk_per_lot, 0),
        "one_r_capital":  round(one_r, 0),
        "max_lots":       max_lots,
        "cost_1lot":      round(cost_1lot, 0),
        "target_2r":      round(premium_est * (1 + sl_pct * rr), 1),
        "sl_price":       round(premium_est * (1 - sl_pct), 1),
        "expectancy_r":   round(expectancy, 2),
        "capital":        capital,
    }


# ── 5. Historical Expiry Pattern Learning ────────────────────────────────────

def learn_from_expiry_history(candles, expiry_dates_str=None):
    """
    Looks at the last N trading sessions' behaviour on their expiry-like patterns.
    Since we have daily candles, we approximate expiry days as Thursdays.
    Learns: what was RSI/EMA/VWAP setup → what happened (range, direction).
    Returns pattern summary + match score vs today.
    """
    if candles is None or candles.empty or len(candles) < 15:
        return {"patterns": [], "summary": "Not enough historical data."}

    c = candles.copy().sort_values("datetime").reset_index(drop=True)
    c["datetime"] = pd.to_datetime(c["datetime"])
    c["weekday"]  = c["datetime"].dt.weekday    # 3 = Thursday
    c["day_range_pct"] = (c["high"] - c["low"]) / c["open"] * 100
    c["close_chg_pct"] = c["close"].pct_change() * 100
    c["direction"]     = np.sign(c["close"] - c["open"])

    # Compute EMA9/20 and RSI14 on daily candles
    c["ema9"]  = c["close"].ewm(span=9,  adjust=False).mean()
    c["ema20"] = c["close"].ewm(span=20, adjust=False).mean()
    delta      = c["close"].diff()
    gain       = delta.clip(lower=0)
    loss       = (-delta).clip(lower=0)
    ag         = gain.ewm(com=13, min_periods=1).mean()
    al         = loss.ewm(com=13, min_periods=1).mean()
    c["rsi"]   = 100 - 100 / (1 + ag / (al + 1e-9))

    # Filter to expiry-like days (Thursdays) or last 5 sessions if no Thursdays
    thursdays = c[c["weekday"] == 3].tail(8)
    if len(thursdays) < 3:
        thursdays = c.tail(8)   # fallback: last 8 sessions

    patterns = []
    for _, row in thursdays.iterrows():
        rsi_v  = _safe(row.get("rsi", 50))
        ema_up = _safe(row.get("ema9", 0)) > _safe(row.get("ema20", 1))
        direction = int(row["direction"])
        outcome = "BULLISH" if direction > 0 else "BEARISH" if direction < 0 else "FLAT"

        patterns.append({
            "date":        row["datetime"].strftime("%d-%b-%Y"),
            "weekday":     row["datetime"].strftime("%A"),
            "rsi":         round(rsi_v, 1),
            "ema_trend":   "Up" if ema_up else "Down",
            "range_pct":   round(_safe(row.get("day_range_pct", 0)), 2),
            "close_chg":   round(_safe(row.get("close_chg_pct", 0)), 2),
            "outcome":     outcome,
            "direction":   direction,
        })

    if not patterns:
        return {"patterns": [], "summary": "No Thursday expiry data found."}

    # Stats
    bull_days = sum(1 for p in patterns if p["direction"] > 0)
    bear_days = sum(1 for p in patterns if p["direction"] < 0)
    avg_range = round(sum(p["range_pct"] for p in patterns) / len(patterns), 2)
    avg_close = round(sum(p["close_chg"] for p in patterns) / len(patterns), 2)

    bull_rsi   = [p["rsi"] for p in patterns if p["direction"] > 0]
    bear_rsi   = [p["rsi"] for p in patterns if p["direction"] < 0]
    avg_bull_r = round(sum(bull_rsi) / len(bull_rsi), 1) if bull_rsi else 50
    avg_bear_r = round(sum(bear_rsi) / len(bear_rsi), 1) if bear_rsi else 50

    summary = (
        "Last {} expiry sessions: {} bullish, {} bearish. "
        "Avg range: {:.1f}%. Avg close change: {:+.2f}%. "
        "Bull days had avg RSI {:.0f}; bear days avg RSI {:.0f}."
    ).format(len(patterns), bull_days, bear_days,
             avg_range, avg_close, avg_bull_r, avg_bear_r)

    # Today's setup match
    today = patterns[-1] if patterns else {}
    if today:
        if today["rsi"] < avg_bear_r and today["ema_trend"] == "Down":
            pattern_bias = "BEARISH (matches historical bear-day profile)"
        elif today["rsi"] > avg_bull_r and today["ema_trend"] == "Up":
            pattern_bias = "BULLISH (matches historical bull-day profile)"
        else:
            pattern_bias = "MIXED — no strong historical pattern match"
    else:
        pattern_bias = "Insufficient data"

    # What could have been traded on the last session (hindsight lesson)
    lesson = ""
    if len(patterns) >= 2:
        prev = patterns[-2]
        if prev["direction"] > 0 and prev["rsi"] > 55 and prev["ema_trend"] == "Up":
            lesson = ("On {}: RSI {:.0f} + EMA bullish → Bought CE → +{:.1f}% day move. "
                      "Setup: clear Bull.").format(prev["date"], prev["rsi"], abs(prev["close_chg"]))
        elif prev["direction"] < 0 and prev["rsi"] < 45 and prev["ema_trend"] == "Down":
            lesson = ("On {}: RSI {:.0f} + EMA bearish → Bought PE → {:.1f}% fall. "
                      "Setup: clear Bear.").format(prev["date"], prev["rsi"], abs(prev["close_chg"]))
        else:
            lesson = ("On {}: mixed signals (RSI {:.0f}, EMA {}) → {:.1f}% move. "
                      "Best trade: wait or small scalp.").format(
                          prev["date"], prev["rsi"], prev["ema_trend"], prev["close_chg"])

    return {
        "patterns":      patterns,
        "summary":       summary,
        "bull_days":     bull_days,
        "bear_days":     bear_days,
        "avg_range":     avg_range,
        "avg_close":     avg_close,
        "pattern_bias":  pattern_bias,
        "lesson":        lesson,
        "avg_bull_rsi":  avg_bull_r,
        "avg_bear_rsi":  avg_bear_r,
    }


# ── 6. Master Confidence Scorer ──────────────────────────────────────────────

def master_confidence(douglas, livermore, ict, history):
    """
    Combine all frameworks into one final confidence score + trade call.
    Returns (confidence_pct, direction, reasoning_lines).
    """
    bull_pts = bear_pts = 0
    reasoning = []

    # Mark Douglas confluence
    d_bull = douglas.get("bull", 0)
    d_bear = douglas.get("bear", 0)
    d_tot  = max(douglas.get("total", 1), 1)
    if d_bull > d_bear:
        bull_pts += d_bull * 10
        reasoning.append("Douglas: {}/{} indicators bullish → {}".format(
            d_bull, d_tot, douglas.get("verdict", "")))
    elif d_bear > d_bull:
        bear_pts += d_bear * 10
        reasoning.append("Douglas: {}/{} indicators bearish → {}".format(
            d_bear, d_tot, douglas.get("verdict", "")))

    # Livermore trend
    trend = livermore.get("trend", "SIDEWAYS")
    if trend == "UPTREND":
        bull_pts += 20
        reasoning.append("Livermore: {} — tape is bullish.".format(livermore.get("stage", "")))
    elif trend == "DOWNTREND":
        bear_pts += 20
        reasoning.append("Livermore: {} — tape is bearish.".format(livermore.get("stage", "")))
    else:
        reasoning.append("Livermore: {} — sit on hands.".format(livermore.get("stage", "Sideways")))

    # ICT smart money
    pcr = ict.get("pcr", 1.0)
    if pcr > 1.15:
        bull_pts += 15
        reasoning.append("ICT: PCR {:.2f} — institutions long, defend PE wall.".format(pcr))
    elif pcr < 0.85:
        bear_pts += 15
        reasoning.append("ICT: PCR {:.2f} — institutions short, defend CE wall.".format(pcr))

    # Historical pattern bias
    bias = history.get("pattern_bias", "")
    if "BULLISH" in bias:
        bull_pts += 15
        reasoning.append("History: " + bias)
    elif "BEARISH" in bias:
        bear_pts += 15
        reasoning.append("History: " + bias)
    else:
        reasoning.append("History: " + bias)

    total_pts = bull_pts + bear_pts
    if total_pts == 0:
        return 30, "WAIT", reasoning

    if bull_pts > bear_pts:
        raw_conf = min(97, 50 + int(bull_pts / max(total_pts, 1) * 50))
        direction = "BUY CE (CALL)"
    elif bear_pts > bull_pts:
        raw_conf = min(97, 50 + int(bear_pts / max(total_pts, 1) * 50))
        direction = "BUY PE (PUT)"
    else:
        raw_conf = 35
        direction = "WAIT / IRON CONDOR"

    return raw_conf, direction, reasoning

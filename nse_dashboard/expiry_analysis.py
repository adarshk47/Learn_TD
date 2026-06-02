"""
Expiry Day Analysis and Next-Expiry Prediction.

Methodology references:
  Augen J. (2009) — "Trading Options at Expiration"
    - Gamma spikes near ATM strike on expiry day
    - Max Pain magnet effect strongest in last 2 hours
    - Volume surge near ATM = directional conviction
    - Typical NIFTY scalp target on expiry: 30-80 points

  Murphy J.J. (1999) — "Technical Analysis of the Financial Markets"
    - OI concentration levels = key support/resistance
    - Volume + OI confirmation for breakout trades

  Natenberg S. (2015) — "Option Volatility and Pricing"
    - Gamma risk near expiry: ATM option doubles for every 0.5 move toward it
    - IV crush after expiry: buy next expiry before IV drops
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import numpy as np


# ── Expiry helpers ────────────────────────────────────────────────────────────

def hours_to_expiry(expiry_str: str) -> float:
    """Parse expiry string and return hours remaining (market hours = 6.25h/day)."""
    try:
        exp_dt = datetime.strptime(expiry_str, "%d-%b-%Y")
    except ValueError:
        try:
            exp_dt = datetime.strptime(expiry_str, "%d-%b-%y")
        except ValueError:
            return 99.0

    now      = datetime.now()
    market_close = exp_dt.replace(hour=15, minute=30, second=0)

    if now >= market_close:
        return 0.0

    # Count business hours remaining
    total_hours = 0.0
    curr = now
    while curr < market_close:
        if curr.weekday() < 5:  # weekday
            day_open  = curr.replace(hour=9, minute=15, second=0)
            day_close = curr.replace(hour=15, minute=30, second=0)
            if curr < day_open:
                curr = day_open
            seg_end = min(day_close, market_close)
            if curr < seg_end:
                total_hours += (seg_end - curr).total_seconds() / 3600
        curr = (curr + timedelta(days=1)).replace(hour=0, minute=0, second=0)

    return round(total_hours, 2)


def is_expiry_today(expiry_str: str) -> bool:
    try:
        exp_dt = datetime.strptime(expiry_str, "%d-%b-%Y")
    except ValueError:
        return False
    return exp_dt.date() == datetime.now().date()


def is_near_expiry(expiry_str: str, threshold_hours: float = 24.0) -> bool:
    return 0 < hours_to_expiry(expiry_str) <= threshold_hours


# ── Expected range calculation (Natenberg IV-based) ──────────────────────────

def expected_range(underlying: float, atm_iv: float, hours: float) -> dict:
    """
    Natenberg: expected ±1σ move for remaining session time.
    Returns: upper, lower, expected_pts, expected_pct
    """
    trading_year = 252 * 6.25   # hours in trading year
    iv_hourly    = (atm_iv / 100) / math.sqrt(trading_year)
    move_pct     = iv_hourly * math.sqrt(max(hours, 0.1)) * 100
    move_pts     = round(underlying * move_pct / 100, 0)
    return {
        "upper":        round(underlying + move_pts, 0),
        "lower":        round(underlying - move_pts, 0),
        "move_pts":     int(move_pts),
        "move_pct":     round(move_pct, 2),
        "hours_used":   hours,
    }


# ── Augen gamma zone analysis ────────────────────────────────────────────────

def augen_gamma_zones(df: pd.DataFrame, underlying: float,
                      hours_left: float) -> list[dict]:
    """
    Augen (2009): on expiry day, strikes within ±2 ATM are gamma hot zones.
    Higher OI at a strike = stronger magnet.
    Returns ranked strikes by their pull on the underlying.
    """
    if df.empty:
        return []

    atm_idx  = (df["strike"] - underlying).abs().idxmin()
    atm_val  = float(df.iloc[atm_idx]["strike"])

    zones = []
    for _, row in df.iterrows():
        strike = float(row["strike"])
        dist   = abs(strike - underlying)
        ce_oi  = float(row.get("ce_oi", 0))
        pe_oi  = float(row.get("pe_oi", 0))

        # Gamma decays exponentially with distance
        gamma_weight = math.exp(-0.5 * (dist / max(underlying * 0.005, 50)) ** 2)

        # Direction: CE OI above spot = resistance, PE OI below spot = support
        if strike > underlying:
            pull_type  = "RESISTANCE (CE wall)"
            pull_score = ce_oi * gamma_weight / 1e5
        else:
            pull_type  = "SUPPORT (PE wall)"
            pull_score = pe_oi * gamma_weight / 1e5

        if pull_score > 0.5:  # only include significant zones
            zones.append({
                "strike":     int(strike),
                "type":       pull_type,
                "ce_oi":      int(ce_oi),
                "pe_oi":      int(pe_oi),
                "pull_score": round(pull_score, 1),
                "dist_pts":   int(dist),
                "gamma_wt":   round(gamma_weight, 2),
            })

    return sorted(zones, key=lambda z: z["pull_score"], reverse=True)[:6]


# ── Expiry day scalping signals ───────────────────────────────────────────────

def expiry_scalp_signal(df: pd.DataFrame, meta: dict, sig: dict,
                        atm_iv: float = 15.0) -> dict:
    """
    Expiry day directional signal for 30-50 point scalps.

    Checks (based on Augen + Murphy):
    1. OI shift near ATM: PE writing > CE writing = bullish
    2. Max Pain vs spot: price below pain = upward pull
    3. ATM CE vs PE OI: which side has more open interest
    4. Net OI change direction (fresh positions)
    5. Volume acceleration near ATM
    """
    underlying = float(meta.get("underlying", 0))
    atm        = float(meta.get("atm", underlying))
    max_pain   = float(sig.get("max_pain", underlying))
    expiry     = meta.get("expiry", "")
    pcr        = float(sig.get("pcr", 1.0))

    h_left     = hours_to_expiry(expiry)
    exp_range  = expected_range(underlying, atm_iv, h_left)

    # ── OI near ATM (Augen: within 2 strikes) ─────────────────────────────
    score     = 0
    reasons   = []

    if not df.empty:
        atm_idx = (df["strike"] - underlying).abs().idxmin()
        near    = df.iloc[max(0, atm_idx - 2): min(len(df), atm_idx + 3)]

        # Indicator 1: OI change near ATM
        near_ce_chg = near["ce_chg_oi"].sum()
        near_pe_chg = near["pe_chg_oi"].sum()

        if near_pe_chg > near_ce_chg * 1.3:
            score += 30
            reasons.append("✅ PE writing {:+,.0f} > CE writing {:+,.0f} near ATM (bullish)".format(
                near_pe_chg, near_ce_chg))
        elif near_ce_chg > near_pe_chg * 1.3:
            score -= 30
            reasons.append("✅ CE writing {:+,.0f} > PE writing {:+,.0f} near ATM (bearish)".format(
                near_ce_chg, near_pe_chg))
        else:
            reasons.append("⚠️ OI change near ATM balanced — no clear direction")

        # Indicator 2: ATM CE vs PE OI
        atm_ce_oi = float(near.iloc[len(near)//2]["ce_oi"]) if len(near) > 0 else 0
        atm_pe_oi = float(near.iloc[len(near)//2]["pe_oi"]) if len(near) > 0 else 0
        if atm_pe_oi > atm_ce_oi * 1.2:
            score += 15
            reasons.append("✅ ATM PE OI {:,.0f} > CE OI {:,.0f} (put writers defending)".format(
                atm_pe_oi, atm_ce_oi))
        elif atm_ce_oi > atm_pe_oi * 1.2:
            score -= 15
            reasons.append("✅ ATM CE OI {:,.0f} > PE OI {:,.0f} (call writers capping)".format(
                atm_ce_oi, atm_pe_oi))

    # Indicator 3: Max Pain pull (Augen: strongest in last 2h)
    pain_gap    = max_pain - underlying
    pain_gap_pct = abs(pain_gap) / max(underlying, 1) * 100
    if abs(pain_gap) > underlying * 0.002:  # > 0.2% away
        if pain_gap > 0:
            score += 20
            reasons.append("✅ Max Pain ₹{:,.0f} is ₹{:,.0f} above spot — upward pull".format(
                int(max_pain), int(pain_gap)))
        else:
            score -= 20
            reasons.append("✅ Max Pain ₹{:,.0f} is ₹{:,.0f} below spot — downward pull".format(
                int(max_pain), int(abs(pain_gap))))

    # Indicator 4: PCR
    if pcr > 1.3:
        score += 15
        reasons.append("✅ PCR={:.2f} — strong put writing, institutions defending lows".format(pcr))
    elif pcr < 0.7:
        score -= 15
        reasons.append("✅ PCR={:.2f} — call writing dominant, capping upside".format(pcr))

    # ── Final signal ───────────────────────────────────────────────────────
    if score > 25:
        direction, sig_color, action = "BUY CALL", "green", "BUY ATM CE"
    elif score < -25:
        direction, sig_color, action = "BUY PUT", "red", "BUY ATM PE"
    else:
        direction, sig_color, action = "WAIT", "orange", "AVOID — no clear edge"

    # Target calculation (Augen: 30-80 pts on NIFTY per scalp)
    lot_map  = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 40, "MIDCPNIFTY": 50}
    sym      = meta.get("symbol", "NIFTY")
    lot_size = lot_map.get(sym, 75)
    pts_per_lot_target = 40 if underlying > 10000 else 20  # NIFTY ~40pts, BANKNIFTY ~80pts

    atm_premium_est = underlying * (atm_iv / 100) * math.sqrt(h_left / (252 * 6.25)) if atm_iv > 0 else underlying * 0.003
    sl_pct          = 0.30  # 30% of premium SL on expiry day (gamma moves fast)
    tgt_pct         = 0.50  # 50% target

    return {
        "direction":        direction,
        "action":           action,
        "signal_color":     sig_color,
        "score":            score,
        "reasons":          reasons,
        "hours_left":       h_left,
        "expected_range":   exp_range,
        "atm_premium_est":  round(atm_premium_est, 1),
        "sl_pct":           sl_pct,
        "tgt_pct":          tgt_pct,
        "pts_target":       pts_per_lot_target,
        "lot_size":         lot_size,
        "max_pain":         max_pain,
        "pain_gap":         pain_gap,
        "pcr":              pcr,
    }


# ── Next expiry prediction ─────────────────────────────────────────────────────

def next_expiry_info(all_expiries: list, current_expiry: str,
                     underlying: float, atm_iv: float) -> dict:
    """
    When current expiry < 24h away, analyze the next expiry.
    Returns expected range, strategy recommendation, and key levels for next expiry.
    """
    if len(all_expiries) < 2:
        return {"error": "No next expiry available"}

    try:
        idx       = all_expiries.index(current_expiry)
        next_exp  = all_expiries[idx + 1]
    except (ValueError, IndexError):
        return {"error": "Cannot determine next expiry"}

    # Parse next expiry date
    try:
        next_dt   = datetime.strptime(next_exp, "%d-%b-%Y")
    except ValueError:
        return {"error": "Cannot parse next expiry: " + next_exp}

    days_to_next = (next_dt.date() - datetime.now().date()).days
    hours_next   = max(days_to_next * 6.25, 1.0)

    rng_next = expected_range(underlying, atm_iv * 1.15, hours_next)  # IV slightly higher for next expiry

    # Strategy recommendation based on Natenberg
    if atm_iv < 15:
        strategy     = "BUY ATM Straddle"
        strategy_why = "Low IV (< 15%) — cheap to buy both sides for next expiry move"
    elif atm_iv > 25:
        strategy     = "Sell Iron Condor"
        strategy_why = "High IV (> 25%) — sell premium, collect theta over {} days".format(days_to_next)
    else:
        strategy     = "Bull Put Spread / Bear Call Spread"
        strategy_why = "Moderate IV — defined risk spread suits {} day hold".format(days_to_next)

    return {
        "next_expiry":    next_exp,
        "days_to_next":   days_to_next,
        "hours_to_next":  hours_next,
        "expected_range": rng_next,
        "strategy":       strategy,
        "strategy_why":   strategy_why,
        "atm_iv":         atm_iv,
        "iv_for_next":    round(atm_iv * 1.15, 1),
    }

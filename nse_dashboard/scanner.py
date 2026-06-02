"""
NSE Market Scanner — Multi-symbol trending analysis.

Methodology references:
  Murphy J.J. (1999) — "Technical Analysis of the Financial Markets"
    Chapter 7: Volume and Open Interest — 4-scenario OI+Price analysis:
      Rising price + Rising OI  → Bullish  (fresh longs)
      Rising price + Falling OI → Weakening (short covering)
      Falling price + Rising OI → Bearish  (fresh shorts)
      Falling price + Falling OI → Recovering (longs liquidating)

  Natenberg S. (2015) — "Option Volatility and Pricing"
    IV Rank = (current_iv - 52w_low) / (52w_high - 52w_low) × 100
    IV Rank < 30 → cheap premiums → prefer buying options
    IV Rank > 70 → expensive premiums → prefer selling strategies

  McMillan L.G. (2012) — "Options as a Strategic Investment"
    PCR > 1.2 → institutional put writing → bullish sentiment
    PCR < 0.7 → call writing dominates → bearish institutional view
    ATM OI concentration → max-pain magnet effect
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# ── Instruments to scan ───────────────────────────────────────────────────────

SCAN_INDICES = [
    ("NIFTY",      True,  50),
    ("BANKNIFTY",  True,  100),
    ("FINNIFTY",   True,  50),
    ("MIDCPNIFTY", True,  25),
]

SCAN_STOCKS = [
    ("RELIANCE",   False, 50),
    ("HDFCBANK",   False, 20),
    ("ICICIBANK",  False, 10),
    ("SBIN",       False, 10),
    ("INFY",       False, 20),
    ("TCS",        False, 50),
    ("BAJFINANCE", False, 50),
    ("AXISBANK",   False, 10),
    ("WIPRO",      False, 10),
    ("ONGC",       False, 5),
]

# Approximate base prices for demo generation
_BASE_PRICES = {
    "NIFTY":      24350, "BANKNIFTY":  52400, "FINNIFTY":  23800, "MIDCPNIFTY": 12200,
    "RELIANCE":    2940, "HDFCBANK":    1820, "ICICIBANK":  1310, "SBIN":        820,
    "INFY":        1660, "TCS":         3980, "BAJFINANCE": 7200, "AXISBANK":   1240,
    "WIPRO":        580, "ONGC":         270,
}

_IV_RANGES = {  # approximate 52-week IV ranges
    "NIFTY":      (10, 25), "BANKNIFTY":  (13, 30), "FINNIFTY":  (12, 28), "MIDCPNIFTY": (12, 26),
    "RELIANCE":   (18, 40), "HDFCBANK":   (16, 38), "ICICIBANK": (18, 42), "SBIN":        (22, 48),
    "INFY":       (18, 38), "TCS":        (16, 36), "BAJFINANCE":(20, 45), "AXISBANK":   (20, 42),
    "WIPRO":      (20, 40), "ONGC":       (22, 46),
}


# ── Murphy OI+Price 4-scenario analysis ──────────────────────────────────────

def murphy_oi_signal(price_chg_pct: float, oi_chg_pct: float) -> tuple[str, str]:
    """
    Murphy (1999) four-scenario OI + Price analysis.
    Returns (signal, explanation).
    """
    price_up = price_chg_pct > 0.1
    price_dn = price_chg_pct < -0.1
    oi_up    = oi_chg_pct > 0.5
    oi_dn    = oi_chg_pct < -0.5

    if price_up and oi_up:
        return "BULLISH", "Rising price + Rising OI → fresh longs added (Murphy: strong bull)"
    elif price_up and oi_dn:
        return "WEAKENING", "Rising price + Falling OI → short covering, not genuine buying"
    elif price_dn and oi_up:
        return "BEARISH", "Falling price + Rising OI → fresh shorts added (Murphy: strong bear)"
    elif price_dn and oi_dn:
        return "RECOVERING", "Falling price + Falling OI → longs liquidating, selling easing"
    else:
        return "NEUTRAL", "Price and OI both near flat — no clear directional signal"


# ── Natenberg IV Rank ─────────────────────────────────────────────────────────

def natenberg_iv_rank(symbol: str, current_iv: float) -> tuple[float, str]:
    """
    Natenberg IV Rank: where is current IV relative to its 52-week range?
    Returns (iv_rank_pct, recommendation).
    """
    lo, hi = _IV_RANGES.get(symbol, (15, 35))
    if hi == lo:
        return 50.0, "Moderate"
    rank = max(0.0, min(100.0, (current_iv - lo) / (hi - lo) * 100))
    if rank < 30:
        note = "Low IV (rank {}%) — cheap premiums → prefer BUYING options".format(int(rank))
    elif rank > 70:
        note = "High IV (rank {}%) — expensive → prefer SELLING strategies".format(int(rank))
    else:
        note = "Moderate IV (rank {}%) — fair value".format(int(rank))
    return round(rank, 1), note


# ── PCR conviction (McMillan) ─────────────────────────────────────────────────

def mcmillan_pcr_signal(pcr: float) -> tuple[str, int]:
    """
    McMillan PCR interpretation with conviction score (-100 to +100).
    Positive = bullish conviction.
    """
    if pcr > 1.8:
        return "STRONGLY BULLISH", 90
    elif pcr > 1.4:
        return "BULLISH", 70
    elif pcr > 1.1:
        return "MILD BULLISH", 40
    elif pcr > 0.85:
        return "NEUTRAL", 0
    elif pcr > 0.65:
        return "MILD BEARISH", -40
    elif pcr > 0.5:
        return "BEARISH", -70
    else:
        return "STRONGLY BEARISH", -90


# ── Demo scan data generator ──────────────────────────────────────────────────

def _demo_scan_result(symbol: str, is_index: bool) -> dict:
    seed = int(datetime.now().strftime("%Y%m%d%H")) + hash(symbol) % 10000
    rng  = random.Random(seed)
    np_rng = np.random.default_rng(seed % (2**31))

    base_price = _BASE_PRICES.get(symbol, 1000)
    price      = base_price + rng.randint(-int(base_price * 0.02), int(base_price * 0.02))
    price_chg  = round(rng.uniform(-1.5, 1.5), 2)  # % change today

    # OI data
    total_ce_oi  = int(np_rng.integers(800000, 5000000))
    total_pe_oi  = int(np_rng.integers(800000, 5000000))
    pcr          = round(total_pe_oi / max(total_ce_oi, 1), 2)

    ce_oi_chg_pct = round(rng.uniform(-8, 12), 1)
    pe_oi_chg_pct = round(rng.uniform(-5, 15), 1)

    # Net OI bias (positive = PE writing dominant = bullish)
    net_oi_bias = round((pe_oi_chg_pct - ce_oi_chg_pct) / max(abs(ce_oi_chg_pct) + abs(pe_oi_chg_pct), 1) * 100, 1)

    # IV
    lo, hi  = _IV_RANGES.get(symbol, (15, 35))
    atm_iv  = round(rng.uniform(lo * 0.9, hi * 0.8), 1)
    iv_rank, iv_note = natenberg_iv_rank(symbol, atm_iv)

    # Murphy signal (based on price change and OI direction)
    combined_oi_chg = pe_oi_chg_pct - ce_oi_chg_pct  # net OI bias for price direction
    murphy_sig, murphy_note = murphy_oi_signal(price_chg, combined_oi_chg)

    # McMillan PCR
    pcr_label, pcr_conv = mcmillan_pcr_signal(pcr)

    # Volume (relative to 5-day avg)
    vol_ratio = round(rng.uniform(0.5, 3.0), 2)

    # Composite conviction (-100 to +100, positive = bullish)
    conviction = round(pcr_conv * 0.5 + net_oi_bias * 0.3 + price_chg * 10 * 0.2, 1)
    conviction = max(-100, min(100, conviction))

    # Final signal
    if conviction > 30:
        signal, sig_color = "BUY CALL", "green"
    elif conviction < -30:
        signal, sig_color = "BUY PUT", "red"
    else:
        signal, sig_color = "WATCH", "orange"

    return {
        "symbol":         symbol,
        "type":           "Index" if is_index else "Stock",
        "price":          price,
        "price_chg":      price_chg,
        "pcr":            pcr,
        "pcr_label":      pcr_label,
        "ce_oi_chg":      ce_oi_chg_pct,
        "pe_oi_chg":      pe_oi_chg_pct,
        "net_oi_bias":    net_oi_bias,
        "murphy_signal":  murphy_sig,
        "murphy_note":    murphy_note,
        "atm_iv":         atm_iv,
        "iv_rank":        iv_rank,
        "iv_note":        iv_note,
        "vol_ratio":      vol_ratio,
        "conviction":     conviction,
        "signal":         signal,
        "signal_color":   sig_color,
    }


# ── Live scan (from Angel One data) ──────────────────────────────────────────

def _live_scan_result(symbol: str, is_index: bool, df: pd.DataFrame,
                      meta: dict, sig: dict) -> dict:
    """Build a scan result dict from already-fetched option chain data."""
    underlying = float(meta.get("underlying", 0))
    pcr        = float(sig.get("pcr", 1.0))
    atm_iv     = 0.0
    if not df.empty and "ce_iv" in df.columns:
        try:
            idx    = (df["strike"] - underlying).abs().idxmin()
            atm_iv = float(df.iloc[idx]["ce_iv"])
        except Exception:
            pass

    total_ce_chg = float(df["ce_chg_oi"].sum()) if not df.empty else 0
    total_pe_chg = float(df["pe_chg_oi"].sum()) if not df.empty else 0
    total_ce_oi  = float(df["ce_oi"].sum())      if not df.empty else 1
    total_pe_oi  = float(df["pe_oi"].sum())      if not df.empty else 1

    ce_oi_chg_pct = round(total_ce_chg / max(total_ce_oi, 1) * 100, 1)
    pe_oi_chg_pct = round(total_pe_chg / max(total_pe_oi, 1) * 100, 1)
    net_oi_bias   = round((pe_oi_chg_pct - ce_oi_chg_pct) /
                           max(abs(pe_oi_chg_pct) + abs(ce_oi_chg_pct), 0.1) * 100, 1)

    price_chg = 0.0  # not available without historical data

    murphy_sig, murphy_note = murphy_oi_signal(float(sig.get("score", 0)) / 10, pe_oi_chg_pct - ce_oi_chg_pct)
    iv_rank, iv_note        = natenberg_iv_rank(symbol, atm_iv if atm_iv > 0 else 15.0)
    pcr_label, pcr_conv     = mcmillan_pcr_signal(pcr)

    conviction = round(pcr_conv * 0.5 + net_oi_bias * 0.3, 1)
    conviction = max(-100, min(100, conviction))

    signal_str = sig.get("signal", "AVOID")
    if signal_str == "BUY CALL":
        sig_color = "green"
    elif signal_str == "BUY PUT":
        sig_color = "red"
    else:
        sig_color  = "orange"
        signal_str = "WATCH"

    return {
        "symbol":         symbol,
        "type":           "Index" if is_index else "Stock",
        "price":          underlying,
        "price_chg":      price_chg,
        "pcr":            pcr,
        "pcr_label":      pcr_label,
        "ce_oi_chg":      ce_oi_chg_pct,
        "pe_oi_chg":      pe_oi_chg_pct,
        "net_oi_bias":    net_oi_bias,
        "murphy_signal":  murphy_sig,
        "murphy_note":    murphy_note,
        "atm_iv":         atm_iv,
        "iv_rank":        iv_rank,
        "iv_note":        iv_note,
        "vol_ratio":      1.0,
        "conviction":     conviction,
        "signal":         signal_str,
        "signal_color":   sig_color,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def scan_demo(include_stocks: bool = False) -> list[dict]:
    """Full demo scan of indices (+ optionally stocks)."""
    results = [_demo_scan_result(sym, idx) for sym, idx, _ in SCAN_INDICES]
    if include_stocks:
        results += [_demo_scan_result(sym, idx) for sym, idx, _ in SCAN_STOCKS]
    return sorted(results, key=lambda x: abs(x["conviction"]), reverse=True)


def scan_live(current_symbol: str, current_df: pd.DataFrame,
              current_meta: dict, current_sig: dict,
              include_stocks: bool = False) -> list[dict]:
    """
    Use live data for the currently-selected instrument; demo for rest.
    (Fetching all symbols would be slow in Streamlit.)
    """
    results = []
    for sym, is_idx, _ in SCAN_INDICES:
        if sym == current_symbol:
            r = _live_scan_result(sym, is_idx, current_df, current_meta, current_sig)
        else:
            r = _demo_scan_result(sym, is_idx)
        results.append(r)

    if include_stocks:
        for sym, is_idx, _ in SCAN_STOCKS:
            results.append(_demo_scan_result(sym, is_idx))

    return sorted(results, key=lambda x: abs(x["conviction"]), reverse=True)


# ── OI Velocity tracker ───────────────────────────────────────────────────────

def compute_oi_velocity(df_now: pd.DataFrame, df_prev: pd.DataFrame | None,
                        seconds_elapsed: float = 60) -> pd.DataFrame:
    """
    OI velocity = (current OI - previous OI) per minute.
    Returns df with added velocity columns: ce_oi_vel, pe_oi_vel, net_oi_vel.
    """
    df = df_now.copy()
    if df_prev is None or df_prev.empty or "ce_oi" not in df_prev.columns:
        df["ce_oi_vel"] = 0.0
        df["pe_oi_vel"] = 0.0
        df["net_oi_vel"] = 0.0
        return df

    minutes = max(seconds_elapsed / 60, 0.5)
    try:
        merged = df.merge(
            df_prev[["strike", "ce_oi", "pe_oi"]].rename(
                columns={"ce_oi": "ce_oi_prev", "pe_oi": "pe_oi_prev"}
            ),
            on="strike", how="left"
        )
        df["ce_oi_vel"]  = ((merged["ce_oi"]  - merged["ce_oi_prev"])  / minutes).fillna(0)
        df["pe_oi_vel"]  = ((merged["pe_oi"]  - merged["pe_oi_prev"])  / minutes).fillna(0)
        df["net_oi_vel"] = df["pe_oi_vel"] - df["ce_oi_vel"]
    except Exception:
        df["ce_oi_vel"] = df["pe_oi_vel"] = df["net_oi_vel"] = 0.0

    return df

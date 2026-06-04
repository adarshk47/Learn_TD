"""
oi_tracker.py — Persistent OI Snapshot Engine

Saves a timestamped snapshot of key OI metrics every time the option chain
is fetched, and computes changes over 5-min / 15-min / 30-min / 1-hr / 2-hr
/ 3-hr / 4-hr / 5-hr windows.

Snapshots survive browser refreshes by writing to a JSON file on disk.
Only the last 6 hours of data is kept (auto-pruned on each save).

Plain-language interpretations (in English):
  CE OI Chg + → More call writing → Bears building resistance → Bearish
  CE OI Chg − → Calls being squared off → Resistance weakening → Bullish
  PE OI Chg + → More put writing → Bulls building support → Bullish
  PE OI Chg − → Puts being squared off → Support weakening → Bearish
  Net Flow   + → PE building faster than CE → Bullish overall
  Net Flow   − → CE building faster than PE → Bearish overall
  PCR rising  → More puts vs calls → Bullish
  PCR falling → More calls vs puts → Bearish
"""

import json
import os
from datetime import datetime, timedelta, timezone

_SNAP_FILE = os.path.join(os.path.dirname(__file__), ".oi_snapshots.json")

# Indian Standard Time = UTC+5:30
_IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    """Return current datetime in IST (naive, for display and comparison)."""
    return datetime.now(_IST).replace(tzinfo=None)


def _ist_str(dt: datetime) -> str:
    """Format a datetime as IST HH:MM string."""
    return dt.strftime("%H:%M IST")

# Time windows to display (label, seconds_back)
WINDOWS = [
    ("5 min",  5  * 60),
    ("15 min", 15 * 60),
    ("30 min", 30 * 60),
    ("1 hr",   60 * 60),
    ("2 hr",   120 * 60),
    ("3 hr",   180 * 60),
    ("4 hr",   240 * 60),
    ("5 hr",   300 * 60),
]

MAX_AGE_HOURS = 6


def _load() -> list:
    try:
        if os.path.exists(_SNAP_FILE):
            with open(_SNAP_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save(snaps: list) -> None:
    try:
        with open(_SNAP_FILE, "w") as f:
            json.dump(snaps, f)
    except Exception:
        pass


def _prune(snaps: list) -> list:
    cutoff = (_now_ist() - timedelta(hours=MAX_AGE_HOURS)).isoformat()
    return [s for s in snaps if s.get("ts", "") >= cutoff]


def record_snapshot(df, underlying: float, pcr: float, call_sum_atm: float,
                    put_sum_atm: float, symbol: str) -> None:
    """
    Save a snapshot of current OI state.
    Call this once per option-chain fetch.

    Parameters
    ----------
    df           : option chain DataFrame (columns: ce_oi, pe_oi, ce_chg_oi, pe_chg_oi, strike)
    underlying   : spot price
    pcr          : put/call ratio
    call_sum_atm : CE OI sum at ATM±1 (in units, not thousands)
    put_sum_atm  : PE OI sum at ATM±1
    symbol       : e.g. 'NIFTY'
    """
    if df is None or df.empty:
        return

    ce_total = float(df["ce_oi"].sum())
    pe_total = float(df["pe_oi"].sum())

    snap = {
        "ts":           _now_ist().isoformat(timespec="seconds"),
        "symbol":       symbol,
        "underlying":   round(underlying, 2),
        "pcr":          round(pcr, 3),
        "ce_oi_total":  round(ce_total, 0),
        "pe_oi_total":  round(pe_total, 0),
        "call_sum_atm": round(call_sum_atm, 0),
        "put_sum_atm":  round(put_sum_atm, 0),
        "net_oi_total": round(pe_total - ce_total, 0),
    }

    snaps = _prune(_load())
    snaps.append(snap)
    _save(snaps)


def get_trend_table(symbol: str, current: dict) -> list[dict]:
    """
    Returns a list of rows, one per time window, each containing:
      window, ce_chg, pe_chg, net_flow, pcr_chg, trend, interpretation,
      snap_time, age_str, has_data
    """
    snaps = _prune(_load())
    # Only this symbol
    snaps = [s for s in snaps if s.get("symbol", "") == symbol]

    now  = _now_ist()
    rows = []

    # All past snapshots (exclude snaps taken in the future, i.e. > now+5s)
    past_snaps = [s for s in snaps if datetime.fromisoformat(s["ts"]) <= now + timedelta(seconds=5)]

    # Earliest snapshot time — tells the user when tracking began
    first_ts = datetime.fromisoformat(past_snaps[0]["ts"]) if past_snaps else None
    age_of_oldest = (now - first_ts).total_seconds() if first_ts else 0

    for label, secs_back in WINDOWS:
        target_dt = now - timedelta(seconds=secs_back)

        if not past_snaps:
            rows.append({
                "window": label, "has_data": False,
                "snap_time": "--",
                "age_str": "No snapshots yet — refresh the page once",
                "ce_chg": None, "pe_chg": None, "net_flow": None,
                "pcr_chg": None, "ce_chg_pct": None, "pe_chg_pct": None,
                "trend": "Wait",
                "interpretation": "First refresh saves the first snapshot. Come back after {}."
                                  .format(label),
                "spot_then": None, "spot_chg": None,
            })
            continue

        # Do we have any snapshot old enough for this window?
        if age_of_oldest < secs_back - 30:
            # Not enough history yet — tell user when data will be ready
            need_more_secs = int(secs_back - age_of_oldest)
            need_more_min  = need_more_secs // 60
            ready_at_ist   = now + timedelta(seconds=need_more_secs)
            rows.append({
                "window": label, "has_data": False,
                "snap_time": _ist_str(first_ts) if first_ts else "--",
                "age_str": "Tracking since {} IST ({:.0f} min ago)".format(
                    first_ts.strftime("%H:%M"), age_of_oldest / 60) if first_ts else "--",
                "ce_chg": None, "pe_chg": None, "net_flow": None,
                "pcr_chg": None, "ce_chg_pct": None, "pe_chg_pct": None,
                "trend": "Collecting",
                "interpretation": "Data ready at {} IST (in ~{} min)".format(
                    _ist_str(ready_at_ist), need_more_min),
                "spot_then": None, "spot_chg": None,
            })
            continue

        # Find snapshot closest to target_dt (from the past only)
        past_candidates = [s for s in past_snaps
                           if datetime.fromisoformat(s["ts"]) <= now]
        snap    = min(past_candidates,
                      key=lambda s: abs((datetime.fromisoformat(s["ts"]) - target_dt).total_seconds()))
        snap_dt = datetime.fromisoformat(snap["ts"])
        age_secs = (now - snap_dt).total_seconds()

        # Deltas (current − snapshot)
        ce_now  = current.get("ce_oi_total", 0)
        pe_now  = current.get("pe_oi_total", 0)
        ce_then = snap.get("ce_oi_total", ce_now)
        pe_then = snap.get("pe_oi_total", pe_now)
        pcr_then = snap.get("pcr", current.get("pcr", 1.0))

        ce_chg      = ce_now - ce_then
        pe_chg      = pe_now - pe_then
        net_flow    = pe_chg - ce_chg          # positive = bullish
        pcr_chg     = current.get("pcr", 1.0) - pcr_then
        spot_chg    = current.get("underlying", 0) - snap.get("underlying", current.get("underlying", 0))

        ce_chg_pct  = ce_chg / max(ce_then, 1) * 100
        pe_chg_pct  = pe_chg / max(pe_then, 1) * 100

        # ATM-only deltas
        call_sum_now  = current.get("call_sum_atm", 0)
        put_sum_now   = current.get("put_sum_atm",  0)
        call_sum_then = snap.get("call_sum_atm", call_sum_now)
        put_sum_then  = snap.get("put_sum_atm",  put_sum_now)
        call_sum_chg  = call_sum_now - call_sum_then
        put_sum_chg   = put_sum_now  - put_sum_then

        # Trend score
        bull = (pe_chg > 0) + (ce_chg < 0) + (pcr_chg > 0) + (net_flow > 0) + (spot_chg > 0)
        bear = (ce_chg > 0) + (pe_chg < 0) + (pcr_chg < 0) + (net_flow < 0) + (spot_chg < 0)

        if bull >= 4:
            trend = "BULLISH 🟢"
        elif bear >= 4:
            trend = "BEARISH 🔴"
        elif bull == bear:
            trend = "NEUTRAL ⚪"
        elif bull > bear:
            trend = "MILD BULL ↑"
        else:
            trend = "MILD BEAR ↓"

        # Plain-language interpretation
        parts = []
        if ce_chg > 5000:
            parts.append("Calls being written ↑ (resistance building, bearish pressure)")
        elif ce_chg < -5000:
            parts.append("Calls squaring off ↓ (resistance weakening, bullish relief)")

        if pe_chg > 5000:
            parts.append("Puts being written ↑ (support building, bulls defending)")
        elif pe_chg < -5000:
            parts.append("Puts squaring off ↓ (support weakening, watch for fall)")

        if net_flow > 10000:
            parts.append("Net: PE > CE build → Bullish flow")
        elif net_flow < -10000:
            parts.append("Net: CE > PE build → Bearish flow")

        if pcr_chg > 0.05:
            parts.append("PCR rising → getting more bullish")
        elif pcr_chg < -0.05:
            parts.append("PCR falling → getting more bearish")

        if not parts:
            parts.append("OI changes small — wait for stronger signal")

        interpretation = " | ".join(parts)
        snap_age_min = age_secs / 60
        age_str = "snapshot @ {} IST ({:.0f} min ago)".format(
            snap_dt.strftime("%H:%M"), snap_age_min)

        rows.append({
            "window":         label,
            "has_data":       True,
            "snap_time":      snap_dt.strftime("%H:%M IST"),
            "age_str":        age_str,
            "ce_chg":         int(ce_chg),
            "pe_chg":         int(pe_chg),
            "net_flow":       int(net_flow),
            "pcr_chg":        round(pcr_chg, 3),
            "ce_chg_pct":     round(ce_chg_pct, 2),
            "pe_chg_pct":     round(pe_chg_pct, 2),
            "call_sum_chg":   int(call_sum_chg),
            "put_sum_chg":    int(put_sum_chg),
            "spot_then":      snap.get("underlying"),
            "spot_chg":       round(spot_chg, 2),
            "trend":          trend,
            "interpretation": interpretation,
        })

    return rows


def first_snapshot_time(symbol: str) -> str | None:
    """Return 'HH:MM IST' of earliest snapshot for this symbol, or None."""
    snaps = _prune(_load())
    mine  = [s for s in snaps if s.get("symbol", "") == symbol]
    if not mine:
        return None
    earliest_ts = min(s["ts"] for s in mine)
    try:
        dt = datetime.fromisoformat(earliest_ts)
        return dt.strftime("%H:%M IST")
    except Exception:
        return earliest_ts

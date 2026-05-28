import pandas as pd
import time
import requests

_HEADERS = {
    "user-agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "accept":          "*/*",
    "accept-language": "en,gu;q=0.9,hi;q=0.8",
    "accept-encoding": "gzip, deflate, br",
    "referer":         "https://www.nseindia.com/option-chain",
}

_session      = None
_session_time = 0


def _build_session():
    s = requests.Session()
    s.get("https://www.nseindia.com",              headers=_HEADERS, timeout=10)
    time.sleep(2)
    s.get("https://www.nseindia.com/option-chain", headers=_HEADERS, timeout=10)
    time.sleep(1)
    return s


def _reset_session():
    global _session, _session_time
    _session      = _build_session()
    _session_time = time.time()


def fetch_option_chain(symbol, is_index=True):
    global _session, _session_time
    if _session is None or (time.time() - _session_time) > 300:
        _reset_session()

    if is_index:
        url = "https://www.nseindia.com/api/option-chain-indices?symbol={}".format(symbol)
    else:
        url = "https://www.nseindia.com/api/option-chain-equities?symbol={}".format(symbol)

    for attempt in range(2):
        try:
            r = _session.get(url, headers=_HEADERS, timeout=15)
            if r.status_code in (401, 403, 404):
                _reset_session()
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 0:
                _reset_session()
            else:
                return {"error": str(e)}
    return {"error": "NSE fetch failed after retry"}


def parse_option_chain(data, num_strikes=20):
    if not data:
        return pd.DataFrame(), {"error": "Empty response from NSE"}
    if "error" in data:
        return pd.DataFrame(), {"error": data["error"]}

    try:
        rec_block      = data.get("records", {})
        all_records    = rec_block.get("data", [])
        expiry_dates   = rec_block.get("expiryDates", [])
        underlying_val = rec_block.get("underlyingValue", 0)

        if not all_records:
            return pd.DataFrame(), {"error": "NSE returned no records"}

        seen = []
        for r in all_records:
            e = r.get("expiryDate", "")
            if e and e not in seen:
                seen.append(e)
        target_expiry = seen[0] if seen else (expiry_dates[0] if expiry_dates else "")

        rows = []
        for rec in all_records:
            if rec.get("expiryDate", "") != target_expiry:
                continue
            strike = rec.get("strikePrice") or rec.get("strike")
            if not strike:
                continue
            ce = rec.get("CE", {})
            pe = rec.get("PE", {})
            rows.append({
                "strike":    float(strike),
                "ce_oi":     ce.get("openInterest", 0),
                "ce_chg_oi": ce.get("changeinOpenInterest", 0),
                "ce_volume": ce.get("totalTradedVolume", 0),
                "ce_iv":     ce.get("impliedVolatility", 0),
                "ce_ltp":    ce.get("lastPrice", 0),
                "ce_bid":    ce.get("bidPrice", 0),
                "ce_ask":    ce.get("askPrice", 0),
                "pe_oi":     pe.get("openInterest", 0),
                "pe_chg_oi": pe.get("changeinOpenInterest", 0),
                "pe_volume": pe.get("totalTradedVolume", 0),
                "pe_iv":     pe.get("impliedVolatility", 0),
                "pe_ltp":    pe.get("lastPrice", 0),
                "pe_bid":    pe.get("bidPrice", 0),
                "pe_ask":    pe.get("askPrice", 0),
            })

        if not rows:
            return pd.DataFrame(), {"error": "No strikes found for expiry: {}".format(target_expiry)}

        df      = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
        atm_idx = (df["strike"] - underlying_val).abs().idxmin()
        half    = num_strikes // 2
        df      = df.iloc[max(0, atm_idx - half): min(len(df), atm_idx + half)].reset_index(drop=True)
        atm     = float(df.iloc[(df["strike"] - underlying_val).abs().idxmin()]["strike"])

        return df, {
            "underlying":   underlying_val,
            "expiry":       target_expiry,
            "all_expiries": expiry_dates or seen,
            "atm":          atm,
            "source":       "NSE Direct",
        }

    except Exception as e:
        return pd.DataFrame(), {"error": str(e)}


def calculate_pcr(df):
    ce = df["ce_oi"].sum()
    pe = df["pe_oi"].sum()
    return round(pe / ce, 2) if ce else 0.0


def calculate_max_pain(df):
    strikes  = df["strike"].tolist()
    best     = strikes[0]
    min_pain = float("inf")
    for s in strikes:
        pain = sum(
            row["ce_oi"] * max(0.0, s - row["strike"]) +
            row["pe_oi"] * max(0.0, row["strike"] - s)
            for _, row in df.iterrows()
        )
        if pain < min_pain:
            min_pain = pain
            best     = s
    return best


def _strategy_for_conviction(score, confidence):
    """
    Map conviction level to recommended options strategy.
    Based on JUDAH-Nifty-Oracle research (83.4% iron condor, 79.2% spread, 74.8% naked).
    """
    abs_score = abs(score)
    if abs_score >= 45 and confidence >= 70:
        return "ATM Option Buy", "High conviction — buy ATM CE/PE outright (74.8% hist win rate)"
    elif abs_score >= 30 and confidence >= 60:
        return "Credit Spread", "Moderate conviction — Bull Put Spread / Bear Call Spread (79.2% hist win rate)"
    elif abs_score >= 15:
        return "Debit Spread", "Low-moderate conviction — cheaper spread; defined risk"
    else:
        return "Iron Condor", "Neutral market — sell both sides for theta decay (83.4% hist win rate)"


def generate_signal(df, meta):
    empty = {
        "signal": "NO DATA", "confidence": 0, "reasons": [],
        "color": "orange", "pcr": 0, "max_pain": 0,
        "max_ce_resistance": 0, "max_pe_support": 0, "score": 0,
        "call_sum": 0, "put_sum": 0, "oi_difference": 0, "itm_ratio": 0,
        "strategy_type": "WAIT", "strategy_note": "",
    }
    if df.empty:
        return empty

    underlying = meta.get("underlying", 0)
    pcr        = calculate_pcr(df)
    max_pain   = calculate_max_pain(df)
    score      = 0
    reasons    = []

    # ── PCR ───────────────────────────────────────────────────────────────────
    if pcr > 1.2:
        score += 25; reasons.append("PCR={} (Bullish — high put writing)".format(pcr))
    elif pcr > 0.8:
        score += 10; reasons.append("PCR={} (Neutral-Bullish)".format(pcr))
    elif pcr < 0.5:
        score -= 25; reasons.append("PCR={} (Bearish — high call writing)".format(pcr))
    else:
        score -= 10; reasons.append("PCR={} (Neutral-Bearish)".format(pcr))

    # ── Max pain pull ─────────────────────────────────────────────────────────
    pain_pct = ((max_pain - underlying) / underlying * 100) if underlying else 0
    if pain_pct > 0.3:
        score += 20; reasons.append("Max Pain {} above spot (upward pull)".format(int(max_pain)))
    elif pain_pct < -0.3:
        score -= 20; reasons.append("Max Pain {} below spot (downward pull)".format(int(max_pain)))
    else:
        reasons.append("Max Pain {} near spot (neutral)".format(int(max_pain)))

    # ── 3-strike OI sum (VarunS2002 approach — most reliable intraday signal) ─
    # call_sum < put_sum → more put writing → bullish
    atm_idx  = (df["strike"] - underlying).abs().idxmin()
    near3    = df.iloc[max(0, atm_idx - 1): min(len(df), atm_idx + 2)]
    call_sum = round(near3["ce_chg_oi"].sum() / 1000, 1)   # in thousands
    put_sum  = round(near3["pe_chg_oi"].sum() / 1000, 1)
    oi_diff  = round(call_sum - put_sum, 1)                  # negative = bullish

    if put_sum > call_sum:
        score += 15; reasons.append(
            "OI Buildup: Put Sum ({:+.0f}K) > Call Sum ({:+.0f}K) near ATM — bullish support".format(
                put_sum * 1000, call_sum * 1000))
    elif call_sum > put_sum:
        score -= 15; reasons.append(
            "OI Buildup: Call Sum ({:+.0f}K) > Put Sum ({:+.0f}K) near ATM — bearish resistance".format(
                call_sum * 1000, put_sum * 1000))

    # ── OI Boundary signals (strike 2 above / below ATM — reversal warning) ──
    call_boundary_idx = min(len(df) - 1, atm_idx + 2)
    put_boundary_idx  = max(0, atm_idx - 1)
    call_boundary_oi  = df.iloc[call_boundary_idx]["ce_chg_oi"]
    put_boundary_oi   = df.iloc[put_boundary_idx]["pe_chg_oi"]

    if call_boundary_oi < 0:
        score += 5
        reasons.append("Call boundary OI unwinding at upper strike (calls being covered)")
    if put_boundary_oi < 0:
        score -= 5
        reasons.append("Put boundary OI unwinding at lower strike (puts being covered)")

    # ── ITM ratio — put chg vs call chg (ratio > 1.5 = strong bullish) ───────
    total_ce_chg = df["ce_chg_oi"].sum()
    total_pe_chg = df["pe_chg_oi"].sum()
    if total_ce_chg > 0 and total_pe_chg > 0:
        itm_ratio = round(total_pe_chg / total_ce_chg, 2)
        if itm_ratio > 1.5:
            score += 10
            reasons.append("ITM Ratio {:.2f}x — strong put writing vs call writing (bullish)".format(itm_ratio))
        elif itm_ratio < 0.67:
            score -= 10
            reasons.append("ITM Ratio {:.2f}x — call writing dominates (bearish)".format(itm_ratio))
    elif total_pe_chg > 0 and total_ce_chg <= 0:
        itm_ratio = 99.0
        score += 10; reasons.append("PE writing with CE unwinding (strongly bullish)")
    elif total_ce_chg > 0 and total_pe_chg <= 0:
        itm_ratio = 0.0
        score -= 10; reasons.append("CE writing with PE unwinding (strongly bearish)")
    else:
        itm_ratio = 0.0

    # ── Max CE / PE OI strikes ────────────────────────────────────────────────
    ce_total = df["ce_oi"].sum()
    pe_total = df["pe_oi"].sum()
    max_ce   = df.loc[df["ce_oi"].idxmax(), "strike"] if ce_total > 0 else underlying
    max_pe   = df.loc[df["pe_oi"].idxmax(), "strike"] if pe_total > 0 else underlying
    if underlying < max_ce:
        reasons.append("Resistance at {} (max CE OI)".format(int(max_ce)))
    if underlying > max_pe:
        reasons.append("Support at {} (max PE OI)".format(int(max_pe)))

    # ── Final signal ──────────────────────────────────────────────────────────
    if score > 20:
        sig, conf, col = "BUY CALL", min(90, 50 + score),      "green"
    elif score < -20:
        sig, conf, col = "BUY PUT",  min(90, 50 + abs(score)), "red"
    else:
        sig, conf, col = "AVOID / WAIT", max(10, 50 - abs(score)), "orange"

    strategy_type, strategy_note = _strategy_for_conviction(score, min(90, 50 + abs(score)))

    return {
        "signal":            sig,
        "confidence":        conf,
        "score":             score,
        "pcr":               pcr,
        "max_pain":          max_pain,
        "max_ce_resistance": max_ce,
        "max_pe_support":    max_pe,
        "reasons":           reasons,
        "color":             col,
        "call_sum":          call_sum,
        "put_sum":           put_sum,
        "oi_difference":     oi_diff,
        "itm_ratio":         itm_ratio,
        "strategy_type":     strategy_type,
        "strategy_note":     strategy_note,
    }

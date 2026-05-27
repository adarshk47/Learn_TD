import pandas as pd
import time
import requests

# ── NSE session headers ───────────────────────────────────────────────────────
_HEADERS = {
    "user-agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "accept":          "*/*",
    "accept-language": "en,gu;q=0.9,hi;q=0.8",
    "accept-encoding": "gzip, deflate, br",
    "referer":         "https://www.nseindia.com/option-chain",
}


def _nse_fetch(url):
    """Open fresh session, warm up cookies, then fetch the API URL."""
    s = requests.Session()
    try:
        s.get("https://www.nseindia.com",               headers=_HEADERS, timeout=10)
        time.sleep(2)
        s.get("https://www.nseindia.com/option-chain",  headers=_HEADERS, timeout=10)
        time.sleep(1)
        r = s.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def fetch_option_chain(symbol, is_index=True):
    if is_index:
        url = "https://www.nseindia.com/api/option-chain-indices?symbol={}".format(symbol)
    else:
        url = "https://www.nseindia.com/api/option-chain-equities?symbol={}".format(symbol)
    return _nse_fetch(url)


def parse_option_chain(data, num_strikes=20):
    # ── guard: empty or error ─────────────────────────────────────────────────
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

        # ── find nearest expiry from the RECORDS themselves ───────────────────
        # (avoids date-format mismatch between header list and record strings)
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
            # handle both key names NSE has used over time
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


def generate_signal(df, meta):
    empty = {
        "signal": "NO DATA", "confidence": 0, "reasons": [],
        "color": "orange", "pcr": 0, "max_pain": 0,
        "max_ce_resistance": 0, "max_pe_support": 0, "score": 0,
    }
    if df.empty:
        return empty

    underlying = meta.get("underlying", 0)
    pcr        = calculate_pcr(df)
    max_pain   = calculate_max_pain(df)
    score      = 0
    reasons    = []

    # PCR
    if pcr > 1.2:
        score += 25; reasons.append("PCR={} (Bullish - high put writing)".format(pcr))
    elif pcr > 0.8:
        score += 10; reasons.append("PCR={} (Neutral-Bullish)".format(pcr))
    elif pcr < 0.5:
        score -= 25; reasons.append("PCR={} (Bearish - high call writing)".format(pcr))
    else:
        score -= 10; reasons.append("PCR={} (Neutral-Bearish)".format(pcr))

    # Max Pain
    pain_pct = ((max_pain - underlying) / underlying * 100) if underlying else 0
    if pain_pct > 0.3:
        score += 20; reasons.append("Max Pain {} above spot (upward pull)".format(int(max_pain)))
    elif pain_pct < -0.3:
        score -= 20; reasons.append("Max Pain {} below spot (downward pull)".format(int(max_pain)))
    else:
        reasons.append("Max Pain {} near spot (neutral)".format(int(max_pain)))

    # ATM OI change
    atm_idx = (df["strike"] - underlying).abs().idxmin()
    near    = df.iloc[max(0, atm_idx - 3): atm_idx + 4]
    net_chg = near["pe_chg_oi"].sum() - near["ce_chg_oi"].sum()
    if net_chg > 0:
        score += 15; reasons.append("Fresh PE writing near ATM (support building)")
    elif net_chg < 0:
        score -= 15; reasons.append("Fresh CE writing near ATM (resistance building)")

    # Support / resistance from max OI
    ce_sum = df["ce_oi"].sum()
    pe_sum = df["pe_oi"].sum()
    max_ce = df.loc[df["ce_oi"].idxmax(), "strike"] if ce_sum > 0 else underlying
    max_pe = df.loc[df["pe_oi"].idxmax(), "strike"] if pe_sum > 0 else underlying
    if underlying < max_ce:
        reasons.append("Resistance at {} (max CE OI)".format(int(max_ce)))
    if underlying > max_pe:
        reasons.append("Support at {} (max PE OI)".format(int(max_pe)))

    if score > 20:
        sig, conf, col = "BUY CALL", min(90, 50 + score),      "green"
    elif score < -20:
        sig, conf, col = "BUY PUT",  min(90, 50 + abs(score)), "red"
    else:
        sig, conf, col = "AVOID / WAIT", max(10, 50 - abs(score)), "orange"

    return {
        "signal": sig, "confidence": conf, "score": score,
        "pcr": pcr, "max_pain": max_pain,
        "max_ce_resistance": max_ce, "max_pe_support": max_pe,
        "reasons": reasons, "color": col,
    }

import requests
import pandas as pd
import time

NSE_HEADERS = {
    "User-Agent":                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language":           "en-US,en;q=0.9",
    "Accept-Encoding":           "gzip, deflate, br",
    "Connection":                "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest":            "document",
    "Sec-Fetch-Mode":            "navigate",
    "Sec-Fetch-Site":            "none",
    "Sec-Fetch-User":            "?1",
    "Cache-Control":             "max-age=0",
}

API_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.nseindia.com/option-chain",
    "X-Requested-With":"XMLHttpRequest",
    "Sec-Fetch-Dest":  "empty",
    "Sec-Fetch-Mode":  "cors",
    "Sec-Fetch-Site":  "same-origin",
    "Connection":      "keep-alive",
}

BASE_URL          = "https://www.nseindia.com"
OPTION_CHAIN_URL  = BASE_URL + "/api/option-chain-indices?symbol={symbol}"
STOCK_OPTION_URL  = BASE_URL + "/api/option-chain-equities?symbol={symbol}"

_session      = None
_session_time = 0


def _build_session():
    """
    NSE requires a browser-like warm-up before API calls will succeed.
    Step 1: visit homepage  (get base cookies)
    Step 2: visit option-chain page  (get NSE-specific cookies)
    Step 3: API call is now accepted
    """
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    try:
        s.get(BASE_URL, timeout=10)
        time.sleep(1)
        s.get(BASE_URL + "/option-chain", timeout=10)
        time.sleep(0.5)
    except Exception:
        pass
    s.headers.update(API_HEADERS)
    return s


def _get_session():
    global _session, _session_time
    if _session is None or (time.time() - _session_time) > 300:
        _session      = _build_session()
        _session_time = time.time()
    return _session


def fetch_option_chain(symbol, is_index=True):
    url = OPTION_CHAIN_URL.format(symbol=symbol) if is_index else STOCK_OPTION_URL.format(symbol=symbol)
    for attempt in range(2):
        try:
            session  = _get_session()
            resp     = session.get(url, timeout=15)
            if resp.status_code in (401, 403, 404) and attempt == 0:
                global _session
                _session = None
                time.sleep(1.5)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == 0:
                global _session
                _session = None
                time.sleep(1.5)
            else:
                return {"error": str(e)}
    return {"error": "Failed after 2 attempts"}


def parse_option_chain(data, num_strikes=20):
    if "error" in data:
        return pd.DataFrame(), {"error": data["error"]}
    try:
        records        = data["records"]["data"]
        expiry_dates   = data["records"]["expiryDates"]
        underlying_val = data["records"]["underlyingValue"]
        nearest_expiry = expiry_dates[0]

        rows = []
        for rec in records:
            if rec.get("expiryDate") != nearest_expiry:
                continue
            strike = rec["strikePrice"]
            ce     = rec.get("CE", {})
            pe     = rec.get("PE", {})
            rows.append({
                "strike":    strike,
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

        df      = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
        atm_idx = (df["strike"] - underlying_val).abs().idxmin()
        half    = num_strikes // 2
        df      = df.iloc[max(0, atm_idx - half): atm_idx + half].reset_index(drop=True)
        atm     = df.iloc[(df["strike"] - underlying_val).abs().idxmin()]["strike"]

        return df, {
            "underlying":   underlying_val,
            "expiry":       nearest_expiry,
            "all_expiries": expiry_dates,
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
            row["ce_oi"] * max(0, s - row["strike"]) +
            row["pe_oi"] * max(0, row["strike"] - s)
            for _, row in df.iterrows()
        )
        if pain < min_pain:
            min_pain = pain
            best     = s
    return best


def generate_signal(df, meta):
    if df.empty:
        return {"signal": "NO DATA", "confidence": 0, "reasons": [], "color": "orange",
                "pcr": 0, "max_pain": 0, "max_ce_resistance": 0, "max_pe_support": 0, "score": 0}

    underlying = meta.get("underlying", 0)
    pcr        = calculate_pcr(df)
    max_pain   = calculate_max_pain(df)
    score      = 0
    reasons    = []

    if pcr > 1.2:
        score += 25
        reasons.append("PCR={} (Bullish - high put writing)".format(pcr))
    elif pcr > 0.8:
        score += 10
        reasons.append("PCR={} (Neutral-Bullish)".format(pcr))
    elif pcr < 0.5:
        score -= 25
        reasons.append("PCR={} (Bearish - high call writing)".format(pcr))
    else:
        score -= 10
        reasons.append("PCR={} (Neutral-Bearish)".format(pcr))

    pain_pct = ((max_pain - underlying) / underlying) * 100 if underlying else 0
    if pain_pct > 0.3:
        score += 20
        reasons.append("Max Pain {} above spot (upward pull)".format(int(max_pain)))
    elif pain_pct < -0.3:
        score -= 20
        reasons.append("Max Pain {} below spot (downward pull)".format(int(max_pain)))
    else:
        reasons.append("Max Pain {} near spot (neutral)".format(int(max_pain)))

    atm_idx = (df["strike"] - underlying).abs().idxmin()
    near_df = df.iloc[max(0, atm_idx - 3): atm_idx + 4]
    net_chg = near_df["pe_chg_oi"].sum() - near_df["ce_chg_oi"].sum()
    if net_chg > 0:
        score += 15
        reasons.append("Fresh PE writing near ATM (support building)")
    elif net_chg < 0:
        score -= 15
        reasons.append("Fresh CE writing near ATM (resistance building)")

    max_ce_strike = df.loc[df["ce_oi"].idxmax(), "strike"]
    max_pe_strike = df.loc[df["pe_oi"].idxmax(), "strike"]
    if underlying < max_ce_strike:
        reasons.append("Resistance at {} (max CE OI)".format(int(max_ce_strike)))
    if underlying > max_pe_strike:
        reasons.append("Support at {} (max PE OI)".format(int(max_pe_strike)))

    if score > 20:
        signal, confidence, color = "BUY CALL", min(90, 50 + score), "green"
    elif score < -20:
        signal, confidence, color = "BUY PUT",  min(90, 50 + abs(score)), "red"
    else:
        signal, confidence, color = "AVOID / WAIT", max(10, 50 - abs(score)), "orange"

    return {
        "signal": signal, "confidence": confidence, "score": score,
        "pcr": pcr, "max_pain": max_pain,
        "max_ce_resistance": max_ce_strike,
        "max_pe_support":    max_pe_strike,
        "reasons": reasons, "color": color,
    }

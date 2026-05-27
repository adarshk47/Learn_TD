import pandas as pd
import numpy as np
import yfinance as yf
import time

# yfinance symbol mapping for NSE
def _yf_symbol(symbol, is_index=False):
    index_map = {
        "NIFTY":      "^NSEI",
        "BANKNIFTY":  "^NSEBANK",
        "FINNIFTY":   "^CNXFIN",
        "MIDCPNIFTY": "^NSEMDCP50",
    }
    if is_index:
        return index_map.get(symbol, "^NSEI")
    return symbol + ".NS"


def fetch_option_chain(symbol, is_index=False):
    try:
        yf_sym = _yf_symbol(symbol, is_index)
        ticker  = yf.Ticker(yf_sym)
        info    = ticker.fast_info

        # Get spot price
        try:
            spot = info.last_price
        except Exception:
            spot = ticker.history(period="1d")["Close"].iloc[-1]

        # Get option expiry dates
        expiries = ticker.options
        if not expiries:
            return {"error": "No option chain available for {}. Try a different stock (e.g. SBIN, RELIANCE).".format(symbol)}

        nearest = expiries[0]
        chain   = ticker.option_chain(nearest)
        calls   = chain.calls
        puts    = chain.puts

        return {
            "_format":   "yfinance",
            "_calls":    calls,
            "_puts":     puts,
            "_spot":     spot,
            "_expiry":   nearest,
            "_expiries": list(expiries),
            "_symbol":   symbol,
        }
    except Exception as e:
        return {"error": str(e)}


def parse_option_chain(data, num_strikes=20):
    if "error" in data:
        return pd.DataFrame(), {"error": data["error"]}

    try:
        calls    = data["_calls"].copy()
        puts     = data["_puts"].copy()
        spot     = data["_spot"]
        expiry   = data["_expiry"]
        expiries = data["_expiries"]

        # Rename yfinance columns to our internal format
        calls = calls[["strike", "openInterest", "volume", "impliedVolatility",
                        "lastPrice", "bid", "ask"]].copy()
        puts  = puts[["strike",  "openInterest", "volume", "impliedVolatility",
                        "lastPrice", "bid", "ask"]].copy()

        calls.columns = ["strike", "ce_oi",  "ce_volume", "ce_iv",  "ce_ltp",  "ce_bid",  "ce_ask"]
        puts.columns  = ["strike", "pe_oi",  "pe_volume", "pe_iv",  "pe_ltp",  "pe_bid",  "pe_ask"]

        # yfinance gives IV as decimal (0.20 = 20%) — convert to %
        calls["ce_iv"] = (calls["ce_iv"] * 100).round(1)
        puts["pe_iv"]  = (puts["pe_iv"]  * 100).round(1)

        # yfinance doesn't provide change-in-OI directly
        calls["ce_chg_oi"] = 0
        puts["pe_chg_oi"]  = 0

        df = pd.merge(calls, puts, on="strike", how="outer").fillna(0)
        df = df.sort_values("strike").reset_index(drop=True)
        df["strike"] = df["strike"].astype(float)

        if df.empty:
            return pd.DataFrame(), {"error": "Empty option chain returned"}

        # Slice around ATM
        atm_idx = (df["strike"] - spot).abs().idxmin()
        half    = num_strikes // 2
        df      = df.iloc[max(0, atm_idx - half): min(len(df), atm_idx + half)].reset_index(drop=True)
        atm     = df.iloc[(df["strike"] - spot).abs().idxmin()]["strike"]

        meta = {
            "underlying":   round(spot, 2),
            "expiry":       expiry,
            "all_expiries": expiries,
            "atm":          atm,
        }
        return df, meta

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
        return {
            "signal": "NO DATA", "confidence": 0, "reasons": [],
            "color": "orange", "pcr": 0, "max_pain": 0,
            "max_ce_resistance": 0, "max_pe_support": 0, "score": 0,
        }

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

    # Ensure OI columns have non-zero values before idxmax
    ce_col = df["ce_oi"]
    pe_col = df["pe_oi"]
    max_ce_strike = df.loc[ce_col.idxmax(), "strike"] if ce_col.sum() > 0 else underlying
    max_pe_strike = df.loc[pe_col.idxmax(), "strike"] if pe_col.sum() > 0 else underlying

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

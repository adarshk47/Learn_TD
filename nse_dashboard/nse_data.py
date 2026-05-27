import requests
import pandas as pd
import time

NSE_HEADERS = {
    User-Agent Mozilla5.0 (Windows NT 10.0; Win64; x64) AppleWebKit537.36 (KHTML, like Gecko) Chrome120.0.0.0 Safari537.36,
    Accept ,
    Accept-Language en-US,en;q=0.9,
    Accept-Encoding gzip, deflate, br,
    Referer httpswww.nseindia.comoption-chain,
}

BASE_URL = httpswww.nseindia.com
OPTION_CHAIN_URL = BASE_URL + apioption-chain-indicessymbol={symbol}
STOCK_OPTION_URL = BASE_URL + apioption-chain-equitiessymbol={symbol}

_session = None
_session_time = 0


def get_session()
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try
        session.get(BASE_URL, timeout=10)
        time.sleep(0.5)
    except Exception
        pass
    return session


def get_or_refresh_session()
    global _session, _session_time
    if _session is None or (time.time() - _session_time)  300
        _session = get_session()
        _session_time = time.time()
    return _session


def fetch_option_chain(symbol str, is_index bool = True) - dict
    session = get_or_refresh_session()
    url = OPTION_CHAIN_URL.format(symbol=symbol) if is_index else STOCK_OPTION_URL.format(symbol=symbol)
    try
        resp = session.get(url, timeout=15)
        if resp.status_code == 401
            global _session
            _session = None
            session = get_or_refresh_session()
            resp = session.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e
        return {error str(e)}


def parse_option_chain(data dict, num_strikes int = 20) - tuple
    if error in data
        return pd.DataFrame(), {error data[error]}
    try
        records = data[records][data]
        expiry_dates = data[records][expiryDates]
        underlying_value = data[records][underlyingValue]
        nearest_expiry = expiry_dates[0]
        rows = []
        for rec in records
            if rec.get(expiryDate) != nearest_expiry
                continue
            strike = rec[strikePrice]
            ce = rec.get(CE, {})
            pe = rec.get(PE, {})
            rows.append({
                strike strike,
                ce_oi ce.get(openInterest, 0),
                ce_chg_oi ce.get(changeinOpenInterest, 0),
                ce_volume ce.get(totalTradedVolume, 0),
                ce_iv ce.get(impliedVolatility, 0),
                ce_ltp ce.get(lastPrice, 0),
                ce_bid ce.get(bidPrice, 0),
                ce_ask ce.get(askPrice, 0),
                pe_oi pe.get(openInterest, 0),
                pe_chg_oi pe.get(changeinOpenInterest, 0),
                pe_volume pe.get(totalTradedVolume, 0),
                pe_iv pe.get(impliedVolatility, 0),
                pe_ltp pe.get(lastPrice, 0),
                pe_bid pe.get(bidPrice, 0),
                pe_ask pe.get(askPrice, 0),
            })
        df = pd.DataFrame(rows).sort_values(strike).reset_index(drop=True)
        atm_idx = (df[strike] - underlying_value).abs().idxmin()
        half = num_strikes  2
        start = max(0, atm_idx - half)
        end = min(len(df), atm_idx + half)
        df = df.iloc[startend].reset_index(drop=True)
        meta = {
            underlying underlying_value,
            expiry nearest_expiry,
            all_expiries expiry_dates,
            atm df.iloc[(df[strike] - underlying_value).abs().idxmin()][strike],
        }
        return df, meta
    except Exception as e
        return pd.DataFrame(), {error str(e)}


def calculate_pcr(df pd.DataFrame) - float
    total_ce_oi = df[ce_oi].sum()
    total_pe_oi = df[pe_oi].sum()
    if total_ce_oi == 0
        return 0.0
    return round(total_pe_oi  total_ce_oi, 2)


def calculate_max_pain(df pd.DataFrame) - float
    strikes = df[strike].tolist()
    best_strike = strikes[0]
    min_pain = float(inf)
    for s in strikes
        pain = 0
        for _, row in df.iterrows()
            pain += row[ce_oi]  max(0, s - row[strike])
            pain += row[pe_oi]  max(0, row[strike] - s)
        if pain  min_pain
            min_pain = pain
            best_strike = s
    return best_strike


def generate_signal(df pd.DataFrame, meta dict) - dict
    if df.empty
        return {signal NO DATA, confidence 0, reason Data unavailable}

    underlying = meta.get(underlying, 0)
    pcr = calculate_pcr(df)
    max_pain = calculate_max_pain(df)
    score = 0
    reasons = []

    if pcr  1.2
        score += 25
        reasons.append(fPCR={pcr} (Bullish - high put writing))
    elif pcr  0.8
        score += 10
        reasons.append(fPCR={pcr} (Neutral-Bullish))
    elif pcr  0.5
        score -= 25
        reasons.append(fPCR={pcr} (Bearish - high call writing))
    else
        score -= 10
        reasons.append(fPCR={pcr} (Neutral-Bearish))

    pain_diff_pct = ((max_pain - underlying)  underlying)  100
    if pain_diff_pct  0.3
        score += 20
        reasons.append(fMax Pain {max_pain} above spot (upward pull))
    elif pain_diff_pct  -0.3
        score -= 20
        reasons.append(fMax Pain {max_pain} below spot (downward pull))
    else
        reasons.append(fMax Pain {max_pain} near spot (neutral))

    atm_idx = (df[strike] - underlying).abs().idxmin()
    near_df = df.iloc[max(0, atm_idx - 3) atm_idx + 4]
    net_oi_chg = near_df[pe_chg_oi].sum() - near_df[ce_chg_oi].sum()
    if net_oi_chg  0
        score += 15
        reasons.append(Fresh PE writing near ATM (support building))
    elif net_oi_chg  0
        score -= 15
        reasons.append(Fresh CE writing near ATM (resistance building))

    max_ce_oi_strike = df.loc[df[ce_oi].idxmax(), strike]
    max_pe_oi_strike = df.loc[df[pe_oi].idxmax(), strike]
    if underlying  max_ce_oi_strike
        reasons.append(fResistance at {max_ce_oi_strike} (max CE OI))
    if underlying  max_pe_oi_strike
        reasons.append(fSupport at {max_pe_oi_strike} (max PE OI))

    if score  20
        signal, confidence, color = BUY CALL, min(90, 50 + score), green
    elif score  -20
        signal, confidence, color = BUY PUT, min(90, 50 + abs(score)), red
    else
        signal, confidence, color = AVOID  WAIT, max(10, 50 - abs(score)), orange

    return {
        signal signal, confidence confidence, score score,
        pcr pcr, max_pain max_pain,
        max_ce_resistance max_ce_oi_strike, max_pe_support max_pe_oi_strike,
        reasons reasons, color color,
    }
"""
Angel One SmartAPI — option chain fetcher.
Handles authentication, instrument master caching, market data, IV calculation.
"""
import os, json, math, time, requests
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import pyotp
from config import ANGEL_ONE

_INSTRUMENT_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
_INSTRUMENT_CACHE = os.path.join(os.path.dirname(__file__), ".instrument_cache.json")
_INSTRUMENT_CACHE_DATE = os.path.join(os.path.dirname(__file__), ".instrument_cache_date.txt")

# ── Black-Scholes IV (Newton-Raphson) ────────────────────────────────────────

def _bs_price(S, K, T, r, sigma, opt):
    if T <= 0 or sigma <= 0:
        return max(0, (S - K) if opt == "C" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from scipy.stats import norm
    if opt == "C":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def _bs_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1e-6
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    from scipy.stats import norm
    return S * norm.pdf(d1) * math.sqrt(T)

def implied_vol(market_price, S, K, T, r=0.065, opt="C"):
    """Return IV as percentage (e.g. 14.5 means 14.5%)."""
    if market_price <= 0 or T <= 0:
        return 0.0
    sigma = 0.25
    for _ in range(50):
        price = _bs_price(S, K, T, r, sigma, opt)
        vega  = _bs_vega(S, K, T, r, sigma)
        diff  = price - market_price
        if abs(diff) < 1e-5:
            break
        if vega < 1e-8:
            break
        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 10.0))
    return round(sigma * 100, 2)

# ── Instrument master ─────────────────────────────────────────────────────────

def _load_instruments():
    today = date.today().isoformat()
    if os.path.exists(_INSTRUMENT_CACHE_DATE):
        cached_date = open(_INSTRUMENT_CACHE_DATE).read().strip()
        if cached_date == today and os.path.exists(_INSTRUMENT_CACHE):
            with open(_INSTRUMENT_CACHE) as f:
                return json.load(f)
    try:
        resp = requests.get(_INSTRUMENT_URL, timeout=30)
        data = resp.json()
        with open(_INSTRUMENT_CACHE, "w") as f:
            json.dump(data, f)
        with open(_INSTRUMENT_CACHE_DATE, "w") as f:
            f.write(today)
        return data
    except Exception as e:
        if os.path.exists(_INSTRUMENT_CACHE):
            with open(_INSTRUMENT_CACHE) as f:
                return json.load(f)
        return []

def _nearest_expiry(expiries):
    """Return the nearest future expiry date string from list."""
    today = date.today()
    future = []
    for e in expiries:
        for fmt in ("%d%b%Y", "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y"):
            try:
                d = datetime.strptime(e.upper(), fmt.upper()).date()
                if d >= today:
                    future.append((d, e))
                break
            except Exception:
                pass
    if not future:
        return expiries[0] if expiries else None
    return sorted(future)[0][1]

def _get_option_tokens(instruments, symbol, expiry_str):
    """
    Return dict: strike -> {"CE": token, "PE": token, "lot_size": n}
    Handles index (NFO exch, OPTIDX) and stock (NFO exch, OPTSTK) instruments.
    """
    result = {}
    for inst in instruments:
        if inst.get("exch_seg") != "NFO":
            continue
        inst_type = inst.get("instrumenttype", "")
        if inst_type not in ("OPTIDX", "OPTSTK"):
            continue
        name = inst.get("name", "").upper()
        if name != symbol.upper():
            continue
        # expiry check — instrument expiry in DDMMMYYYY
        inst_exp = inst.get("expiry", "").upper()
        # normalise both to YYYY-MM-DD for comparison
        def to_date(s):
            for fmt in ("%d%b%Y", "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y"):
                try:
                    return datetime.strptime(s.upper(), fmt.upper()).date()
                except Exception:
                    pass
            return None
        inst_date = to_date(inst_exp)
        tgt_date  = to_date(expiry_str)
        if inst_date != tgt_date:
            continue
        try:
            strike    = float(inst.get("strike", 0)) / 100  # Angel stores strike*100
            opt_type  = inst.get("symbol", "")[-2:]         # last 2 chars: CE or PE
            token     = inst.get("token", "")
            lot_size  = int(inst.get("lotsize", 1))
            if strike not in result:
                result[strike] = {"lot_size": lot_size}
            result[strike][opt_type] = token
        except Exception:
            pass
    return result

def _get_all_expiries(instruments, symbol):
    expiries = set()
    for inst in instruments:
        if inst.get("exch_seg") != "NFO":
            continue
        if inst.get("instrumenttype") not in ("OPTIDX", "OPTSTK"):
            continue
        if inst.get("name", "").upper() != symbol.upper():
            continue
        e = inst.get("expiry", "")
        if e:
            expiries.add(e)
    def to_date(s):
        for fmt in ("%d%b%Y", "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y"):
            try:
                return datetime.strptime(s.upper(), fmt.upper()).date()
            except Exception:
                pass
        return date.max
    return sorted(expiries, key=to_date)

# ── Angel One session ─────────────────────────────────────────────────────────

_smart = None
_auth_token = None
_login_time = 0

def _login():
    global _smart, _auth_token, _login_time
    from SmartApi import SmartConnect
    cfg   = ANGEL_ONE
    totp  = pyotp.TOTP(cfg["totp_secret"]).now()
    smart = SmartConnect(api_key=cfg["api_key"])
    data  = smart.generateSession(cfg["client_id"], cfg["mpin"], totp)
    if not data.get("status"):
        raise RuntimeError("Angel One login failed: " + str(data.get("message", data)))
    _smart      = smart
    _auth_token = data["data"]["jwtToken"]
    _login_time = time.time()
    return smart

def _get_session():
    global _smart, _login_time
    # re-login every 6 hours
    if _smart is None or (time.time() - _login_time) > 6 * 3600:
        _login()
    return _smart

# ── Main public function ──────────────────────────────────────────────────────

def fetch_historical_candles(symbol, is_index=True, days=30):
    """Wrapper: get historical daily OHLCV using our Angel One session."""
    try:
        smart = _get_session()
    except Exception:
        return None
    from backtest import get_historical_candles
    return get_historical_candles(smart, symbol, is_index, days)


def fetch_option_chain_angelone(symbol, is_index=True, num_strikes=20):
    """
    Returns (df, meta) with same schema as nse_data.parse_option_chain.
    df columns: strike, ce_oi, ce_chg_oi, ce_volume, ce_iv, ce_ltp, ce_bid, ce_ask,
                        pe_oi, pe_chg_oi, pe_volume, pe_iv, pe_ltp, pe_bid, pe_ask
    meta keys: underlying, expiry, all_expiries, atm, source
    """
    try:
        smart = _get_session()
    except Exception as e:
        return pd.DataFrame(), {"error": "Angel One login failed: " + str(e)}

    # 1. Load instrument master
    try:
        instruments = _load_instruments()
    except Exception as e:
        return pd.DataFrame(), {"error": "Instrument master load failed: " + str(e)}

    # 2. Find all expiries for symbol and pick nearest
    all_expiries = _get_all_expiries(instruments, symbol)
    if not all_expiries:
        return pd.DataFrame(), {"error": "No NFO instruments found for symbol: " + symbol}

    nearest_exp = _nearest_expiry(all_expiries)
    if not nearest_exp:
        return pd.DataFrame(), {"error": "No future expiry found for " + symbol}

    # 3. Get strike → token mapping
    token_map = _get_option_tokens(instruments, symbol, nearest_exp)
    if not token_map:
        return pd.DataFrame(), {"error": "No option tokens for {}, expiry {}".format(symbol, nearest_exp)}

    # 4. Get underlying spot from LTP
    try:
        if is_index:
            idx_exchange_map = {
                "NIFTY":      ("NSE", "99926000"),
                "BANKNIFTY":  ("NSE", "99926009"),
                "FINNIFTY":   ("NSE", "99926037"),
                "MIDCPNIFTY": ("NSE", "99926074"),
                "SENSEX":     ("BSE", "1"),
            }
            exch, spot_token = idx_exchange_map.get(symbol.upper(), ("NSE", "99926000"))
            ltp_data = smart.ltpData(exch, symbol, spot_token)
            underlying = float(ltp_data["data"]["ltp"])
        else:
            # equity spot
            eq_inst = next((i for i in instruments
                            if i.get("exch_seg") == "NSE"
                            and i.get("symbol", "").upper() == symbol.upper() + "-EQ"), None)
            if eq_inst:
                ltp_data   = smart.ltpData("NSE", symbol + "-EQ", eq_inst["token"])
                underlying = float(ltp_data["data"]["ltp"])
            else:
                # fallback: average of first few strike LTPs
                underlying = list(token_map.keys())[len(token_map) // 2]
    except Exception as e:
        underlying = list(token_map.keys())[len(token_map) // 2]

    # 5. Select strikes around ATM
    all_strikes = sorted(token_map.keys())
    atm_strike  = min(all_strikes, key=lambda s: abs(s - underlying))
    atm_idx     = all_strikes.index(atm_strike)
    half        = num_strikes // 2
    selected_strikes = all_strikes[max(0, atm_idx - half): atm_idx + half + 1]

    # 6. Fetch market data in batches of 50
    def _batched_market_data(tokens_exchange_pairs):
        results = {}
        batch_size = 50
        items = list(tokens_exchange_pairs.items())
        for i in range(0, len(items), batch_size):
            batch = items[i: i + batch_size]
            exchange_tokens = {}
            for token, exch in batch:
                exchange_tokens.setdefault(exch, []).append(token)
            for exch, tok_list in exchange_tokens.items():
                try:
                    resp = smart.getMarketData("FULL", {exch: tok_list})
                    fetched = resp.get("data", {}).get("fetched", [])
                    for item in fetched:
                        t = item.get("symbolToken") or item.get("token")
                        results[str(t)] = item
                except Exception:
                    pass
            time.sleep(0.1)
        return results

    # Collect all tokens to fetch
    tokens_to_fetch = {}
    for strike in selected_strikes:
        info = token_map[strike]
        for opt_type in ("CE", "PE"):
            tok = info.get(opt_type)
            if tok:
                tokens_to_fetch[str(tok)] = "NFO"

    market_data = _batched_market_data(tokens_to_fetch)

    # 7. Calculate time to expiry
    def _exp_to_date(s):
        for fmt in ("%d%b%Y", "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y"):
            try:
                return datetime.strptime(s.upper(), fmt.upper()).date()
            except Exception:
                pass
        return date.today() + timedelta(days=7)

    exp_date = _exp_to_date(nearest_exp)
    T = max(0.0, (exp_date - date.today()).days) / 365.0

    # 8. Build rows
    rows = []
    prev_oi_file = os.path.join(os.path.dirname(__file__), ".prev_oi_{}.json".format(symbol))
    prev_oi = {}
    if os.path.exists(prev_oi_file):
        try:
            with open(prev_oi_file) as f:
                prev_oi = json.load(f)
        except Exception:
            pass

    curr_oi = {}
    for strike in selected_strikes:
        info     = token_map[strike]
        lot_size = info.get("lot_size", 1)

        def _get(opt_type):
            tok = str(info.get(opt_type, ""))
            md  = market_data.get(tok, {})
            oi      = int(md.get("opnInterest", md.get("openInterest", 0)))
            volume  = int(md.get("tradeVolume", md.get("totTrdVol", 0)))
            ltp     = float(md.get("ltp", 0))
            bid     = float(md.get("buyPrice1", md.get("bidPrice", ltp)))
            ask     = float(md.get("sellPrice1", md.get("askPrice", ltp)))
            iv      = implied_vol(ltp, underlying, strike, T, opt=opt_type[0])
            key     = "{}_{}".format(strike, opt_type)
            chg_oi  = oi - prev_oi.get(key, oi)
            curr_oi[key] = oi
            return oi, chg_oi, volume, iv, ltp, bid, ask

        ce_oi, ce_chg, ce_vol, ce_iv, ce_ltp, ce_bid, ce_ask = _get("CE")
        pe_oi, pe_chg, pe_vol, pe_iv, pe_ltp, pe_bid, pe_ask = _get("PE")

        rows.append({
            "strike":    strike,
            "ce_oi":     ce_oi,  "ce_chg_oi": ce_chg,  "ce_volume": ce_vol,
            "ce_iv":     ce_iv,  "ce_ltp":    ce_ltp,   "ce_bid":    ce_bid, "ce_ask": ce_ask,
            "pe_oi":     pe_oi,  "pe_chg_oi": pe_chg,   "pe_volume": pe_vol,
            "pe_iv":     pe_iv,  "pe_ltp":    pe_ltp,   "pe_bid":    pe_bid, "pe_ask": pe_ask,
        })

    # Save current OI for next refresh
    try:
        with open(prev_oi_file, "w") as f:
            json.dump(curr_oi, f)
    except Exception:
        pass

    if not rows:
        return pd.DataFrame(), {"error": "No market data fetched for selected strikes"}

    df  = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    atm = float(df.iloc[(df["strike"] - underlying).abs().idxmin()]["strike"])

    # Format expiry for display
    try:
        exp_display = exp_date.strftime("%d-%b-%Y").upper()
    except Exception:
        exp_display = nearest_exp

    meta = {
        "underlying":   underlying,
        "expiry":       exp_display,
        "all_expiries": [_exp_to_date(e).strftime("%d-%b-%Y").upper() for e in all_expiries[:5]],
        "atm":          atm,
        "source":       "Angel One SmartAPI",
    }
    return df, meta

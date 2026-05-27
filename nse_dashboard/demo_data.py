import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_demo_option_chain(symbol: str = "NIFTY") -> tuple:
    random.seed(42)
    np.random.seed(42)

    base_prices = {"NIFTY": 24350, "BANKNIFTY": 52400, "FINNIFTY": 23800, "MIDCPNIFTY": 12200}
    strike_gaps = {"NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50, "MIDCPNIFTY": 25}

    underlying = base_prices.get(symbol, 24350) + random.randint(-200, 200)
    gap = strike_gaps.get(symbol, 50)
    atm = round(underlying / gap) * gap
    strikes = [atm + (i - 10) * gap for i in range(21)]
    expiry = (datetime.now() + timedelta(days=(3 - datetime.now().weekday()) % 7 + 1)).strftime("%d-%b-%Y").upper()

    rows = []
    for strike in strikes:
        dist = (strike - atm) / gap

        ce_oi_base = max(100, int(500000 * np.exp(-0.08 * max(0, dist) ** 2) * (1 + max(0, dist) * 0.3)))
        ce_oi = ce_oi_base + random.randint(-50000, 50000)
        ce_chg_oi = random.randint(-80000, 80000)
        ce_iv = max(8, 14 + dist * 1.5 + random.uniform(-1, 1))
        ce_ltp = max(0.5, (atm - strike + 200) * 0.3 + random.uniform(-2, 2)) if strike <= atm else max(0.5, 50 * np.exp(-0.05 * (strike - atm) / gap) + random.uniform(-1, 1))

        pe_oi_base = max(100, int(500000 * np.exp(-0.08 * max(0, -dist) ** 2) * (1 + max(0, -dist) * 0.3)))
        pe_oi = pe_oi_base + random.randint(-50000, 50000)
        pe_chg_oi = random.randint(-80000, 80000)
        pe_iv = max(8, 14 - dist * 1.5 + random.uniform(-1, 1))
        pe_ltp = max(0.5, (strike - atm + 200) * 0.3 + random.uniform(-2, 2)) if strike >= atm else max(0.5, 50 * np.exp(0.05 * (strike - atm) / gap) + random.uniform(-1, 1))

        rows.append({
            "strike": strike,
            "ce_oi": max(0, ce_oi), "ce_chg_oi": ce_chg_oi,
            "ce_volume": max(0, ce_oi // 3 + random.randint(-1000, 5000)),
            "ce_iv": round(ce_iv, 1), "ce_ltp": round(ce_ltp, 2),
            "ce_bid": round(ce_ltp - 0.5, 2), "ce_ask": round(ce_ltp + 0.5, 2),
            "pe_oi": max(0, pe_oi), "pe_chg_oi": pe_chg_oi,
            "pe_volume": max(0, pe_oi // 3 + random.randint(-1000, 5000)),
            "pe_iv": round(pe_iv, 1), "pe_ltp": round(pe_ltp, 2),
            "pe_bid": round(pe_ltp - 0.5, 2), "pe_ask": round(pe_ltp + 0.5, 2),
        })

    rows[14]["ce_oi"] = 1800000
    rows[14]["ce_chg_oi"] = 250000
    rows[7]["pe_oi"] = 1650000
    rows[7]["pe_chg_oi"] = 200000

    df = pd.DataFrame(rows)
    meta = {
        "underlying": underlying, "expiry": expiry,
        "all_expiries": [expiry], "atm": atm, "demo": True,
    }
    return df, meta
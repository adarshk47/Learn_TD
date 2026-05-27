"""
NSE CSV Option Chain Parser.
The user downloads the CSV from nseindia.com/option-chain manually,
then uploads it via the Streamlit file_uploader widget.
Columns expected (NSE format):
  OI, Chng in OI, Volume, IV, LTP, Net Chng, Strike Price,
  Net Chng, LTP, IV, Volume, Chng in OI, OI
  (CE side left of Strike, PE side right)
"""
import io
import pandas as pd


def parse_nse_csv(file_obj) -> tuple[pd.DataFrame, dict]:
    """
    file_obj: file-like object from st.file_uploader.
    Returns (df, meta) with same schema as nse_data.parse_option_chain.
    """
    try:
        content = file_obj.read()
        text    = content.decode("utf-8", errors="replace")
        lines   = text.strip().splitlines()
    except Exception as e:
        return pd.DataFrame(), {"error": "Could not read file: " + str(e)}

    # ── Find underlying value and expiry from NSE CSV header lines ──────────
    underlying = 0.0
    expiry     = "Unknown"
    data_start = 0

    for i, line in enumerate(lines):
        low = line.lower()
        if "underlying index" in low or "underlying value" in low or "spot price" in low:
            parts = line.split(",")
            for p in parts:
                try:
                    val = float(p.replace('"', '').strip())
                    if val > 1000:
                        underlying = val
                except ValueError:
                    pass
        if "expiry date" in low or "expiry" in low:
            parts = line.split(",")
            for p in parts:
                p = p.strip().strip('"')
                if any(m in p.upper() for m in ["JAN","FEB","MAR","APR","MAY","JUN",
                                                  "JUL","AUG","SEP","OCT","NOV","DEC"]):
                    expiry = p
        # detect header row
        if "strike price" in low or "strike" in low:
            data_start = i
            break

    if data_start == 0:
        return pd.DataFrame(), {"error": "Could not find 'Strike Price' column in CSV. "
                                         "Make sure you downloaded the NSE option chain CSV."}

    try:
        df_raw = pd.read_csv(io.StringIO("\n".join(lines[data_start:])), header=0)
    except Exception as e:
        return pd.DataFrame(), {"error": "CSV parse error: " + str(e)}

    # ── Normalise column names ───────────────────────────────────────────────
    df_raw.columns = [str(c).strip().lower().replace(" ", "_") for c in df_raw.columns]

    # NSE CSV columns (left=CE, right=PE around strike_price):
    # oi  chng_in_oi  volume  iv  ltp  net_chng  strike_price  net_chng  ltp  iv  volume  chng_in_oi  oi
    # They appear as duplicated names; pandas suffixes them .1, .2 etc.

    def _col(names):
        for n in names:
            if n in df_raw.columns:
                return n
        return None

    strike_col = _col(["strike_price", "strike"])
    if not strike_col:
        return pd.DataFrame(), {"error": "Strike column not found. Columns found: " + str(list(df_raw.columns))}

    def _safe_float(series):
        return pd.to_numeric(series.astype(str).str.replace(",", "").str.strip(), errors="coerce").fillna(0)

    rows = []
    for _, row in df_raw.iterrows():
        try:
            strike = float(str(row[strike_col]).replace(",", ""))
        except Exception:
            continue
        if strike <= 0:
            continue

        def _v(col_name, fallback=0.0):
            c = _col([col_name])
            return float(_safe_float(pd.Series([row[c]]))[0]) if c else fallback

        # CE: first occurrence (no suffix)
        # PE: second occurrence (.1 suffix added by pandas)
        rows.append({
            "strike":    strike,
            "ce_oi":     _v("oi"),
            "ce_chg_oi": _v("chng_in_oi"),
            "ce_volume": _v("volume"),
            "ce_iv":     _v("iv"),
            "ce_ltp":    _v("ltp"),
            "ce_bid":    0.0,
            "ce_ask":    0.0,
            "pe_oi":     _v("oi.1"),
            "pe_chg_oi": _v("chng_in_oi.1"),
            "pe_volume": _v("volume.1"),
            "pe_iv":     _v("iv.1"),
            "pe_ltp":    _v("ltp.1"),
            "pe_bid":    0.0,
            "pe_ask":    0.0,
        })

    if not rows:
        return pd.DataFrame(), {"error": "No valid strike rows found in CSV."}

    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)

    if underlying == 0.0:
        # fallback: midpoint of strikes
        underlying = float(df["strike"].median())

    atm = float(df.iloc[(df["strike"] - underlying).abs().idxmin()]["strike"])

    meta = {
        "underlying":   underlying,
        "expiry":       expiry,
        "all_expiries": [expiry],
        "atm":          atm,
        "source":       "NSE CSV Upload",
    }
    return df, meta

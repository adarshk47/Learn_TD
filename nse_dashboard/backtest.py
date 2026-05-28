import os, json
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

_FWD_FILE = os.path.join(os.path.dirname(__file__), ".forward_signals.json")
_WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), ".model_weights.json")

INDEX_TOKENS = {
    "NIFTY":      ("NSE", "99926000"),
    "BANKNIFTY":  ("NSE", "99926009"),
    "FINNIFTY":   ("NSE", "99926037"),
    "MIDCPNIFTY": ("NSE", "99926074"),
    "SENSEX":     ("BSE", "1"),
}

# ── Historical candles ────────────────────────────────────────────────────────

def get_historical_candles(smart, symbol, is_index, days=30):
    today = date.today()
    from_d = (today - timedelta(days=days + 15)).strftime("%Y-%m-%d 09:15")
    to_d   = today.strftime("%Y-%m-%d 15:30")
    try:
        if is_index:
            exch, token = INDEX_TOKENS.get(symbol.upper(), ("NSE", "99926000"))
        else:
            return pd.DataFrame()

        resp = smart.getCandleData({
            "exchange":    exch,
            "symboltoken": token,
            "interval":    "ONE_DAY",
            "fromdate":    from_d,
            "todate":      to_d,
        })
        raw = resp.get("data", [])
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=["datetime","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").tail(days).reset_index(drop=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


# ── Indicator computation ─────────────────────────────────────────────────────

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI-14, EMA-9, EMA-21, volume ratio to a candle dataframe.
    Strategy: RSI + dual EMA crossover + volume confirmation (3-signal consensus).
    """
    df = df.copy()

    # EMA trend indicators (exponential weights handle short history better than SMA)
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    # RSI-14 (Wilder smoothing)
    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0.0).rolling(14, min_periods=1).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=1).mean()
    rs    = gain / loss.replace(0, 1e-10)
    df["rsi14"] = 100 - 100 / (1 + rs)

    # Volume ratio vs 5-day average
    vol_avg         = df["volume"].rolling(5, min_periods=1).mean()
    df["vol_ratio"] = (df["volume"] / vol_avg.replace(0, 1)).clip(0, 5)

    # Percentage change
    df["pct_chg"]  = df["close"].pct_change() * 100
    df["next_chg"] = df["pct_chg"].shift(-1)

    return df


def _safe_float(val, default=50.0):
    try:
        v = float(val)
        return default if (v != v) else v  # NaN check
    except Exception:
        return default


# ── Signal generation ─────────────────────────────────────────────────────────

def _make_signal(row, params):
    """
    High-selectivity signal: RSI-14 extreme + EMA-9/21 crossover agree.

    RSI is the primary gate (must be extreme before a signal fires).
    EMA crossover confirms the trend direction.
    On random-walk data this fires ~20-30% of the time (vs 90%+ for SMA alone).
    On trending real data the quality signals fire more consistently.

    min_agree=2: RSI extreme AND EMA cross agree
    min_agree=3: adds requirement that price is also on the correct side of EMA-21
                 (very selective — suitable for daily swing trades)
    """
    rsi   = _safe_float(row.get("rsi14", 50.0))
    ema9  = _safe_float(row.get("ema9",  row["close"]))
    ema21 = _safe_float(row.get("ema21", row["close"]))
    close = float(row["close"])
    vol_ok = _safe_float(row.get("vol_ratio", 1.0)) > params.get("vol_threshold", 1.3)

    rsi_bull = params.get("rsi_bull", 62)
    rsi_bear = params.get("rsi_bear", 38)
    strict   = int(params.get("min_agree", 3)) >= 3

    # RSI extreme is the mandatory gate — without this, AVOID
    rsi_up = rsi > rsi_bull
    rsi_dn = rsi < rsi_bear

    ema_up   = ema9 > ema21
    ema_dn   = ema9 < ema21
    price_up = close > ema21
    price_dn = close < ema21

    if strict:
        # All three agree: RSI extreme + EMA trend + price above/below EMA-21
        bull_signal = rsi_up and ema_up and price_up
        bear_signal = rsi_dn and ema_dn and price_dn
    else:
        # RSI extreme + EMA trend agree (faster, ~30% signal rate)
        bull_signal = rsi_up and ema_up
        bear_signal = rsi_dn and ema_dn

    if bull_signal:
        return "BUY CALL"
    elif bear_signal:
        return "BUY PUT"
    return "AVOID"


# ── Option A: Signal Accuracy ─────────────────────────────────────────────────

def run_accuracy_backtest(candles, params=None):
    if candles.empty or len(candles) < 15:
        return {}, pd.DataFrame()
    if params is None:
        params = load_weights()

    df = _compute_indicators(candles)

    rows = []
    # Start after EMA-21 warm-up period
    for i in range(21, len(df) - 1):
        row    = df.iloc[i]
        signal = _make_signal(row, params)
        nxt    = _safe_float(row.get("next_chg"), 0.0)

        if signal == "BUY CALL":
            correct = nxt > 0.2
        elif signal == "BUY PUT":
            correct = nxt < -0.2
        else:
            correct = abs(nxt) < 0.3

        rows.append({
            "date":     row["datetime"].strftime("%d-%b-%Y"),
            "close":    round(float(row["close"]), 1),
            "rsi14":    round(_safe_float(row.get("rsi14"), 50.0), 1),
            "pct_chg":  round(_safe_float(row.get("pct_chg"), 0.0), 2),
            "signal":   signal,
            "next_chg": round(nxt, 2),
            "correct":  correct,
        })

    if not rows:
        return {}, pd.DataFrame()

    rdf = pd.DataFrame(rows)
    rdf["equity"] = rdf.apply(lambda r: 1 if (r["signal"] != "AVOID" and r["correct"])
                               else (-1 if (r["signal"] != "AVOID" and not r["correct"]) else 0), axis=1).cumsum()

    traded  = rdf[rdf["signal"] != "AVOID"]
    calls   = rdf[rdf["signal"] == "BUY CALL"]
    puts    = rdf[rdf["signal"] == "BUY PUT"]
    avoids  = rdf[rdf["signal"] == "AVOID"]

    def acc(sub): return round(sub["correct"].mean() * 100, 1) if len(sub) else 0

    metrics = {
        "total_signals":  len(rdf),
        "total_traded":   len(traded),
        "total_avoided":  len(avoids),
        "wins":           int(traded["correct"].sum()),
        "losses":         int((~traded["correct"]).sum()),
        "win_rate":       acc(traded),
        "call_accuracy":  acc(calls),
        "put_accuracy":   acc(puts),
        "avoid_accuracy": acc(avoids),
        "signal_rate":    round(len(traded) / max(len(rdf), 1) * 100, 1),
    }
    return metrics, rdf


# ── Option B: P&L Simulation ──────────────────────────────────────────────────

def run_pnl_simulation(candles, lot_size=75, commission=40):
    if candles.empty or len(candles) < 15:
        return {}, pd.DataFrame()

    df     = _compute_indicators(candles)
    params = load_weights()
    rows   = []
    equity = 0
    max_eq = 0
    drawdown = 0

    for i in range(21, len(df) - 1):
        row    = df.iloc[i]
        signal = _make_signal(row, params)
        nxt    = _safe_float(row.get("next_chg"), 0.0)
        premium = float(row["close"]) * 0.008  # ~0.8% ATM premium estimate

        if signal == "BUY CALL":
            if nxt > 0.2:
                pnl    = premium * 1.5 * lot_size - commission
                result = "WIN"
            else:
                pnl    = -premium * 0.4 * lot_size - commission
                result = "LOSS"
        elif signal == "BUY PUT":
            if nxt < -0.2:
                pnl    = premium * 1.5 * lot_size - commission
                result = "WIN"
            else:
                pnl    = -premium * 0.4 * lot_size - commission
                result = "LOSS"
        else:
            pnl    = 0
            result = "SKIP"

        equity   = round(equity + pnl, 0)
        max_eq   = max(max_eq, equity)
        drawdown = min(drawdown, equity - max_eq)

        rows.append({
            "date":     row["datetime"].strftime("%d-%b-%Y"),
            "close":    round(float(row["close"]), 1),
            "rsi14":    round(_safe_float(row.get("rsi14"), 50.0), 1),
            "signal":   signal,
            "next_chg": round(nxt, 2),
            "premium":  round(premium, 1),
            "pnl":      round(pnl, 0),
            "result":   result,
            "equity":   equity,
        })

    if not rows:
        return {}, pd.DataFrame()

    rdf    = pd.DataFrame(rows)
    traded = rdf[rdf["result"] != "SKIP"]
    wins   = rdf[rdf["result"] == "WIN"]

    daily_returns = rdf["pnl"] / 100000
    sharpe = round(daily_returns.mean() / daily_returns.std() * (252**0.5), 2) \
             if daily_returns.std() > 0 else 0

    metrics = {
        "total_trades": len(traded),
        "wins":         len(wins),
        "losses":       len(rdf[rdf["result"] == "LOSS"]),
        "win_rate":     round(len(wins) / len(traded) * 100, 1) if len(traded) else 0,
        "total_pnl":    int(equity),
        "max_drawdown": int(drawdown),
        "avg_pnl":      round(traded["pnl"].mean(), 0) if len(traded) else 0,
        "best_trade":   int(rdf["pnl"].max()),
        "worst_trade":  int(rdf["pnl"].min()),
        "sharpe_ratio": sharpe,
    }
    return metrics, rdf


# ── Option C: Forward Signal Tracker ─────────────────────────────────────────

def save_forward_signal(symbol, signal, confidence, underlying, pcr, max_pain):
    try:
        data = _load_fwd()
        data.append({
            "date":       date.today().isoformat(),
            "symbol":     symbol,
            "signal":     signal,
            "confidence": confidence,
            "underlying": underlying,
            "pcr":        pcr,
            "max_pain":   max_pain,
            "outcome":    "PENDING",
            "next_close": None,
        })
        with open(_FWD_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def _load_fwd():
    if os.path.exists(_FWD_FILE):
        try:
            with open(_FWD_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []

def load_forward_signals():
    return _load_fwd()

def update_forward_outcomes(candles):
    if candles is None or candles.empty:
        return
    data = _load_fwd()
    price_map = {row["datetime"].strftime("%Y-%m-%d"): float(row["close"])
                 for _, row in candles.iterrows()}

    changed = False
    for entry in data:
        if entry["outcome"] != "PENDING":
            continue
        sig_date   = entry["date"]
        next_dates = sorted(d for d in price_map if d > sig_date)
        if not next_dates:
            continue
        next_close = price_map[next_dates[0]]
        entry["next_close"] = next_close
        chg = (next_close - entry["underlying"]) / entry["underlying"] * 100
        if entry["signal"] == "BUY CALL":
            entry["outcome"] = "CORRECT" if chg > 0.2 else "WRONG"
        elif entry["signal"] == "BUY PUT":
            entry["outcome"] = "CORRECT" if chg < -0.2 else "WRONG"
        else:
            entry["outcome"] = "CORRECT" if abs(chg) < 0.3 else "WRONG"
        changed = True

    if changed:
        try:
            with open(_FWD_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass


# ── Weight management ─────────────────────────────────────────────────────────

def load_weights():
    """
    Load signal params. Defaults use RSI-14 + EMA-9/21 consensus strategy.
    rsi_bull / rsi_bear: RSI thresholds for momentum signal
    vol_threshold: volume ratio required for confirmation (relaxed mode only)
    min_agree: 2=RSI+EMA must agree, 3=RSI+EMA+price all must agree (more selective)
    """
    defaults = {
        "rsi_bull":      65,
        "rsi_bear":      35,
        "vol_threshold": 1.3,
        "min_agree":     3,
    }
    if os.path.exists(_WEIGHTS_FILE):
        try:
            with open(_WEIGHTS_FILE) as f:
                saved = json.load(f)
            # Accept saved if it has the new-format key
            if isinstance(saved, dict) and "rsi_bull" in saved:
                defaults.update(saved)
        except Exception:
            pass
    return defaults

def save_weights(weights):
    try:
        with open(_WEIGHTS_FILE, "w") as f:
            json.dump(weights, f, indent=2)
    except Exception:
        pass


# ── Weight calibration (grid search) ─────────────────────────────────────────

def run_weight_calibration(candles):
    """
    Grid search over RSI thresholds, volume threshold and min_agree.
    54 combinations — much faster than the old 576-combo SMA search.
    Works best with real OHLCV data (Angel One); synthetic data will
    still show ~50% because it is a random walk with no real trend.
    """
    if candles.empty or len(candles) < 15:
        return {}, pd.DataFrame()

    grid_results = []
    for rsi_bull in [55, 60, 65]:
        for rsi_bear in [35, 40, 45]:
            for vol_thresh in [1.1, 1.3, 1.5]:
                for min_agree in [2, 3]:
                    params  = {"rsi_bull": rsi_bull, "rsi_bear": rsi_bear,
                               "vol_threshold": vol_thresh, "min_agree": min_agree}
                    metrics, _ = run_accuracy_backtest(candles, params)
                    if metrics:
                        grid_results.append({
                            "rsi_bull":      rsi_bull,
                            "rsi_bear":      rsi_bear,
                            "vol_threshold": vol_thresh,
                            "min_agree":     min_agree,
                            "win_rate":      metrics["win_rate"],
                            "signal_rate":   metrics.get("signal_rate", 0),
                            "trades":        metrics["total_traded"],
                        })

    if not grid_results:
        return {}, pd.DataFrame()

    grid_df = pd.DataFrame(grid_results).sort_values("win_rate", ascending=False)
    best    = grid_df.iloc[0].to_dict()
    best_params = {
        "rsi_bull":      int(best["rsi_bull"]),
        "rsi_bear":      int(best["rsi_bear"]),
        "vol_threshold": float(best["vol_threshold"]),
        "min_agree":     int(best["min_agree"]),
    }
    return {"best_params": best_params, "best_accuracy": best["win_rate"]}, grid_df

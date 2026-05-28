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
            return pd.DataFrame()  # stock historical not yet supported

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
    except Exception as e:
        return pd.DataFrame()

# ── Option A: Signal Accuracy ─────────────────────────────────────────────────

def _make_signal(row, params):
    score = 0
    if row["close"] > row["sma5"]:
        score += params.get("sma_score", 15)
    else:
        score -= params.get("sma_score", 15)
    if row["sma5"] > row["sma10"]:
        score += params.get("trend_score", 10)
    else:
        score -= params.get("trend_score", 10)
    mom_thresh = params.get("mom_threshold", 0.3)
    if row["pct_chg"] > mom_thresh:
        score += params.get("mom_score", 10)
    elif row["pct_chg"] < -mom_thresh:
        score -= params.get("mom_score", 10)
    threshold = params.get("signal_threshold", 15)
    if score > threshold:
        return "BUY CALL"
    elif score < -threshold:
        return "BUY PUT"
    return "AVOID"

def run_accuracy_backtest(candles, params=None):
    if candles.empty or len(candles) < 12:
        return {}, pd.DataFrame()
    if params is None:
        params = load_weights()

    df = candles.copy()
    df["sma5"]     = df["close"].rolling(5).mean()
    df["sma10"]    = df["close"].rolling(10).mean()
    df["pct_chg"]  = df["close"].pct_change() * 100
    df["next_chg"] = df["pct_chg"].shift(-1)

    rows = []
    for i in range(10, len(df) - 1):
        row      = df.iloc[i]
        signal   = _make_signal(row, params)
        nxt      = float(row["next_chg"]) if not pd.isna(row["next_chg"]) else 0.0

        if signal == "BUY CALL":
            correct = nxt > 0.2
        elif signal == "BUY PUT":
            correct = nxt < -0.2
        else:
            correct = abs(nxt) < 0.3

        rows.append({
            "date":     row["datetime"].strftime("%d-%b-%Y"),
            "close":    round(float(row["close"]), 1),
            "pct_chg":  round(float(row["pct_chg"]) if not pd.isna(row["pct_chg"]) else 0, 2),
            "signal":   signal,
            "next_chg": round(nxt, 2),
            "correct":  correct,
        })

    if not rows:
        return {}, pd.DataFrame()

    rdf = pd.DataFrame(rows)
    rdf["equity"] = rdf.apply(lambda r: 1 if (r["signal"] != "AVOID" and r["correct"])
                               else (-1 if (r["signal"] != "AVOID" and not r["correct"]) else 0), axis=1).cumsum()

    traded   = rdf[rdf["signal"] != "AVOID"]
    calls    = rdf[rdf["signal"] == "BUY CALL"]
    puts     = rdf[rdf["signal"] == "BUY PUT"]
    avoids   = rdf[rdf["signal"] == "AVOID"]

    def acc(sub): return round(sub["correct"].mean() * 100, 1) if len(sub) else 0

    metrics = {
        "total_signals":   len(rdf),
        "total_traded":    len(traded),
        "wins":            int(traded["correct"].sum()),
        "losses":          int((~traded["correct"]).sum()),
        "win_rate":        acc(traded),
        "call_accuracy":   acc(calls),
        "put_accuracy":    acc(puts),
        "avoid_accuracy":  acc(avoids),
    }
    return metrics, rdf

# ── Option B: P&L Simulation ──────────────────────────────────────────────────

def run_pnl_simulation(candles, lot_size=75, commission=40):
    if candles.empty or len(candles) < 12:
        return {}, pd.DataFrame()

    df = candles.copy()
    df["sma5"]    = df["close"].rolling(5).mean()
    df["sma10"]   = df["close"].rolling(10).mean()
    df["pct_chg"] = df["close"].pct_change() * 100
    df["next_chg"] = df["pct_chg"].shift(-1)

    params   = load_weights()
    rows     = []
    equity   = 0
    max_eq   = 0
    drawdown = 0

    for i in range(10, len(df) - 1):
        row    = df.iloc[i]
        signal = _make_signal(row, params)
        nxt    = float(row["next_chg"]) if not pd.isna(row["next_chg"]) else 0.0
        premium = float(row["close"]) * 0.008  # 0.8% ATM premium estimate

        if signal == "BUY CALL":
            if nxt > 0.2:
                pnl = premium * 1.5 * lot_size - commission
                result = "WIN"
            else:
                pnl = -premium * 0.4 * lot_size - commission
                result = "LOSS"
        elif signal == "BUY PUT":
            if nxt < -0.2:
                pnl = premium * 1.5 * lot_size - commission
                result = "WIN"
            else:
                pnl = -premium * 0.4 * lot_size - commission
                result = "LOSS"
        else:
            pnl    = 0
            result = "SKIP"

        equity  = round(equity + pnl, 0)
        max_eq  = max(max_eq, equity)
        drawdown = min(drawdown, equity - max_eq)

        rows.append({
            "date":    row["datetime"].strftime("%d-%b-%Y"),
            "close":   round(float(row["close"]), 1),
            "signal":  signal,
            "next_chg": round(nxt, 2),
            "premium":  round(premium, 1),
            "pnl":     round(pnl, 0),
            "result":  result,
            "equity":  equity,
        })

    if not rows:
        return {}, pd.DataFrame()

    rdf    = pd.DataFrame(rows)
    traded = rdf[rdf["result"] != "SKIP"]
    wins   = rdf[rdf["result"] == "WIN"]

    daily_returns = rdf["pnl"] / 100000  # normalise vs 1L capital
    sharpe = round(daily_returns.mean() / daily_returns.std() * (252**0.5), 2) if daily_returns.std() > 0 else 0

    metrics = {
        "total_trades":  len(traded),
        "wins":          len(wins),
        "losses":        len(rdf[rdf["result"] == "LOSS"]),
        "win_rate":      round(len(wins) / len(traded) * 100, 1) if len(traded) else 0,
        "total_pnl":     int(equity),
        "max_drawdown":  int(drawdown),
        "avg_pnl":       round(traded["pnl"].mean(), 0) if len(traded) else 0,
        "best_trade":    int(rdf["pnl"].max()),
        "worst_trade":   int(rdf["pnl"].min()),
        "sharpe_ratio":  sharpe,
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
    """Match saved signals with next-day candle data and mark outcomes."""
    if candles is None or candles.empty:
        return
    data = _load_fwd()
    price_map = {row["datetime"].strftime("%Y-%m-%d"): float(row["close"])
                 for _, row in candles.iterrows()}

    changed = False
    for entry in data:
        if entry["outcome"] != "PENDING":
            continue
        sig_date = entry["date"]
        # Find the next trading date's close
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

# ── Weight calibration ────────────────────────────────────────────────────────

def load_weights():
    defaults = {
        "sma_score": 15, "trend_score": 10, "mom_score": 10,
        "signal_threshold": 15, "mom_threshold": 0.3,
    }
    if os.path.exists(_WEIGHTS_FILE):
        try:
            with open(_WEIGHTS_FILE) as f:
                saved = json.load(f)
            if isinstance(saved, dict) and "sma_score" in saved:
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

def run_weight_calibration(candles):
    if candles.empty or len(candles) < 12:
        return {}, pd.DataFrame()

    grid_results = []
    for sma in [10, 15, 20]:
        for trend in [8, 10, 12]:
            for mom in [8, 10, 12]:
                for thresh in [12, 15, 18]:
                    params = {"sma_score": sma, "trend_score": trend,
                              "mom_score": mom, "signal_threshold": thresh}
                    metrics, _ = run_accuracy_backtest(candles, params)
                    if metrics:
                        grid_results.append({
                            "sma_score": sma, "trend_score": trend,
                            "mom_score": mom, "signal_threshold": thresh,
                            "win_rate":  metrics["win_rate"],
                            "trades":    metrics["total_traded"],
                        })

    if not grid_results:
        return {}, pd.DataFrame()

    grid_df = pd.DataFrame(grid_results).sort_values("win_rate", ascending=False)
    best    = grid_df.iloc[0].to_dict()
    best_params = {k: int(best[k]) for k in ["sma_score","trend_score","mom_score","signal_threshold"]}
    return {"best_params": best_params, "best_accuracy": best["win_rate"]}, grid_df

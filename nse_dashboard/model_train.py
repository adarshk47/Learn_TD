"""
Model training / weight calibration for NSE Options Intelligence dashboard.

Grid-searches over the same 5 weights used by backtest._make_signal() so that
"Apply These Weights" actually changes the win rate shown in the backtest.

Weight schema (must match backtest._make_signal):
    sma_score        — score added when close > SMA5
    trend_score      — score added when SMA5 > SMA10
    mom_score        — score added when daily % change exceeds mom_threshold
    signal_threshold — minimum |score| to issue a BUY CALL / BUY PUT signal
    mom_threshold    — % change magnitude that triggers momentum score
"""

from __future__ import annotations

import json
import os
from itertools import product
from typing import Any

import pandas as pd

# ── File path (shared with backtest.py) ───────────────────────────────────────

_MODEL_WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), ".model_weights.json")

# ── Defaults (must match backtest.load_weights defaults) ─────────────────────

_DEFAULTS: dict[str, Any] = {
    "sma_score":        15,
    "trend_score":      10,
    "mom_score":        10,
    "signal_threshold": 15,
    "mom_threshold":    0.3,
}


# ── Public API ────────────────────────────────────────────────────────────────

def calibrate_weights(candles_df: pd.DataFrame) -> dict:
    """
    Grid-search to find the best combination of signal scoring weights.

    Evaluates each combination using backtest.run_accuracy_backtest() — the
    exact same function used to display win rate — so applying the best params
    will actually change the win rate.

    Search space (4 × 4 × 4 × 3 × 3 = 576 combinations):
        sma_score        : [10, 15, 20, 25]
        trend_score      : [5, 8, 10, 12]
        mom_score        : [5, 8, 10, 12]
        signal_threshold : [10, 12, 15]
        mom_threshold    : [0.2, 0.3, 0.5]

    Returns
    -------
    dict with keys:
        best_params    dict          — weights that maximised win rate
        accuracy_grid  pd.DataFrame  — full grid sorted by accuracy descending
        best_accuracy  float         — best win rate percentage
    """
    from backtest import run_accuracy_backtest

    if candles_df is None or candles_df.empty or len(candles_df) < 12:
        return {
            "best_params":   _DEFAULTS.copy(),
            "accuracy_grid": pd.DataFrame(),
            "best_accuracy": 0.0,
        }

    sma_scores        = [10, 15, 20, 25]
    trend_scores      = [5, 8, 10, 12]
    mom_scores        = [5, 8, 10, 12]
    signal_thresholds = [10, 12, 15]
    mom_thresholds    = [0.2, 0.3, 0.5]

    grid_rows   = []
    best_acc    = -1.0
    best_params = _DEFAULTS.copy()

    for sma_s, trend_s, mom_s, sig_thresh, mom_thresh in product(
        sma_scores, trend_scores, mom_scores, signal_thresholds, mom_thresholds
    ):
        test_weights = {
            "sma_score":        sma_s,
            "trend_score":      trend_s,
            "mom_score":        mom_s,
            "signal_threshold": sig_thresh,
            "mom_threshold":    mom_thresh,
        }

        metrics, _ = run_accuracy_backtest(candles_df, params=test_weights)
        accuracy = metrics.get("win_rate", 0.0) if metrics else 0.0

        # Only record combos that actually generated trades
        total_traded = metrics.get("total_traded", 0) if metrics else 0
        grid_rows.append({
            "sma_score":        sma_s,
            "trend_score":      trend_s,
            "mom_score":        mom_s,
            "signal_threshold": sig_thresh,
            "mom_threshold":    mom_thresh,
            "trades":           total_traded,
            "accuracy":         accuracy,
        })

        if accuracy > best_acc and total_traded >= 3:
            best_acc    = accuracy
            best_params = dict(test_weights)

    accuracy_grid = (
        pd.DataFrame(grid_rows)
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "best_params":   best_params,
        "accuracy_grid": accuracy_grid,
        "best_accuracy": best_acc,
    }


def apply_weights(weights: dict) -> None:
    """Persist calibrated weights to .model_weights.json."""
    try:
        with open(_MODEL_WEIGHTS_FILE, "w") as f:
            json.dump(weights, f, indent=2)
    except Exception:
        pass


def load_weights() -> dict:
    """Load saved weights; falls back to defaults for any missing keys."""
    from backtest import load_weights as bt_load
    return bt_load()


def get_feature_importance(results_df: pd.DataFrame) -> dict:
    """
    Estimate which indicators contribute most to correct signals via
    Pearson correlation between each feature proxy and directional accuracy.
    """
    import math
    import numpy as np

    empty = {
        "momentum":    0.0,
        "score":       0.0,
        "consistency": 0.0,
        "pcr_proxy":   0.0,
        "pain_proxy":  0.0,
    }

    if results_df is None or results_df.empty:
        return empty

    df = results_df.copy()
    if not {"pct_chg", "correct"}.issubset(df.columns):
        return empty

    traded = (
        df[df["signal"].isin(["BUY CALL", "BUY PUT"])]
        if "signal" in df.columns else df
    )
    if len(traded) < 3:
        traded = df

    result = {}

    def _corr(a, b):
        try:
            v = a.corr(b.astype(float))
            return round(v if not math.isnan(v) else 0.0, 4)
        except Exception:
            return 0.0

    result["momentum"]    = _corr(traded["pct_chg"], traded["correct"])
    result["score"]       = _corr(traded.get("score", pd.Series(0, index=traded.index)), traded["correct"])
    try:
        cons = pd.Series(np.sign(traded["pct_chg"].values)).rolling(3).mean().fillna(0.0)
        result["consistency"] = _corr(cons, traded["correct"].reset_index(drop=True))
    except Exception:
        result["consistency"] = 0.0
    result["pcr_proxy"]   = _corr((traded["pct_chg"] > 0).astype(float), traded["correct"])
    result["pain_proxy"]  = _corr(traded["pct_chg"].abs(), traded["correct"])

    return result

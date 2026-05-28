"""
Model weight calibration for NSE Options Intelligence dashboard.

Uses RSI-14 + EMA-9/21 + volume consensus strategy.
Grid-searches over the same params used by backtest._make_signal() so that
"Apply These Weights" actually changes the win rate shown in the backtest.

Weight schema (must match backtest._make_signal):
    rsi_bull      — RSI-14 threshold above which momentum is bullish
    rsi_bear      — RSI-14 threshold below which momentum is bearish
    vol_threshold — volume ratio (current / 5-day avg) needed for confirmation
    min_agree     — how many of 3 indicators must agree (2=balanced, 3=strict)

Note: Win rate on synthetic/random-walk data will always be near 50%.
Use Angel One (Live) data source for meaningful calibration results.
"""

from __future__ import annotations

import json
import os
from itertools import product
from typing import Any

import pandas as pd

_MODEL_WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), ".model_weights.json")

_DEFAULTS: dict[str, Any] = {
    "rsi_bull":      65,
    "rsi_bear":      35,
    "vol_threshold": 1.3,
    "min_agree":     3,
}


def calibrate_weights(candles_df: pd.DataFrame) -> dict:
    """
    Grid-search to find the best RSI/volume params for signal accuracy.

    Search space (3 × 3 × 3 × 2 = 54 combinations):
        rsi_bull      : [55, 60, 65]
        rsi_bear      : [35, 40, 45]
        vol_threshold : [1.1, 1.3, 1.5]
        min_agree     : [2, 3]

    Returns dict with keys: best_params, accuracy_grid, best_accuracy
    """
    from backtest import run_accuracy_backtest

    if candles_df is None or candles_df.empty or len(candles_df) < 15:
        return {
            "best_params":   _DEFAULTS.copy(),
            "accuracy_grid": pd.DataFrame(),
            "best_accuracy": 0.0,
        }

    rsi_bulls      = [55, 60, 65]
    rsi_bears      = [35, 40, 45]
    vol_thresholds = [1.1, 1.3, 1.5]
    min_agrees     = [2, 3]

    grid_rows   = []
    best_acc    = -1.0
    best_params = _DEFAULTS.copy()

    for rsi_bull, rsi_bear, vol_thresh, min_agree in product(
        rsi_bulls, rsi_bears, vol_thresholds, min_agrees
    ):
        test_weights = {
            "rsi_bull":      rsi_bull,
            "rsi_bear":      rsi_bear,
            "vol_threshold": vol_thresh,
            "min_agree":     min_agree,
        }

        metrics, _ = run_accuracy_backtest(candles_df, params=test_weights)
        accuracy      = metrics.get("win_rate", 0.0)   if metrics else 0.0
        total_traded  = metrics.get("total_traded", 0) if metrics else 0
        signal_rate   = metrics.get("signal_rate", 0)  if metrics else 0.0

        grid_rows.append({
            "rsi_bull":      rsi_bull,
            "rsi_bear":      rsi_bear,
            "vol_threshold": vol_thresh,
            "min_agree":     min_agree,
            "trades":        total_traded,
            "signal_rate":   signal_rate,
            "accuracy":      accuracy,
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
    try:
        with open(_MODEL_WEIGHTS_FILE, "w") as f:
            json.dump(weights, f, indent=2)
    except Exception:
        pass


def load_weights() -> dict:
    from backtest import load_weights as bt_load
    return bt_load()


def get_feature_importance(results_df: pd.DataFrame) -> dict:
    import math
    import numpy as np

    empty = {"rsi": 0.0, "ema_trend": 0.0, "volume": 0.0}

    if results_df is None or results_df.empty:
        return empty
    if not {"pct_chg", "correct"}.issubset(results_df.columns):
        return empty

    df     = results_df.copy()
    traded = df[df["signal"].isin(["BUY CALL", "BUY PUT"])] if "signal" in df.columns else df
    if len(traded) < 3:
        traded = df

    def _corr(a, b):
        try:
            v = a.corr(b.astype(float))
            return round(v if not math.isnan(v) else 0.0, 4)
        except Exception:
            return 0.0

    result = {
        "rsi":       _corr(traded.get("rsi14", pd.Series(50, index=traded.index)), traded["correct"]),
        "ema_trend": _corr(traded["pct_chg"], traded["correct"]),
        "volume":    _corr(traded["pct_chg"].abs(), traded["correct"]),
    }
    return result

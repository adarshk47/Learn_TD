"""
Model training / weight calibration for NSE Options Intelligence dashboard.

Provides grid-search calibration of signal scoring weights based on historical
candle data from backtest.py, and utilities to persist / load those weights.
"""

from __future__ import annotations

import json
import math
import os
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

# ── File paths ────────────────────────────────────────────────────────────────

_MODEL_WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), ".model_weights.json")


# ── Default weights ───────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "pcr_bull_score":      25,
    "pcr_bear_score":     -25,
    "pcr_bull_threshold":  1.2,
    "pcr_bear_threshold":  0.5,
    "pain_score":          20,
    "pain_threshold_pct":  0.3,
    "atm_oi_score":        15,
}


# ── Internal signal simulator ─────────────────────────────────────────────────

def _simulate_signal(
    prev_close: float,
    close: float,
    sma5: float,
    momentum: float,
    weights: dict,
) -> str:
    """
    Reproduce the signal logic from backtest._signal_from_candles_row
    using a given weight set. Returns "BUY CALL", "BUY PUT", or "AVOID".
    """
    score = 0.0

    # SMA5 cross contribution
    if sma5 and sma5 > 0:
        gap_pct = (close - sma5) / sma5 * 100
        if gap_pct > 0.3:
            score += weights.get("pcr_bull_score", 25) * 0.6
        elif gap_pct < -0.3:
            score += weights.get("pcr_bear_score", -25) * 0.6

    # Momentum contribution
    if momentum > 1.0:
        score += weights.get("pain_score", 20) * 0.7
    elif momentum < -1.0:
        score -= weights.get("pain_score", 20) * 0.7

    # Previous day change contribution
    if prev_close and prev_close > 0:
        day_chg = (close - prev_close) / prev_close * 100
        if day_chg > 0.5:
            score += weights.get("atm_oi_score", 15) * 0.4
        elif day_chg < -0.5:
            score -= weights.get("atm_oi_score", 15) * 0.4

    if score > 15:
        return "BUY CALL"
    elif score < -15:
        return "BUY PUT"
    else:
        return "AVOID"


def _evaluate_weights(candles_df: pd.DataFrame, weights: dict) -> float:
    """
    Simulate signals on the full candle history using the given weights.
    Returns overall accuracy (fraction of correct directional calls, 0–100).
    """
    df = candles_df.copy().reset_index(drop=True)
    df["sma5"]     = df["close"].rolling(5).mean()
    df["momentum"] = df["close"].pct_change(3) * 100

    correct_count = 0
    total_count   = 0

    for i in range(5, len(df) - 1):
        row       = df.iloc[i]
        prev_row  = df.iloc[i - 1]
        next_row  = df.iloc[i + 1]

        close      = row["close"]
        sma5_val   = row["sma5"]
        momentum   = row["momentum"] if not (isinstance(row["momentum"], float) and math.isnan(row["momentum"])) else 0.0
        prev_close = prev_row["close"]
        next_close = next_row["close"]

        signal = _simulate_signal(prev_close, close, sma5_val, momentum, weights)

        if signal == "AVOID":
            continue

        next_chg = (next_close - close) / close * 100 if close else 0.0
        correct = (signal == "BUY CALL" and next_chg > 0) or (signal == "BUY PUT" and next_chg < 0)

        total_count   += 1
        correct_count += int(correct)

    return round(correct_count / total_count * 100, 2) if total_count > 0 else 0.0


# ── Public API ────────────────────────────────────────────────────────────────

def calibrate_weights(candles_df: pd.DataFrame) -> dict:
    """
    Grid-search to find the best combination of signal scoring weights.

    Evaluates each parameter combination on the supplied candle history and
    returns the combination that produces the highest signal accuracy.

    Parameters
    ----------
    candles_df : pd.DataFrame
        Output of backtest.get_historical_candles — must have columns:
        datetime, open, high, low, close, volume.

    Search space
    ------------
    pcr_bull_score        : [15, 20, 25, 30]
    pcr_bear_score        : [-15, -20, -25, -30]
    pcr_bull_threshold    : [1.0, 1.1, 1.2, 1.3]
    pain_score            : [10, 15, 20]
    pain_threshold_pct    : [0.2, 0.3, 0.5]

    Returns
    -------
    dict with keys:
        best_params    dict        — weight values that maximised accuracy
        accuracy_grid  pd.DataFrame — full grid with accuracy per combination
        best_accuracy  float        — best accuracy percentage
    """
    if candles_df is None or candles_df.empty or len(candles_df) < 10:
        return {
            "best_params":   _DEFAULTS.copy(),
            "accuracy_grid": pd.DataFrame(),
            "best_accuracy": 0.0,
        }

    pcr_bull_scores     = [15, 20, 25, 30]
    pcr_bear_scores     = [-15, -20, -25, -30]
    pcr_bull_thresholds = [1.0, 1.1, 1.2, 1.3]
    pain_scores         = [10, 15, 20]
    pain_thresholds     = [0.2, 0.3, 0.5]

    grid_rows   = []
    best_acc    = -1.0
    best_params = _DEFAULTS.copy()

    for bull_s, bear_s, pcr_thresh, pain_s, pain_thresh in product(
        pcr_bull_scores,
        pcr_bear_scores,
        pcr_bull_thresholds,
        pain_scores,
        pain_thresholds,
    ):
        test_weights = {
            "pcr_bull_score":      bull_s,
            "pcr_bear_score":      bear_s,
            "pcr_bull_threshold":  pcr_thresh,
            "pcr_bear_threshold":  round(1.0 - pcr_thresh + 0.5, 2),
            "pain_score":          pain_s,
            "pain_threshold_pct":  pain_thresh,
            "atm_oi_score":        15,  # held constant in grid search
        }

        accuracy = _evaluate_weights(candles_df, test_weights)

        grid_rows.append({
            "pcr_bull_score":     bull_s,
            "pcr_bear_score":     bear_s,
            "pcr_bull_threshold": pcr_thresh,
            "pain_score":         pain_s,
            "pain_threshold_pct": pain_thresh,
            "accuracy":           accuracy,
        })

        if accuracy > best_acc:
            best_acc    = accuracy
            best_params = dict(test_weights)

    accuracy_grid = pd.DataFrame(grid_rows).sort_values("accuracy", ascending=False).reset_index(drop=True)

    return {
        "best_params":   best_params,
        "accuracy_grid": accuracy_grid,
        "best_accuracy": best_acc,
    }


def apply_weights(weights: dict) -> None:
    """
    Persist calibrated weights to .model_weights.json.

    Parameters
    ----------
    weights : dict — typically the ``best_params`` from calibrate_weights()
    """
    try:
        with open(_MODEL_WEIGHTS_FILE, "w") as f:
            json.dump(weights, f, indent=2)
    except Exception:
        pass


def load_weights() -> dict:
    """
    Load saved weights from .model_weights.json.

    Returns default weights if the file doesn't exist or cannot be parsed.
    Missing keys are filled from the defaults so the returned dict is always
    complete.

    Returns
    -------
    dict — complete weight configuration
    """
    defaults = _DEFAULTS.copy()

    if os.path.exists(_MODEL_WEIGHTS_FILE):
        try:
            with open(_MODEL_WEIGHTS_FILE) as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                # Merge: saved values override defaults, but missing keys are filled
                defaults.update(saved)
                return defaults
        except Exception:
            pass

    return defaults


def get_feature_importance(results_df: pd.DataFrame) -> dict:
    """
    Estimate which indicators contribute most to correct signals.

    Uses simple Pearson correlation between each feature proxy and whether the
    next-day direction matched the signal.

    Expected columns in results_df (from backtest.run_accuracy_backtest):
        date, close, pct_chg, signal, score, next_chg, correct, equity_curve

    The function derives feature proxies from the available columns:
        - momentum    : pct_chg (today's % change)
        - score       : overall signal score (direct)
        - consistency : rolling sign-agreement of pct_chg

    Parameters
    ----------
    results_df : pd.DataFrame — output of backtest.run_accuracy_backtest()[1]

    Returns
    -------
    dict — {feature_name: correlation_with_correct_calls}
             Higher absolute value → more predictive.
             Values are Pearson r, range [-1, +1].
    """
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

    required = {"pct_chg", "correct"}
    if not required.issubset(df.columns):
        return empty

    # Only consider traded signals (not AVOID) for directional accuracy
    traded = df[df.get("signal", pd.Series(dtype=str)).isin(["BUY CALL", "BUY PUT"])] \
        if "signal" in df.columns else df

    if len(traded) < 3:
        # Fall back to full df
        traded = df

    result = {}

    # ── Momentum (today's % change) ───────────────────────────────────────────
    try:
        corr_momentum = traded["pct_chg"].corr(traded["correct"].astype(float))
        result["momentum"] = round(corr_momentum if not math.isnan(corr_momentum) else 0.0, 4)
    except Exception:
        result["momentum"] = 0.0

    # ── Score (if column exists) ──────────────────────────────────────────────
    if "score" in traded.columns:
        try:
            corr_score = traded["score"].corr(traded["correct"].astype(float))
            result["score"] = round(corr_score if not math.isnan(corr_score) else 0.0, 4)
        except Exception:
            result["score"] = 0.0
    else:
        result["score"] = 0.0

    # ── Consistency: 3-day rolling sign agreement ─────────────────────────────
    try:
        sign_series    = np.sign(traded["pct_chg"].values)
        consistency    = pd.Series(sign_series).rolling(3).mean().fillna(0.0)
        corr_cons      = consistency.corr(traded["correct"].reset_index(drop=True).astype(float))
        result["consistency"] = round(corr_cons if not math.isnan(corr_cons) else 0.0, 4)
    except Exception:
        result["consistency"] = 0.0

    # ── PCR proxy: use pct_chg > 0 as crude PCR proxy (bullish day → PCR likely high) ──
    try:
        pcr_proxy = (traded["pct_chg"] > 0).astype(float)
        corr_pcr  = pcr_proxy.corr(traded["correct"].astype(float))
        result["pcr_proxy"] = round(corr_pcr if not math.isnan(corr_pcr) else 0.0, 4)
    except Exception:
        result["pcr_proxy"] = 0.0

    # ── Max pain proxy: use abs(pct_chg) as mean-reversion indicator ─────────
    try:
        pain_proxy = traded["pct_chg"].abs()
        corr_pain  = pain_proxy.corr(traded["correct"].astype(float))
        result["pain_proxy"] = round(corr_pain if not math.isnan(corr_pain) else 0.0, 4)
    except Exception:
        result["pain_proxy"] = 0.0

    return result

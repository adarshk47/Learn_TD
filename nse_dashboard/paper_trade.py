"""
Paper Trading module for NSE Options Intelligence dashboard.

Provides session_state-based portfolio management with JSON export.
All state is stored in Streamlit session_state — no database required.
"""

import json
import os
from datetime import datetime

import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 100_000  # Rs. 1 lakh

LOT_SIZES = {
    "NIFTY":      75,
    "BANKNIFTY":  30,
    "FINNIFTY":   40,
    "MIDCPNIFTY": 50,
    "SENSEX":     20,
    "DEFAULT":    100,
}

_TRADES_FILE = os.path.join(os.path.dirname(__file__), ".paper_trades.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_lot_size(symbol: str) -> int:
    """Return the standard lot size for a given symbol."""
    return LOT_SIZES.get(symbol.upper(), LOT_SIZES["DEFAULT"])


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Portfolio lifecycle ───────────────────────────────────────────────────────

def init_portfolio(session_state) -> None:
    """
    Initialise paper-trading keys in session_state if they don't exist yet.

    Keys created:
        pt_capital   float  — current cash balance
        pt_trades    list   — list of trade dicts
        pt_next_id   int    — monotonically increasing trade ID counter
    """
    if "pt_capital" not in session_state:
        session_state.pt_capital = float(INITIAL_CAPITAL)
    if "pt_trades" not in session_state:
        # Try to restore from persistent JSON on first load
        session_state.pt_trades = _load_trades_from_file()
        if session_state.pt_trades:
            # Recompute capital from closed trades
            spent = sum(
                t["entry_price"] * t["lots"] * t["lot_size"]
                for t in session_state.pt_trades
                if t["status"] == "OPEN"
            )
            realised = sum(
                t.get("pnl", 0.0)
                for t in session_state.pt_trades
                if t["status"] == "CLOSED"
            )
            session_state.pt_capital = float(INITIAL_CAPITAL) - spent + realised
        else:
            session_state.pt_trades = []
    if "pt_next_id" not in session_state:
        existing_ids = [t.get("id", 0) for t in session_state.pt_trades]
        session_state.pt_next_id = max(existing_ids, default=0) + 1


def reset_portfolio(session_state) -> None:
    """Reset portfolio to initial state, clearing all trades."""
    session_state.pt_capital  = float(INITIAL_CAPITAL)
    session_state.pt_trades   = []
    session_state.pt_next_id  = 1
    # Also wipe the persistent file
    try:
        if os.path.exists(_TRADES_FILE):
            os.remove(_TRADES_FILE)
    except Exception:
        pass


# ── Trade management ──────────────────────────────────────────────────────────

def place_trade(
    session_state,
    symbol: str,
    trade_type: str,
    strike: float,
    expiry: str,
    lots: int,
    entry_price: float,
) -> tuple:
    """
    Open a new paper trade.

    Parameters
    ----------
    session_state : Streamlit session_state
    symbol        : e.g. "NIFTY"
    trade_type    : "BUY CALL" or "BUY PUT"
    strike        : option strike price
    expiry        : expiry date string (display only)
    lots          : number of lots
    entry_price   : per-unit option premium

    Returns
    -------
    (success: bool, message: str)
    """
    init_portfolio(session_state)

    if trade_type not in ("BUY CALL", "BUY PUT"):
        return False, "trade_type must be 'BUY CALL' or 'BUY PUT'"

    if lots <= 0:
        return False, "Lots must be a positive integer"

    if entry_price <= 0:
        return False, "Entry price must be positive"

    lot_size = get_lot_size(symbol)
    cost     = entry_price * lots * lot_size

    if cost > session_state.pt_capital:
        return (
            False,
            "Insufficient capital. Need ₹{:,.2f}, available ₹{:,.2f}".format(
                cost, session_state.pt_capital
            ),
        )

    trade = {
        "id":          session_state.pt_next_id,
        "symbol":      symbol.upper(),
        "type":        trade_type,
        "strike":      strike,
        "expiry":      expiry,
        "lots":        lots,
        "lot_size":    lot_size,
        "entry_price": entry_price,
        "entry_time":  _now_str(),
        "status":      "OPEN",
        "exit_price":  None,
        "exit_time":   None,
        "pnl":         0.0,
    }

    session_state.pt_trades.append(trade)
    session_state.pt_capital  -= cost
    session_state.pt_next_id  += 1
    _save_trades_to_file(session_state.pt_trades)

    return (
        True,
        "Trade #{} opened: {} {} {}CE/PE @ ₹{:.2f} x {} lots. "
        "Cost ₹{:,.2f}. Capital remaining ₹{:,.2f}".format(
            trade["id"], trade_type, symbol, int(strike),
            entry_price, lots, cost, session_state.pt_capital,
        ),
    )


def close_trade(session_state, trade_id: int, exit_price: float) -> tuple:
    """
    Close an open paper trade by ID.

    P&L calculation:
        BUY CALL : pnl = (exit_price - entry_price) * lots * lot_size
        BUY PUT  : pnl = (entry_price - exit_price) * lots * lot_size

    Parameters
    ----------
    session_state : Streamlit session_state
    trade_id      : int — trade ID (from trade["id"])
    exit_price    : float — per-unit exit premium

    Returns
    -------
    (success: bool, message: str)
    """
    init_portfolio(session_state)

    if exit_price <= 0:
        return False, "Exit price must be positive"

    trade = next(
        (t for t in session_state.pt_trades if t["id"] == trade_id), None
    )
    if trade is None:
        return False, "Trade #{} not found".format(trade_id)

    if trade["status"] == "CLOSED":
        return False, "Trade #{} is already closed".format(trade_id)

    lots      = trade["lots"]
    lot_size  = trade["lot_size"]
    entry     = trade["entry_price"]

    if trade["type"] == "BUY CALL":
        pnl = (exit_price - entry) * lots * lot_size
    else:  # BUY PUT — profit when price falls
        pnl = (entry - exit_price) * lots * lot_size

    pnl = round(pnl, 2)

    # Return original cost + profit/loss to capital
    original_cost = entry * lots * lot_size
    session_state.pt_capital += original_cost + pnl

    trade["status"]     = "CLOSED"
    trade["exit_price"] = exit_price
    trade["exit_time"]  = _now_str()
    trade["pnl"]        = pnl

    _save_trades_to_file(session_state.pt_trades)

    direction = "PROFIT" if pnl >= 0 else "LOSS"
    return (
        True,
        "Trade #{} closed @ ₹{:.2f}. {}: ₹{:,.2f}. "
        "Capital now ₹{:,.2f}".format(
            trade_id, exit_price, direction, abs(pnl),
            session_state.pt_capital,
        ),
    )


# ── Portfolio summary ─────────────────────────────────────────────────────────

def get_summary(session_state, live_df=None, underlying: float = None) -> dict:
    """
    Return a snapshot of the paper portfolio.

    Parameters
    ----------
    session_state : Streamlit session_state
    live_df       : optional — current option chain DataFrame (for unrealised P&L)
    underlying    : optional — current spot price (for context)

    Returns
    -------
    dict with keys:
        capital          float   — current cash
        initial          float   — starting capital constant
        open_trades      list    — list of OPEN trade dicts
        closed_trades    list    — list of CLOSED trade dicts
        realized_pnl     float   — sum of closed trade P&Ls
        unrealized_pnl   float   — estimated from live_df LTPs
        total_pnl        float   — realized + unrealized
        total_return_pct float   — total P&L as % of initial capital
    """
    init_portfolio(session_state)

    open_trades   = [t for t in session_state.pt_trades if t["status"] == "OPEN"]
    closed_trades = [t for t in session_state.pt_trades if t["status"] == "CLOSED"]

    realized_pnl = sum(t.get("pnl", 0.0) for t in closed_trades)

    # Estimate unrealised P&L from live option chain
    unrealized_pnl = 0.0
    if live_df is not None and not live_df.empty and open_trades:
        for trade in open_trades:
            strike = trade["strike"]
            # Find nearest matching strike in live_df
            try:
                idx         = (live_df["strike"] - strike).abs().idxmin()
                matched_row = live_df.iloc[idx]
                if trade["type"] == "BUY CALL":
                    current_ltp = float(matched_row.get("ce_ltp", trade["entry_price"]))
                    unrealized_pnl += (current_ltp - trade["entry_price"]) * trade["lots"] * trade["lot_size"]
                else:  # BUY PUT
                    current_ltp = float(matched_row.get("pe_ltp", trade["entry_price"]))
                    unrealized_pnl += (trade["entry_price"] - current_ltp) * trade["lots"] * trade["lot_size"]
            except Exception:
                pass

    unrealized_pnl = round(unrealized_pnl, 2)
    total_pnl      = round(realized_pnl + unrealized_pnl, 2)
    total_return   = round(total_pnl / INITIAL_CAPITAL * 100, 2)

    return {
        "capital":          round(session_state.pt_capital, 2),
        "initial":          INITIAL_CAPITAL,
        "open_trades":      open_trades,
        "closed_trades":    closed_trades,
        "realized_pnl":     round(realized_pnl, 2),
        "unrealized_pnl":   unrealized_pnl,
        "total_pnl":        total_pnl,
        "total_return_pct": total_return,
    }


# ── CSV export ────────────────────────────────────────────────────────────────

def export_to_csv(session_state) -> str:
    """
    Serialise all trades to CSV string.

    Returns
    -------
    str — CSV text (UTF-8). Empty string if no trades exist.
    """
    init_portfolio(session_state)

    if not session_state.pt_trades:
        return ""

    rows = []
    for t in session_state.pt_trades:
        rows.append({
            "ID":          t.get("id"),
            "Symbol":      t.get("symbol"),
            "Type":        t.get("type"),
            "Strike":      t.get("strike"),
            "Expiry":      t.get("expiry"),
            "Lots":        t.get("lots"),
            "Lot Size":    t.get("lot_size"),
            "Entry Price": t.get("entry_price"),
            "Entry Time":  t.get("entry_time"),
            "Exit Price":  t.get("exit_price", ""),
            "Exit Time":   t.get("exit_time", ""),
            "P&L":         t.get("pnl", 0.0),
            "Status":      t.get("status"),
        })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


# ── Persistence helpers ───────────────────────────────────────────────────────

def _save_trades_to_file(trades: list) -> None:
    """Persist trades list to JSON file for session restore."""
    try:
        with open(_TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2)
    except Exception:
        pass


def _load_trades_from_file() -> list:
    """Load trades from JSON file if it exists."""
    if os.path.exists(_TRADES_FILE):
        try:
            with open(_TRADES_FILE) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []

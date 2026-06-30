"""Trading strategy entry point for the Algothon competition.

Implement your strategy in `getMyPosition()` — eval.py calls it once per day
with the full price history up to that day and trades the position delta.
"""

import numpy as np

# --- Strategy parameters ---
N_INSTRUMENTS = 50
MIN_HISTORY_DAYS = 2  # need at least two days to compute a return
TRADE_BUDGET_PER_SIGNAL = 5_000  # target dollar exposure per unit of signal

# Running positions across daily calls (eval trades the delta each day).
current_pos = np.zeros(N_INSTRUMENTS)


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Log return of each instrument from the previous day to the latest day."""
    return np.log(prices[:, -1] / prices[:, -2])


def normalize_l2(vector: np.ndarray) -> np.ndarray:
    """Scale a vector to unit L2 norm; returns zeros if the norm is zero."""
    norm = np.sqrt(vector.dot(vector))
    if norm == 0:
        return vector
    return vector / norm


def signal_to_share_deltas(
    signal: np.ndarray,
    latest_prices: np.ndarray,
    dollar_budget: float,
) -> np.ndarray:
    """Convert a unit signal into integer share deltas at the latest prices."""
    return np.array([int(dollars / price) for dollars, price in zip(dollar_budget * signal, latest_prices)])


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """Return desired integer positions for all instruments.

    Called once per simulated day with price history of shape (n_instruments, n_days).
    Positions are cumulative: each call may adjust `current_pos`, and eval.py
    executes only the difference from the previous day's positions.
    """
    global current_pos

    n_instruments, n_days = prcSoFar.shape
    if n_days < MIN_HISTORY_DAYS:
        return np.zeros(n_instruments)

    # Rank instruments by yesterday's return, then spread a fixed budget
    # proportionally across the normalised return vector.
    last_returns = compute_log_returns(prcSoFar)
    normalized_signal = normalize_l2(last_returns)
    share_deltas = signal_to_share_deltas(
        normalized_signal, prcSoFar[:, -1], TRADE_BUDGET_PER_SIGNAL
    )

    current_pos = np.array([int(shares) for shares in current_pos + share_deltas])
    return current_pos

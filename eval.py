#!/usr/bin/env python
"""Backtest harness for evaluating trading strategies."""

import numpy as np
import pandas as pd
from main import getMyPosition as get_position

# --- Simulation parameters (see README for competition rules) ---
COMMISSION_RATE = 0.0005  # 5 bps charged on total dollar volume traded
DOLLAR_POSITION_LIMIT = 10_000  # max long/short exposure per instrument at trade time
DEFAULT_PRICES_FILE = "./prices.txt"
DEFAULT_TEST_DAYS = 200
TRADING_DAYS_PER_YEAR = 249  # used for annualised Sharpe ratio
SCORE_RISK_PENALTY = 0.1  # score = mean(PL) - penalty * std(PL)


def load_prices(filepath: str) -> tuple[np.ndarray, int, int]:
    """Load daily prices from a whitespace-separated file.

    The file stores one row per day and one column per instrument.
    Returns prices as (n_instruments, n_days) and the shape dimensions.
    """
    df = pd.read_csv(filepath, sep=r"\s+", header=None, index_col=None)
    n_days, n_instruments = df.shape
    prices = df.values.T
    return prices, n_instruments, n_days


def clip_to_position_limits(
    target_positions: np.ndarray,
    current_prices: np.ndarray,
    dollar_limit: float,
) -> np.ndarray:
    """Enforce per-instrument dollar exposure limits at the current price."""
    share_limits = np.array([int(dollar_limit / price) for price in current_prices])
    return np.clip(target_positions, -share_limits, share_limits)


def execute_rebalance(
    current_positions: np.ndarray,
    target_positions: np.ndarray,
    current_prices: np.ndarray,
    commission_rate: float,
) -> tuple[float, float]:
    """Trade from current to target positions; return cash delta and dollar volume."""
    delta_positions = target_positions - current_positions
    dollar_volumes = current_prices * np.abs(delta_positions)
    total_dollar_volume = float(np.sum(dollar_volumes))
    commission = total_dollar_volume * commission_rate
    cash_delta = -float(current_prices.dot(delta_positions)) - commission
    return cash_delta, total_dollar_volume


def compute_portfolio_value(
    cash: float,
    positions: np.ndarray,
    current_prices: np.ndarray,
) -> float:
    """Mark-to-market portfolio value (cash plus position value)."""
    return cash + float(positions.dot(current_prices))


def annualised_sharpe(daily_pl: np.ndarray) -> float:
    """Annualised Sharpe ratio from a series of daily P&L observations."""
    pl_std = float(np.std(daily_pl))
    if pl_std == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * np.mean(daily_pl) / pl_std)


def run_backtest(
    price_history: np.ndarray,
    num_test_days: int,
    get_position_fn=get_position,
    commission_rate: float = COMMISSION_RATE,
    dollar_limit: float = DOLLAR_POSITION_LIMIT,
    verbose: bool = True,
) -> dict:
    """Simulate strategy over the last `num_test_days` of price history.

    On each day except the final one, the strategy proposes new positions,
    trades are executed at the day's close, and P&L is tracked. The last day
    only marks positions to market (no new trades).
    """
    _, n_days = price_history.shape
    start_day = n_days + 1 - num_test_days

    cash = 0.0
    positions = np.zeros(price_history.shape[0])
    portfolio_value = 0.0
    total_dollar_volume = 0.0
    daily_pl = []

    for day in range(start_day, n_days + 1):
        prices_so_far = price_history[:, :day]
        current_prices = prices_so_far[:, -1]

        if day < n_days:
            # Trade through close; skip execution on the final evaluation day.
            requested_positions = get_position_fn(prices_so_far)
            target_positions = clip_to_position_limits(
                requested_positions, current_prices, dollar_limit
            )
            cash_delta, traded_volume = execute_rebalance(
                positions, target_positions, current_prices, commission_rate
            )
            cash += cash_delta
            total_dollar_volume += traded_volume
        else:
            target_positions = positions.copy()

        positions = np.array(target_positions)
        new_portfolio_value = compute_portfolio_value(cash, positions, current_prices)
        today_pl = new_portfolio_value - portfolio_value
        portfolio_value = new_portfolio_value

        cumulative_return = 0.0
        if total_dollar_volume > 0:
            cumulative_return = portfolio_value / total_dollar_volume

        if day > start_day:
            if verbose:
                print(
                    "Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf"
                    % (day, portfolio_value, today_pl, total_dollar_volume, cumulative_return)
                )
            daily_pl.append(today_pl)

    daily_pl = np.array(daily_pl)
    pl_mean = float(np.mean(daily_pl))
    pl_std = float(np.std(daily_pl))
    sharpe = annualised_sharpe(daily_pl)
    score = pl_mean - SCORE_RISK_PENALTY * pl_std

    return {
        "mean_pl": pl_mean,
        "return": cumulative_return,
        "pl_std": pl_std,
        "sharpe": sharpe,
        "total_dollar_volume": total_dollar_volume,
        "score": score,
    }


def print_results(results: dict) -> None:
    """Print a summary of backtest metrics."""
    print("=====")
    print("mean(PL): %.1lf" % results["mean_pl"])
    print("return: %.5lf" % results["return"])
    print("StdDev(PL): %.2lf" % results["pl_std"])
    print("annSharpe(PL): %.2lf " % results["sharpe"])
    print("totDvolume: %.0lf " % results["total_dollar_volume"])
    print("Score: %.2lf" % results["score"])


def main() -> None:
    prices, n_instruments, n_days = load_prices(DEFAULT_PRICES_FILE)
    print("Loaded %d instruments for %d days" % (n_instruments, n_days))

    results = run_backtest(prices, DEFAULT_TEST_DAYS)
    print_results(results)


if __name__ == "__main__":
    main()

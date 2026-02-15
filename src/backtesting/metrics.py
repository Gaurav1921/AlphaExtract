"""
Backtesting Metrics
--------------------
Calculates accuracy, precision, returns, and risk-adjusted performance
from backtest results.
"""

import math
from typing import Dict, List, Optional


def hit_rate(results: List[Dict]) -> float:
    """
    Percentage of signals that correctly predicted market direction.

    A signal is a "hit" if:
      - Signal is BUY/STRONG_BUY and market went up
      - Signal is SELL/STRONG_SELL and market went down
      - Signal is HOLD and market was flat
    """
    if not results:
        return 0.0

    hits = 0
    scorable = 0

    for r in results:
        direction = r.get("actual_direction")
        signal = r.get("signal", "HOLD")
        if direction is None or direction == "unknown":
            continue

        scorable += 1
        if signal in ("STRONG_BUY", "BUY") and direction == "up":
            hits += 1
        elif signal in ("STRONG_SELL", "SELL") and direction == "down":
            hits += 1
        elif signal == "HOLD" and direction == "flat":
            hits += 1

    return hits / scorable if scorable else 0.0


def directional_accuracy(results: List[Dict]) -> float:
    """
    Looser metric: ignores HOLD/flat — did bullish/bearish calls get direction right?
    """
    if not results:
        return 0.0

    hits = 0
    scorable = 0

    for r in results:
        direction = r.get("actual_direction")
        signal = r.get("signal", "HOLD")
        if direction in (None, "unknown", "flat") or signal == "HOLD":
            continue

        scorable += 1
        if signal in ("STRONG_BUY", "BUY") and direction == "up":
            hits += 1
        elif signal in ("STRONG_SELL", "SELL") and direction == "down":
            hits += 1

    return hits / scorable if scorable else 0.0


def precision_by_signal(results: List[Dict]) -> Dict[str, Dict]:
    """
    For each signal type: how often was it correct, and how many times was it issued.
    """
    signal_stats = {}

    for r in results:
        signal = r.get("signal", "HOLD")
        direction = r.get("actual_direction")
        if direction in (None, "unknown"):
            continue

        if signal not in signal_stats:
            signal_stats[signal] = {"correct": 0, "total": 0}

        signal_stats[signal]["total"] += 1
        is_correct = (
            (signal in ("STRONG_BUY", "BUY") and direction == "up")
            or (signal in ("STRONG_SELL", "SELL") and direction == "down")
            or (signal == "HOLD" and direction == "flat")
        )
        if is_correct:
            signal_stats[signal]["correct"] += 1

    for signal, stats in signal_stats.items():
        stats["precision"] = stats["correct"] / stats["total"] if stats["total"] else 0.0

    return signal_stats


def avg_return_by_signal(results: List[Dict]) -> Dict[str, Dict]:
    """Average actual market return grouped by signal type."""
    buckets: Dict[str, List[float]] = {}

    for r in results:
        signal = r.get("signal", "HOLD")
        ret = r.get("actual_return_pct")
        if ret is None:
            continue
        buckets.setdefault(signal, []).append(ret)

    output = {}
    for signal, returns in buckets.items():
        output[signal] = {
            "avg_return_pct": round(sum(returns) / len(returns), 2),
            "min_return_pct": round(min(returns), 2),
            "max_return_pct": round(max(returns), 2),
            "count": len(returns),
        }

    return output


def sharpe_ratio(results: List[Dict], risk_free_annual: float = 0.05) -> Optional[float]:
    """
    Simplified Sharpe ratio assuming equal position sizing per signal.

    Uses the actual return as the 'strategy return' for each signal period.
    """
    returns = [r["actual_return_pct"] / 100 for r in results if r.get("actual_return_pct") is not None]
    if len(returns) < 2:
        return None

    avg_return = sum(returns) / len(returns)

    # Per-period risk-free (assume 90-day window, annualize)
    risk_free_period = risk_free_annual / 4
    excess_returns = [r - risk_free_period for r in returns]
    avg_excess = sum(excess_returns) / len(excess_returns)

    variance = sum((r - avg_excess) ** 2 for r in excess_returns) / (len(excess_returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.001

    # Annualize: multiply by sqrt(4) since ~4 filing periods per year
    return round((avg_excess / std_dev) * 2, 3)


def confusion_matrix(results: List[Dict]) -> Dict:
    """
    Build a confusion matrix: predicted direction vs actual direction.

    Simplifies to 3x3: predicted {bullish, neutral, bearish} × actual {up, flat, down}.
    """
    matrix = {
        "bullish": {"up": 0, "flat": 0, "down": 0},
        "neutral": {"up": 0, "flat": 0, "down": 0},
        "bearish": {"up": 0, "flat": 0, "down": 0},
    }

    for r in results:
        signal = r.get("signal", "HOLD")
        direction = r.get("actual_direction")
        if direction in (None, "unknown"):
            continue

        if signal in ("STRONG_BUY", "BUY"):
            predicted = "bullish"
        elif signal in ("STRONG_SELL", "SELL"):
            predicted = "bearish"
        else:
            predicted = "neutral"

        matrix[predicted][direction] += 1

    return matrix


def compare_models(results_a: List[Dict], results_b: List[Dict], label_a: str = "Model A", label_b: str = "Model B") -> Dict:
    """Side-by-side comparison of two model results."""
    return {
        label_a: {
            "hit_rate": round(hit_rate(results_a), 4),
            "directional_accuracy": round(directional_accuracy(results_a), 4),
            "sharpe": sharpe_ratio(results_a),
            "avg_return_by_signal": avg_return_by_signal(results_a),
            "n": len(results_a),
        },
        label_b: {
            "hit_rate": round(hit_rate(results_b), 4),
            "directional_accuracy": round(directional_accuracy(results_b), 4),
            "sharpe": sharpe_ratio(results_b),
            "avg_return_by_signal": avg_return_by_signal(results_b),
            "n": len(results_b),
        },
    }

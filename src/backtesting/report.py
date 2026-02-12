"""
Backtest Report Generator
--------------------------
Generates human-readable reports from backtest results.
Outputs to terminal and optionally saves JSON summaries.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.config.settings import Settings

logger = logging.getLogger(__name__)


def generate_terminal_report(backtest_data: Dict) -> str:
    """
    Generate a formatted terminal report from backtest results.

    Args:
        backtest_data: Output from BacktestResult.to_dict()

    Returns:
        Formatted string report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  ALPHAEXTRACT BACKTEST REPORT")
    lines.append("=" * 70)

    # Config
    config = backtest_data.get("config", {})
    lines.append(f"\n  Tickers:        {', '.join(config.get('tickers', []))}")
    lines.append(f"  Mode:           {config.get('mode', 'unknown')}")
    lines.append(f"  Return Window:  {config.get('return_window_days', '?')} days")
    lines.append(f"  Flat Threshold: {config.get('flat_threshold', 0.02) * 100:.0f}%")

    # Summary
    summary = backtest_data.get("summary", {})
    lines.append(f"\n{'─' * 70}")
    lines.append("  PERFORMANCE SUMMARY")
    lines.append(f"{'─' * 70}")
    lines.append(f"  Total Signals:          {summary.get('total_signals', 0)}")
    lines.append(f"  Scorable:               {summary.get('scorable', 0)}")
    lines.append(f"  Hit Rate:               {summary.get('hit_rate', 0):.1%}")
    lines.append(f"  Directional Accuracy:   {summary.get('directional_accuracy', 0):.1%}")
    sharpe = summary.get("sharpe_ratio")
    lines.append(f"  Sharpe Ratio:           {sharpe:.3f}" if sharpe is not None else "  Sharpe Ratio:           N/A")

    # Precision by signal
    precision = backtest_data.get("precision_by_signal", {})
    if precision:
        lines.append(f"\n{'─' * 70}")
        lines.append("  PRECISION BY SIGNAL")
        lines.append(f"{'─' * 70}")
        lines.append(f"  {'Signal':<15} {'Correct':>8} {'Total':>8} {'Precision':>10}")
        lines.append(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*10}")
        for signal in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]:
            if signal in precision:
                s = precision[signal]
                lines.append(f"  {signal:<15} {s['correct']:>8} {s['total']:>8} {s['precision']:>10.1%}")

    # Average return by signal
    avg_ret = backtest_data.get("avg_return_by_signal", {})
    if avg_ret:
        lines.append(f"\n{'─' * 70}")
        lines.append("  AVERAGE RETURN BY SIGNAL")
        lines.append(f"{'─' * 70}")
        lines.append(f"  {'Signal':<15} {'Avg Return':>10} {'Min':>8} {'Max':>8} {'Count':>7}")
        lines.append(f"  {'─'*15} {'─'*10} {'─'*8} {'─'*8} {'─'*7}")
        for signal in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]:
            if signal in avg_ret:
                s = avg_ret[signal]
                lines.append(
                    f"  {signal:<15} {s['avg_return_pct']:>+9.1f}% {s['min_return_pct']:>+7.1f}% "
                    f"{s['max_return_pct']:>+7.1f}% {s['count']:>7}"
                )

    # Confusion matrix
    cm = backtest_data.get("confusion_matrix", {})
    if cm:
        lines.append(f"\n{'─' * 70}")
        lines.append("  CONFUSION MATRIX (Predicted vs Actual)")
        lines.append(f"{'─' * 70}")
        lines.append(f"  {'':>12} {'Up':>8} {'Flat':>8} {'Down':>8}")
        lines.append(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8}")
        for predicted in ["bullish", "neutral", "bearish"]:
            if predicted in cm:
                row = cm[predicted]
                lines.append(f"  {predicted:>12} {row.get('up',0):>8} {row.get('flat',0):>8} {row.get('down',0):>8}")

    # Worst misses
    results = backtest_data.get("results", [])
    misses = _get_worst_misses(results)
    if misses:
        lines.append(f"\n{'─' * 70}")
        lines.append("  WORST MISSES (Biggest wrong calls)")
        lines.append(f"{'─' * 70}")
        for miss in misses[:5]:
            lines.append(
                f"  {miss['ticker']} {miss['filing_date']}: "
                f"Signal={miss['signal']}, Actual={miss['actual_return_pct']:+.1f}% ({miss['actual_direction']})"
            )

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def generate_comparison_report(comparison_data: Dict) -> str:
    """Generate a side-by-side comparison report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  MODEL COMPARISON: FinBERT vs Ensemble")
    lines.append("=" * 70)

    comparison = comparison_data.get("comparison", {})
    for label, stats in comparison.items():
        lines.append(f"\n  [{label}]")
        lines.append(f"    Signals:              {stats.get('n', 0)}")
        lines.append(f"    Hit Rate:             {stats.get('hit_rate', 0):.1%}")
        lines.append(f"    Directional Accuracy: {stats.get('directional_accuracy', 0):.1%}")
        sharpe = stats.get("sharpe")
        lines.append(f"    Sharpe Ratio:         {sharpe:.3f}" if sharpe is not None else "    Sharpe Ratio:         N/A")

    # Winner
    labels = list(comparison.keys())
    if len(labels) == 2:
        a_acc = comparison[labels[0]].get("directional_accuracy", 0)
        b_acc = comparison[labels[1]].get("directional_accuracy", 0)
        if a_acc > b_acc:
            winner = labels[0]
            delta = a_acc - b_acc
        elif b_acc > a_acc:
            winner = labels[1]
            delta = b_acc - a_acc
        else:
            winner = "Tie"
            delta = 0

        lines.append(f"\n  Winner: {winner} (+{delta:.1%} directional accuracy)")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def save_report(backtest_data: Dict, name: str = "report") -> Path:
    """Save full backtest report as JSON."""
    output_dir = Settings.BACKTEST_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{name}_{timestamp}.json"
    path.write_text(json.dumps(backtest_data, indent=2), encoding="utf-8")
    logger.info(f"Saved report: {path.name}")
    return path


def _get_worst_misses(results: List[Dict], n: int = 5) -> List[Dict]:
    """Find the biggest wrong calls — bullish signals with biggest losses, etc."""
    misses = []
    for r in results:
        signal = r.get("signal", "HOLD")
        ret = r.get("actual_return_pct")
        direction = r.get("actual_direction")
        if ret is None or direction in (None, "unknown"):
            continue

        is_miss = (
            (signal in ("STRONG_BUY", "BUY") and direction == "down")
            or (signal in ("STRONG_SELL", "SELL") and direction == "up")
        )
        if is_miss:
            misses.append(r)

    # Sort by magnitude of the wrong direction
    misses.sort(key=lambda r: abs(r.get("actual_return_pct", 0)), reverse=True)
    return misses[:n]

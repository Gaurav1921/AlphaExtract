"""
Historical Options Tracking
-----------------------------
Stores options snapshots over time and provides trend analysis for
put/call ratios and IV skew. Builds on OptionsOverlay to maintain
a time-series of options-derived metrics.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class OptionsHistoryEntry:
    """A single historical options data point."""
    ticker: str
    date: str
    put_call_ratio: Optional[float] = None
    put_call_volume_ratio: Optional[float] = None
    avg_call_iv: Optional[float] = None
    avg_put_iv: Optional[float] = None
    iv_skew: Optional[float] = None
    options_score: float = 0.0
    options_signal: str = "N/A"
    total_call_oi: int = 0
    total_put_oi: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OptionsTimeSeries:
    """Time-series of options data for a single ticker."""
    ticker: str
    entries: List[Dict]
    data_points: int
    date_range: Optional[Dict] = None
    trend: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class OptionsHistoryTracker:
    """
    Tracks options snapshots over time and computes trends.

    Stores daily snapshots in a per-ticker JSON file and provides
    trend analysis for put/call ratio and IV skew.
    """

    def __init__(self, history_dir: Path = None):
        self.history_dir = history_dir or Settings.OPTIONS_HISTORY_DIR
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def _history_path(self, ticker: str) -> Path:
        return self.history_dir / f"{ticker.upper()}_options_history.json"

    def _load_history(self, ticker: str) -> List[Dict]:
        """Load existing history for a ticker."""
        path = self._history_path(ticker)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("entries", [])
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load options history for {ticker}: {e}")
            return []

    def _save_history(self, ticker: str, entries: List[Dict]):
        """Save history entries to disk."""
        ticker = ticker.upper()
        path = self._history_path(ticker)
        data = {
            "ticker": ticker,
            "last_updated": datetime.now().isoformat(),
            "entries": entries,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"Saved {len(entries)} history entries for {ticker}")

    def record_snapshot(self, snapshot) -> OptionsHistoryEntry:
        """
        Record an OptionsSnapshot into the history file.

        Args:
            snapshot: An OptionsSnapshot from OptionsOverlay.fetch_options().

        Returns:
            The OptionsHistoryEntry that was recorded.
        """
        ticker = snapshot.ticker.upper()
        entry = OptionsHistoryEntry(
            ticker=ticker,
            date=snapshot.snapshot_date,
            put_call_ratio=snapshot.put_call_ratio,
            put_call_volume_ratio=snapshot.put_call_volume_ratio,
            avg_call_iv=snapshot.avg_call_iv,
            avg_put_iv=snapshot.avg_put_iv,
            iv_skew=snapshot.iv_skew,
            options_score=snapshot.options_score,
            options_signal=snapshot.options_signal,
            total_call_oi=snapshot.total_call_oi,
            total_put_oi=snapshot.total_put_oi,
        )

        entries = self._load_history(ticker)

        # Avoid duplicate entries for the same date
        existing_dates = {e["date"] for e in entries}
        if entry.date in existing_dates:
            # Update existing entry for today
            entries = [e for e in entries if e["date"] != entry.date]

        entries.append(entry.to_dict())
        entries.sort(key=lambda e: e["date"])

        self._save_history(ticker, entries)
        return entry

    def get_history(self, ticker: str, limit: int = 0) -> OptionsTimeSeries:
        """
        Get historical options data for a ticker.

        Args:
            ticker: Stock ticker.
            limit: Max entries to return (0 = all).

        Returns:
            OptionsTimeSeries with entries and trend analysis.
        """
        ticker = ticker.upper()
        entries = self._load_history(ticker)

        if limit > 0:
            entries = entries[-limit:]

        date_range = None
        if entries:
            date_range = {
                "first": entries[0]["date"],
                "last": entries[-1]["date"],
            }

        trend = self._compute_trend(entries) if len(entries) >= 2 else None

        return OptionsTimeSeries(
            ticker=ticker,
            entries=entries,
            data_points=len(entries),
            date_range=date_range,
            trend=trend,
        )

    def _compute_trend(self, entries: List[Dict]) -> Dict:
        """Compute trend metrics from historical entries."""
        pc_ratios = [e["put_call_ratio"] for e in entries if e.get("put_call_ratio") is not None]
        iv_skews = [e["iv_skew"] for e in entries if e.get("iv_skew") is not None]
        scores = [e["options_score"] for e in entries if e.get("options_score") is not None]

        trend = {}

        if len(pc_ratios) >= 2:
            trend["put_call_ratio"] = {
                "current": pc_ratios[-1],
                "previous": pc_ratios[-2],
                "change": round(pc_ratios[-1] - pc_ratios[-2], 4),
                "avg": round(sum(pc_ratios) / len(pc_ratios), 4),
                "min": round(min(pc_ratios), 4),
                "max": round(max(pc_ratios), 4),
                "direction": self._trend_direction(pc_ratios),
            }

        if len(iv_skews) >= 2:
            trend["iv_skew"] = {
                "current": iv_skews[-1],
                "previous": iv_skews[-2],
                "change": round(iv_skews[-1] - iv_skews[-2], 4),
                "avg": round(sum(iv_skews) / len(iv_skews), 4),
                "min": round(min(iv_skews), 4),
                "max": round(max(iv_skews), 4),
                "direction": self._trend_direction(iv_skews),
            }

        if len(scores) >= 2:
            trend["options_score"] = {
                "current": scores[-1],
                "previous": scores[-2],
                "change": round(scores[-1] - scores[-2], 4),
                "avg": round(sum(scores) / len(scores), 4),
                "direction": self._trend_direction(scores),
            }

        return trend

    @staticmethod
    def _trend_direction(values: List[float]) -> str:
        """Determine trend direction from a list of values."""
        if len(values) < 2:
            return "stable"
        # Use simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return "stable"
        slope = numerator / denominator
        if slope > 0.01:
            return "rising"
        if slope < -0.01:
            return "falling"
        return "stable"

    def fetch_and_record(self, ticker: str) -> Optional[OptionsHistoryEntry]:
        """
        Fetch current options data and record to history.

        Convenience method that fetches via OptionsOverlay and records.
        """
        from src.market.options_overlay import OptionsOverlay

        overlay = OptionsOverlay()
        snapshot = overlay.fetch_options(ticker)
        if snapshot is None:
            logger.warning(f"No options data to record for {ticker}")
            return None
        return self.record_snapshot(snapshot)

    def batch_fetch_and_record(self, tickers: List[str]) -> List[OptionsHistoryEntry]:
        """Fetch and record options history for multiple tickers."""
        results = []
        for ticker in tickers:
            entry = self.fetch_and_record(ticker)
            if entry:
                results.append(entry)
        return results

    def get_summary(self, ticker: str) -> Dict:
        """Get a concise summary of options history for a ticker."""
        ts = self.get_history(ticker)
        summary = {
            "ticker": ts.ticker,
            "data_points": ts.data_points,
            "date_range": ts.date_range,
        }

        if ts.trend:
            pc_trend = ts.trend.get("put_call_ratio", {})
            iv_trend = ts.trend.get("iv_skew", {})
            summary["put_call_ratio_trend"] = pc_trend.get("direction", "N/A")
            summary["iv_skew_trend"] = iv_trend.get("direction", "N/A")
            summary["current_pc_ratio"] = pc_trend.get("current")
            summary["current_iv_skew"] = iv_trend.get("current")
        else:
            summary["put_call_ratio_trend"] = "N/A"
            summary["iv_skew_trend"] = "N/A"

        return summary

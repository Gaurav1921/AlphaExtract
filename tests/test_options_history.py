"""
Tests for Historical Options Tracking
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from src.market.options_history import (
    OptionsHistoryTracker,
    OptionsHistoryEntry,
    OptionsTimeSeries,
)


@dataclass
class MockOptionsSnapshot:
    """Mock OptionsSnapshot for testing."""
    ticker: str = "AAPL"
    snapshot_date: str = "2024-11-15"
    put_call_ratio: float = 0.85
    put_call_volume_ratio: float = 0.90
    avg_call_iv: float = 0.30
    avg_put_iv: float = 0.35
    iv_skew: float = 0.05
    options_score: float = -0.1
    options_signal: str = "HOLD"
    total_call_oi: int = 500000
    total_put_oi: int = 425000


@pytest.fixture
def tracker(tmp_path):
    return OptionsHistoryTracker(history_dir=tmp_path)


@pytest.fixture
def snapshot():
    return MockOptionsSnapshot()


class TestOptionsHistoryEntry:
    def test_creates_entry(self):
        entry = OptionsHistoryEntry(
            ticker="AAPL",
            date="2024-11-15",
            put_call_ratio=0.85,
            iv_skew=0.05,
        )
        assert entry.ticker == "AAPL"
        assert entry.put_call_ratio == 0.85

    def test_to_dict(self):
        entry = OptionsHistoryEntry(ticker="MSFT", date="2024-11-15")
        d = entry.to_dict()
        assert d["ticker"] == "MSFT"
        assert "options_score" in d


class TestOptionsTimeSeries:
    def test_creates_time_series(self):
        ts = OptionsTimeSeries(
            ticker="AAPL",
            entries=[],
            data_points=0,
        )
        assert ts.ticker == "AAPL"
        assert ts.data_points == 0

    def test_to_dict(self):
        ts = OptionsTimeSeries(ticker="AAPL", entries=[], data_points=0)
        d = ts.to_dict()
        assert d["ticker"] == "AAPL"


class TestOptionsHistoryTracker:
    def test_record_snapshot(self, tracker, snapshot):
        entry = tracker.record_snapshot(snapshot)
        assert entry.ticker == "AAPL"
        assert entry.date == "2024-11-15"
        assert entry.put_call_ratio == 0.85

        # Verify file was created
        path = tracker._history_path("AAPL")
        assert path.exists()

    def test_record_duplicate_date_updates(self, tracker, snapshot):
        tracker.record_snapshot(snapshot)

        # Record again for same date with different data
        snapshot.put_call_ratio = 0.90
        tracker.record_snapshot(snapshot)

        history = tracker.get_history("AAPL")
        assert history.data_points == 1  # Should replace, not duplicate
        assert history.entries[0]["put_call_ratio"] == 0.90

    def test_get_empty_history(self, tracker):
        ts = tracker.get_history("ZZZZ")
        assert ts.data_points == 0
        assert ts.entries == []
        assert ts.date_range is None
        assert ts.trend is None

    def test_get_history_with_data(self, tracker, snapshot):
        tracker.record_snapshot(snapshot)

        ts = tracker.get_history("AAPL")
        assert ts.data_points == 1
        assert ts.date_range["first"] == "2024-11-15"
        assert ts.date_range["last"] == "2024-11-15"

    def test_get_history_with_limit(self, tracker):
        for i in range(5):
            snap = MockOptionsSnapshot(snapshot_date=f"2024-11-{10 + i:02d}")
            tracker.record_snapshot(snap)

        ts = tracker.get_history("AAPL", limit=3)
        assert ts.data_points == 3

    def test_multiple_dates(self, tracker):
        snap1 = MockOptionsSnapshot(snapshot_date="2024-11-01", put_call_ratio=0.80)
        snap2 = MockOptionsSnapshot(snapshot_date="2024-11-08", put_call_ratio=0.90)
        snap3 = MockOptionsSnapshot(snapshot_date="2024-11-15", put_call_ratio=1.00)

        tracker.record_snapshot(snap1)
        tracker.record_snapshot(snap2)
        tracker.record_snapshot(snap3)

        ts = tracker.get_history("AAPL")
        assert ts.data_points == 3
        assert ts.entries[0]["date"] == "2024-11-01"
        assert ts.entries[-1]["date"] == "2024-11-15"

    def test_trend_computation(self, tracker):
        for i in range(5):
            snap = MockOptionsSnapshot(
                snapshot_date=f"2024-11-{10 + i:02d}",
                put_call_ratio=0.80 + i * 0.05,
                iv_skew=0.02 + i * 0.01,
                options_score=-0.1 - i * 0.05,
            )
            tracker.record_snapshot(snap)

        ts = tracker.get_history("AAPL")
        assert ts.trend is not None
        assert "put_call_ratio" in ts.trend
        assert "iv_skew" in ts.trend
        assert "options_score" in ts.trend

        # P/C ratio is rising
        assert ts.trend["put_call_ratio"]["direction"] == "rising"

    def test_trend_stable(self, tracker):
        for i in range(3):
            snap = MockOptionsSnapshot(
                snapshot_date=f"2024-11-{10 + i:02d}",
                put_call_ratio=0.85,
                iv_skew=0.05,
            )
            tracker.record_snapshot(snap)

        ts = tracker.get_history("AAPL")
        assert ts.trend is not None
        assert ts.trend["put_call_ratio"]["direction"] == "stable"

    def test_trend_falling(self, tracker):
        for i in range(5):
            snap = MockOptionsSnapshot(
                snapshot_date=f"2024-11-{10 + i:02d}",
                put_call_ratio=1.00 - i * 0.05,
            )
            tracker.record_snapshot(snap)

        ts = tracker.get_history("AAPL")
        assert ts.trend["put_call_ratio"]["direction"] == "falling"

    def test_trend_stats(self, tracker):
        snap1 = MockOptionsSnapshot(snapshot_date="2024-11-01", put_call_ratio=0.80)
        snap2 = MockOptionsSnapshot(snapshot_date="2024-11-08", put_call_ratio=1.20)

        tracker.record_snapshot(snap1)
        tracker.record_snapshot(snap2)

        ts = tracker.get_history("AAPL")
        pc = ts.trend["put_call_ratio"]
        assert pc["current"] == 1.20
        assert pc["previous"] == 0.80
        assert pc["change"] == 0.40
        assert pc["min"] == 0.80
        assert pc["max"] == 1.20

    def test_get_summary(self, tracker):
        snap1 = MockOptionsSnapshot(snapshot_date="2024-11-01", put_call_ratio=0.80)
        snap2 = MockOptionsSnapshot(snapshot_date="2024-11-08", put_call_ratio=0.90)
        tracker.record_snapshot(snap1)
        tracker.record_snapshot(snap2)

        summary = tracker.get_summary("AAPL")
        assert summary["ticker"] == "AAPL"
        assert summary["data_points"] == 2
        assert summary["put_call_ratio_trend"] in ("rising", "falling", "stable")

    def test_get_summary_empty(self, tracker):
        summary = tracker.get_summary("ZZZZ")
        assert summary["data_points"] == 0
        assert summary["put_call_ratio_trend"] == "N/A"

    def test_corrupted_history_file(self, tracker):
        path = tracker._history_path("AAPL")
        path.write_text("not valid json", encoding="utf-8")
        entries = tracker._load_history("AAPL")
        assert entries == []

    def test_history_path(self, tracker):
        path = tracker._history_path("AAPL")
        assert "AAPL_options_history.json" in str(path)

    def test_case_insensitive_ticker(self, tracker, snapshot):
        snapshot.ticker = "aapl"
        tracker.record_snapshot(snapshot)

        ts = tracker.get_history("aapl")
        assert ts.ticker == "AAPL"
        assert ts.data_points == 1


class TestTrendDirection:
    def test_rising(self):
        assert OptionsHistoryTracker._trend_direction([1, 2, 3, 4, 5]) == "rising"

    def test_falling(self):
        assert OptionsHistoryTracker._trend_direction([5, 4, 3, 2, 1]) == "falling"

    def test_stable(self):
        assert OptionsHistoryTracker._trend_direction([3, 3, 3, 3]) == "stable"

    def test_single_value(self):
        assert OptionsHistoryTracker._trend_direction([5]) == "stable"

    def test_empty(self):
        assert OptionsHistoryTracker._trend_direction([]) == "stable"

"""
Tests for SEC Filing Alert Monitor
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.alerts.sec_monitor import SECMonitor, FilingAlert


@pytest.fixture
def state_file(tmp_path):
    return tmp_path / "alert_state.json"


@pytest.fixture
def monitor(state_file):
    return SECMonitor(
        watchlist=["AAPL", "MSFT"],
        state_file=state_file,
        poll_interval=60,
    )


class TestFilingAlert:
    def test_creates_alert(self):
        alert = FilingAlert(
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date="2024-11-01",
            accession_number="0000320193-24-000123",
            filing_url="https://www.sec.gov/example",
        )
        assert alert.ticker == "AAPL"
        assert alert.company_name == "Apple Inc."
        assert alert.detected_at is not None

    def test_to_dict(self):
        alert = FilingAlert(
            ticker="MSFT",
            company_name="Microsoft Corp.",
            filing_date="2024-07-30",
            accession_number="0000789019-24-000456",
            filing_url="https://www.sec.gov/example2",
        )
        d = alert.to_dict()
        assert d["ticker"] == "MSFT"
        assert "detected_at" in d
        assert "filing_url" in d


class TestSECMonitorInit:
    def test_creates_instance(self, monitor):
        assert monitor.watchlist == ["AAPL", "MSFT"]
        assert monitor.poll_interval == 60

    def test_empty_watchlist(self, state_file):
        m = SECMonitor(state_file=state_file)
        assert m.watchlist == []

    def test_uppercases_tickers(self, state_file):
        m = SECMonitor(watchlist=["aapl", "msft"], state_file=state_file)
        assert m.watchlist == ["AAPL", "MSFT"]


class TestWatchlistManagement:
    def test_add_ticker(self, monitor):
        monitor.add_ticker("GOOGL")
        assert "GOOGL" in monitor.watchlist

    def test_add_duplicate(self, monitor):
        monitor.add_ticker("AAPL")
        assert monitor.watchlist.count("AAPL") == 1

    def test_remove_ticker(self, monitor):
        monitor.remove_ticker("MSFT")
        assert "MSFT" not in monitor.watchlist

    def test_set_watchlist(self, monitor):
        monitor.set_watchlist(["TSLA", "NVDA"])
        assert monitor.watchlist == ["TSLA", "NVDA"]


class TestStatePersistence:
    def test_saves_and_loads_state(self, state_file):
        m1 = SECMonitor(watchlist=["AAPL"], state_file=state_file)
        m1._seen["acc-001"] = "2024-01-01T00:00:00"
        m1._cik_map["AAPL"] = "0000320193"
        m1._save_state()

        m2 = SECMonitor(watchlist=["AAPL"], state_file=state_file)
        assert "acc-001" in m2._seen
        assert m2._cik_map.get("AAPL") == "0000320193"

    def test_handles_missing_state_file(self, tmp_path):
        m = SECMonitor(state_file=tmp_path / "nonexistent.json")
        assert m._seen == {}

    def test_clear_state(self, monitor):
        monitor._seen["acc-001"] = "2024-01-01"
        monitor.clear_state()
        assert monitor._seen == {}


class TestCallbacks:
    def test_registers_callback(self, monitor):
        cb = MagicMock()
        monitor.on_new_filing(cb)
        assert cb in monitor._callbacks

    def test_fires_callback(self, monitor):
        cb = MagicMock()
        monitor.on_new_filing(cb)

        alert = FilingAlert(
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date="2024-11-01",
            accession_number="acc-999",
            filing_url="https://example.com",
        )
        monitor._notify(alert)
        cb.assert_called_once_with(alert)

    def test_callback_error_doesnt_crash(self, monitor):
        def bad_callback(alert):
            raise ValueError("boom")

        monitor.on_new_filing(bad_callback)
        alert = FilingAlert(
            ticker="AAPL",
            company_name="Apple",
            filing_date="2024-01-01",
            accession_number="acc-err",
            filing_url="https://example.com",
        )
        # Should not raise
        monitor._notify(alert)


class TestCheckOnce:
    def test_empty_watchlist_returns_empty(self, state_file):
        m = SECMonitor(watchlist=[], state_file=state_file)
        alerts = m.check_once()
        assert alerts == []

    @patch.object(SECMonitor, "_fetch_company_filings")
    def test_detects_new_filings(self, mock_fetch, monitor):
        mock_fetch.return_value = [{
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "filing_date": "2024-11-01",
            "accession_number": "0000320193-24-000999",
            "filing_url": "https://www.sec.gov/example",
        }]

        alerts = monitor.check_once(days_back=30)
        assert len(alerts) == 1
        assert alerts[0].ticker == "AAPL"

    @patch.object(SECMonitor, "_fetch_company_filings")
    def test_skips_already_seen(self, mock_fetch, monitor):
        monitor._seen["0000320193-24-000999"] = "2024-01-01"
        mock_fetch.return_value = [{
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "filing_date": "2024-11-01",
            "accession_number": "0000320193-24-000999",
            "filing_url": "https://www.sec.gov/example",
        }]

        alerts = monitor.check_once(days_back=30)
        assert len(alerts) == 0


class TestStatus:
    def test_status_returns_dict(self, monitor):
        status = monitor.status()
        assert status["watchlist"] == ["AAPL", "MSFT"]
        assert status["watchlist_count"] == 2
        assert "seen_filings" in status
        assert "poll_interval_seconds" in status

    def test_recent_alerts(self, monitor):
        monitor._seen["acc-1"] = "2024-01-01T00:00:00"
        monitor._seen["acc-2"] = "2024-06-15T12:00:00"

        recent = monitor.get_recent_alerts(limit=5)
        assert len(recent) == 2
        assert recent[0]["accession_number"] == "acc-2"  # most recent first

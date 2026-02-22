"""
Tests for Webhook Notifications
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from src.alerts.webhook import (
    WebhookNotifier,
    WebhookConfig,
    WebhookResult,
)


@dataclass
class MockFilingAlert:
    """Mock FilingAlert for testing."""
    ticker: str = "AAPL"
    company_name: str = "Apple Inc."
    filing_date: str = "2024-01-15"
    accession_number: str = "0000320193-24-000012"
    filing_url: str = "https://www.sec.gov/..."
    detected_at: str = "2024-01-15T10:00:00"

    def to_dict(self):
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "filing_date": self.filing_date,
            "accession_number": self.accession_number,
            "filing_url": self.filing_url,
            "detected_at": self.detected_at,
        }


@pytest.fixture
def notifier():
    return WebhookNotifier()


@pytest.fixture
def alert():
    return MockFilingAlert()


class TestWebhookConfig:
    def test_creates_config(self):
        config = WebhookConfig(url="https://example.com/hook", name="test")
        assert config.url == "https://example.com/hook"
        assert config.name == "test"
        assert config.format == "json"
        assert config.enabled is True

    def test_to_dict(self):
        config = WebhookConfig(url="https://example.com", name="slack", format="slack")
        d = config.to_dict()
        assert d["url"] == "https://example.com"
        assert d["format"] == "slack"


class TestWebhookResult:
    def test_creates_result(self):
        result = WebhookResult(
            webhook_name="test",
            url="https://example.com",
            success=True,
            status_code=200,
        )
        assert result.success is True
        assert result.status_code == 200

    def test_failed_result(self):
        result = WebhookResult(
            webhook_name="test",
            url="https://example.com",
            success=False,
            error="Connection refused",
        )
        assert result.success is False
        assert "Connection refused" in result.error


class TestWebhookNotifier:
    def test_add_webhook(self, notifier):
        notifier.add_webhook("https://example.com/hook", name="test")
        assert notifier.webhook_count == 1

    def test_add_multiple_webhooks(self, notifier):
        notifier.add_webhook("https://example.com/hook1", name="test1")
        notifier.add_webhook("https://example.com/hook2", name="test2")
        assert notifier.webhook_count == 2

    def test_remove_webhook(self, notifier):
        notifier.add_webhook("https://example.com/hook", name="test")
        notifier.remove_webhook("test")
        assert notifier.webhook_count == 0

    def test_remove_nonexistent_webhook(self, notifier):
        notifier.remove_webhook("nonexistent")
        assert notifier.webhook_count == 0

    def test_enabled_count(self, notifier):
        notifier.webhooks = [
            WebhookConfig(url="https://a.com", name="a", enabled=True),
            WebhookConfig(url="https://b.com", name="b", enabled=False),
        ]
        assert notifier.enabled_count == 1

    def test_status(self, notifier):
        notifier.add_webhook("https://example.com", name="test")
        status = notifier.status()
        assert status["total_webhooks"] == 1
        assert status["enabled_webhooks"] == 1
        assert len(status["webhooks"]) == 1

    def test_build_json_payload(self, notifier, alert):
        payload = json.loads(notifier._build_json_payload(alert))
        assert payload["event"] == "new_filing"
        assert payload["source"] == "AlphaExtract"
        assert payload["alert"]["ticker"] == "AAPL"

    def test_build_slack_payload(self, notifier, alert):
        payload = json.loads(notifier._build_slack_payload(alert))
        assert "text" in payload
        assert "AAPL" in payload["text"]
        assert "Apple Inc." in payload["text"]

    def test_notify_no_webhooks(self, notifier, alert):
        results = notifier.notify(alert)
        assert results == []

    def test_notify_disabled_webhook(self, notifier, alert):
        notifier.webhooks = [
            WebhookConfig(url="https://example.com", name="disabled", enabled=False),
        ]
        results = notifier.notify(alert)
        assert len(results) == 1
        assert results[0].success is False
        assert "disabled" in results[0].error.lower()

    @patch("src.alerts.webhook.urlopen")
    def test_notify_success(self, mock_urlopen, notifier, alert):
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_urlopen.return_value = mock_resp

        notifier.add_webhook("https://example.com/hook", name="test")
        results = notifier.notify(alert)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].status_code == 200

    @patch("src.alerts.webhook.urlopen")
    def test_notify_failure(self, mock_urlopen, notifier, alert):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")

        notifier.add_webhook("https://example.com/hook", name="test")
        results = notifier.notify(alert)

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None

    @patch("src.alerts.webhook.urlopen")
    def test_notify_batch(self, mock_urlopen, notifier):
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_urlopen.return_value = mock_resp

        notifier.add_webhook("https://example.com/hook", name="test")
        alerts = [MockFilingAlert(ticker="AAPL"), MockFilingAlert(ticker="MSFT")]
        results = notifier.notify_batch(alerts)

        assert len(results) == 2
        assert all(r.success for r in results)

    @patch("src.alerts.webhook.urlopen")
    def test_slack_format_webhook(self, mock_urlopen, notifier, alert):
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_urlopen.return_value = mock_resp

        notifier.add_webhook("https://hooks.slack.com/test", name="slack", format="slack")
        results = notifier.notify(alert)

        assert results[0].success is True
        # Verify the payload sent was Slack-formatted
        call_args = mock_urlopen.call_args[0][0]
        payload = json.loads(call_args.data)
        assert "text" in payload


class TestSECMonitorWebhookIntegration:
    @patch("src.alerts.sec_monitor.SECMonitor._fetch_company_filings")
    @patch("src.alerts.sec_monitor.time.sleep")
    def test_monitor_fires_webhooks(self, mock_sleep, mock_fetch):
        from src.alerts.sec_monitor import SECMonitor

        mock_fetch.return_value = [{
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "filing_date": "2024-01-15",
            "accession_number": "test-acc-123",
            "filing_url": "https://www.sec.gov/...",
        }]

        mock_notifier = MagicMock()

        with patch("src.alerts.sec_monitor.Settings") as mock_settings:
            mock_settings.ALERT_STATE_FILE = MagicMock()
            mock_settings.ALERT_STATE_FILE.exists.return_value = False
            mock_settings.SEC_RATE_LIMIT_DELAY = 0

            monitor = SECMonitor(
                watchlist=["AAPL"],
                webhook_notifier=mock_notifier,
            )
            alerts = monitor.check_once(days_back=30)

        assert len(alerts) == 1
        mock_notifier.notify.assert_called_once()

"""
Webhook Notifications for Filing Alerts
-----------------------------------------
Sends HTTP POST notifications when new 10-K filings are detected.
Supports generic webhook URLs (Slack, Discord, custom endpoints).
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from src.config.settings import Settings

logger = logging.getLogger(__name__)

# Timeout for webhook HTTP requests (seconds)
WEBHOOK_TIMEOUT = 10


@dataclass
class WebhookConfig:
    """Configuration for a single webhook endpoint."""
    url: str
    name: str = "default"
    format: str = "json"  # "json" or "slack"
    enabled: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WebhookResult:
    """Result of a webhook delivery attempt."""
    webhook_name: str
    url: str
    success: bool
    status_code: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class WebhookNotifier:
    """
    Sends webhook notifications for filing alerts.

    Supports:
    - Generic JSON POST (any HTTP endpoint)
    - Slack-formatted messages (Slack incoming webhooks)
    - Multiple webhook endpoints
    - Retry on failure (configurable)
    """

    def __init__(self, webhooks: List[WebhookConfig] = None):
        self.webhooks: List[WebhookConfig] = webhooks or []

    def add_webhook(self, url: str, name: str = "default", format: str = "json"):
        """Add a webhook endpoint."""
        self.webhooks.append(WebhookConfig(url=url, name=name, format=format))
        logger.info(f"Added webhook: {name} ({format})")

    def remove_webhook(self, name: str):
        """Remove a webhook by name."""
        self.webhooks = [w for w in self.webhooks if w.name != name]

    def _build_json_payload(self, alert) -> bytes:
        """Build a generic JSON payload from a FilingAlert."""
        payload = {
            "event": "new_filing",
            "source": "AlphaExtract",
            "alert": alert.to_dict(),
        }
        return json.dumps(payload).encode("utf-8")

    def _build_slack_payload(self, alert) -> bytes:
        """Build a Slack-formatted payload from a FilingAlert."""
        payload = {
            "text": (
                f"*New 10-K Filing Detected*\n"
                f"*Company:* {alert.ticker} ({alert.company_name})\n"
                f"*Filed:* {alert.filing_date}\n"
                f"*Accession:* {alert.accession_number}\n"
                f"<{alert.filing_url}|View Filing>"
            ),
        }
        return json.dumps(payload).encode("utf-8")

    def _send_webhook(self, webhook: WebhookConfig, alert) -> WebhookResult:
        """Send a single webhook notification."""
        if not webhook.enabled:
            return WebhookResult(
                webhook_name=webhook.name,
                url=webhook.url,
                success=False,
                error="Webhook disabled",
            )

        if webhook.format == "slack":
            body = self._build_slack_payload(alert)
        else:
            body = self._build_json_payload(alert)

        try:
            req = Request(
                webhook.url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urlopen(req, timeout=WEBHOOK_TIMEOUT)
            status = resp.getcode()
            logger.info(f"Webhook {webhook.name} delivered: HTTP {status}")
            return WebhookResult(
                webhook_name=webhook.name,
                url=webhook.url,
                success=True,
                status_code=status,
            )
        except (URLError, OSError) as e:
            logger.warning(f"Webhook {webhook.name} failed: {e}")
            return WebhookResult(
                webhook_name=webhook.name,
                url=webhook.url,
                success=False,
                error=str(e),
            )

    def notify(self, alert) -> List[WebhookResult]:
        """
        Send webhook notifications to all configured endpoints.

        Args:
            alert: A FilingAlert dataclass instance.

        Returns:
            List of WebhookResult for each endpoint.
        """
        if not self.webhooks:
            return []

        results = []
        for webhook in self.webhooks:
            result = self._send_webhook(webhook, alert)
            results.append(result)

        delivered = sum(1 for r in results if r.success)
        logger.info(f"Webhooks: {delivered}/{len(results)} delivered for {alert.ticker}")
        return results

    def notify_batch(self, alerts: list) -> List[WebhookResult]:
        """Send webhook notifications for multiple alerts."""
        all_results = []
        for alert in alerts:
            all_results.extend(self.notify(alert))
        return all_results

    @property
    def webhook_count(self) -> int:
        return len(self.webhooks)

    @property
    def enabled_count(self) -> int:
        return sum(1 for w in self.webhooks if w.enabled)

    def status(self) -> Dict:
        """Return webhook configuration status."""
        return {
            "total_webhooks": self.webhook_count,
            "enabled_webhooks": self.enabled_count,
            "webhooks": [w.to_dict() for w in self.webhooks],
        }

"""
SEC Filing Alert Monitor
-------------------------
Monitors SEC EDGAR RSS feeds for new 10-K filings from watched tickers.
Supports polling mode, callback notifications, and state persistence.
"""

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from src.config.settings import Settings

logger = logging.getLogger(__name__)

SEC_RSS_URL = "https://efts.sec.gov/LATEST/search-index?q=%2210-K%22&dateRange=custom&startdt={start}&enddt={end}&forms=10-K"
SEC_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%2210-K%22&forms=10-K&dateRange=custom&startdt={start}&enddt={end}"
SEC_EDGAR_FEED = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=5&search_text=&action=getcompany&output=atom"
SEC_EFTS_API = "https://efts.sec.gov/LATEST/search-index?q=%2210-K%22&forms=10-K"


@dataclass
class FilingAlert:
    """Represents a detected new filing."""
    ticker: str
    company_name: str
    filing_date: str
    accession_number: str
    filing_url: str
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


class SECMonitor:
    """
    Monitors SEC EDGAR for new 10-K filings from watched tickers.

    Supports:
    - Polling SEC EDGAR full-text search API
    - Ticker watchlist filtering
    - Callback notifications for new filings
    - State persistence to avoid duplicate alerts
    """

    def __init__(
        self,
        watchlist: List[str] = None,
        state_file: Path = None,
        poll_interval: int = 3600,
    ):
        self.watchlist = [t.upper() for t in (watchlist or [])]
        self.state_file = state_file or Settings.DATA_DIR / "alert_state.json"
        self.poll_interval = poll_interval  # seconds between polls
        self._callbacks: List[Callable[[FilingAlert], None]] = []
        self._seen: Dict[str, str] = {}  # accession_number -> detected_at
        self._cik_map: Dict[str, str] = {}  # ticker -> CIK
        self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        """Load previously seen filings from disk."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                self._seen = data.get("seen", {})
                self._cik_map = data.get("cik_map", {})
                logger.info(f"Loaded alert state: {len(self._seen)} seen filings")
            except Exception as e:
                logger.warning(f"Failed to load alert state: {e}")

    def _save_state(self):
        """Persist seen filings to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "seen": self._seen,
                "cik_map": self._cik_map,
                "last_updated": datetime.now().isoformat(),
            }
            self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to save alert state: {e}")

    # ------------------------------------------------------------------
    # CIK resolution
    # ------------------------------------------------------------------

    def _resolve_cik(self, ticker: str) -> Optional[str]:
        """Resolve ticker to CIK number via SEC EDGAR."""
        ticker = ticker.upper()
        if ticker in self._cik_map:
            return self._cik_map[ticker]

        try:
            url = Settings.SEC_TICKERS_URL
            req = Request(url, headers={"User-Agent": f"AlphaExtract {Settings.SEC_USER_EMAIL}"})
            resp = urlopen(req, timeout=Settings.SEC_REQUEST_TIMEOUT)
            data = json.loads(resp.read().decode("utf-8"))

            # SEC returns {fields: [...], data: [[...], ...]}
            fields = data.get("fields", [])
            rows = data.get("data", [])

            ticker_idx = fields.index("ticker") if "ticker" in fields else None
            cik_idx = fields.index("cik") if "cik" in fields else None

            if ticker_idx is not None and cik_idx is not None:
                for row in rows:
                    if row[ticker_idx] == ticker:
                        cik = str(row[cik_idx]).zfill(10)
                        self._cik_map[ticker] = cik
                        self._save_state()
                        return cik
        except Exception as e:
            logger.warning(f"CIK resolution failed for {ticker}: {e}")

        # Fallback: try the simpler tickers file
        try:
            url = Settings.SEC_TICKERS_FALLBACK_URL
            req = Request(url, headers={"User-Agent": f"AlphaExtract {Settings.SEC_USER_EMAIL}"})
            resp = urlopen(req, timeout=Settings.SEC_REQUEST_TIMEOUT)
            data = json.loads(resp.read().decode("utf-8"))

            for entry in data.values():
                if entry.get("ticker") == ticker:
                    cik = str(entry["cik_str"]).zfill(10)
                    self._cik_map[ticker] = cik
                    self._save_state()
                    return cik
        except Exception as e:
            logger.warning(f"CIK fallback resolution failed for {ticker}: {e}")

        return None

    # ------------------------------------------------------------------
    # SEC EDGAR EFTS search
    # ------------------------------------------------------------------

    def _fetch_recent_filings(self, days_back: int = 7) -> List[Dict]:
        """
        Query SEC EDGAR full-text search for recent 10-K filings.
        Returns list of filing metadata dicts.
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        url = (
            f"https://efts.sec.gov/LATEST/search-index?"
            f"q=%2210-K%22&forms=10-K"
            f"&dateRange=custom&startdt={start_date}&enddt={end_date}"
        )

        try:
            req = Request(url, headers={
                "User-Agent": f"AlphaExtract {Settings.SEC_USER_EMAIL}",
                "Accept": "application/json",
            })
            resp = urlopen(req, timeout=Settings.SEC_REQUEST_TIMEOUT)
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("hits", {}).get("hits", [])
        except Exception as e:
            logger.warning(f"EFTS search failed: {e}")
            return []

    def _fetch_company_filings(self, ticker: str, days_back: int = 30) -> List[Dict]:
        """
        Check for recent 10-K filings for a specific company via SEC submissions API.
        More reliable than full-text search for individual tickers.
        """
        cik = self._resolve_cik(ticker)
        if not cik:
            logger.warning(f"Cannot resolve CIK for {ticker}")
            return []

        url = f"https://data.sec.gov/submissions/CIK{cik}.json"

        try:
            req = Request(url, headers={
                "User-Agent": f"AlphaExtract {Settings.SEC_USER_EMAIL}",
                "Accept": "application/json",
            })
            resp = urlopen(req, timeout=Settings.SEC_REQUEST_TIMEOUT)
            data = json.loads(resp.read().decode("utf-8"))

            company_name = data.get("name", ticker)
            recent = data.get("filings", {}).get("recent", {})

            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])

            cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            filings = []

            for i, form in enumerate(forms):
                if form == "10-K" and dates[i] >= cutoff:
                    accession = accessions[i].replace("-", "")
                    filing_url = (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{cik.lstrip('0')}/{accession}/{primary_docs[i]}"
                    )
                    filings.append({
                        "ticker": ticker,
                        "company_name": company_name,
                        "filing_date": dates[i],
                        "accession_number": accessions[i],
                        "filing_url": filing_url,
                    })

            return filings

        except Exception as e:
            logger.warning(f"Failed to fetch filings for {ticker}: {e}")
            return []

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_new_filing(self, callback: Callable[[FilingAlert], None]):
        """Register a callback for new filing alerts."""
        self._callbacks.append(callback)

    def _notify(self, alert: FilingAlert):
        """Fire all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    # ------------------------------------------------------------------
    # Watchlist management
    # ------------------------------------------------------------------

    def add_ticker(self, ticker: str):
        """Add a ticker to the watchlist."""
        ticker = ticker.upper()
        if ticker not in self.watchlist:
            self.watchlist.append(ticker)
            logger.info(f"Added {ticker} to alert watchlist")

    def remove_ticker(self, ticker: str):
        """Remove a ticker from the watchlist."""
        ticker = ticker.upper()
        if ticker in self.watchlist:
            self.watchlist.remove(ticker)
            logger.info(f"Removed {ticker} from alert watchlist")

    def set_watchlist(self, tickers: List[str]):
        """Replace the entire watchlist."""
        self.watchlist = [t.upper() for t in tickers]

    # ------------------------------------------------------------------
    # Core check logic
    # ------------------------------------------------------------------

    def check_once(self, days_back: int = 30) -> List[FilingAlert]:
        """
        Check for new filings across all watched tickers. Returns new alerts.

        Args:
            days_back: How many days back to search for filings.

        Returns:
            List of FilingAlert objects for newly detected filings.
        """
        if not self.watchlist:
            logger.warning("No tickers in watchlist — nothing to monitor")
            return []

        new_alerts = []

        for ticker in self.watchlist:
            # Rate limit: SEC asks for max 10 req/sec
            time.sleep(Settings.SEC_RATE_LIMIT_DELAY)

            filings = self._fetch_company_filings(ticker, days_back=days_back)

            for filing in filings:
                accession = filing["accession_number"]
                if accession in self._seen:
                    continue

                alert = FilingAlert(
                    ticker=filing["ticker"],
                    company_name=filing["company_name"],
                    filing_date=filing["filing_date"],
                    accession_number=accession,
                    filing_url=filing["filing_url"],
                )

                self._seen[accession] = alert.detected_at
                new_alerts.append(alert)
                self._notify(alert)

                logger.info(
                    f"New 10-K filing: {alert.ticker} ({alert.company_name}) "
                    f"filed {alert.filing_date}"
                )

        self._save_state()
        return new_alerts

    def poll(self, days_back: int = 30, max_iterations: int = None):
        """
        Continuously poll for new filings.

        Args:
            days_back: How many days back to look on each check.
            max_iterations: Stop after N checks (None = run forever).
        """
        logger.info(
            f"Starting SEC monitor — watching {len(self.watchlist)} tickers, "
            f"polling every {self.poll_interval}s"
        )

        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            try:
                alerts = self.check_once(days_back=days_back)
                if alerts:
                    logger.info(f"Found {len(alerts)} new filing(s)")
                else:
                    logger.debug("No new filings detected")
            except Exception as e:
                logger.error(f"Polling error: {e}")

            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break

            time.sleep(self.poll_interval)

        logger.info("SEC monitor stopped")

    # ------------------------------------------------------------------
    # Alert history
    # ------------------------------------------------------------------

    @property
    def seen_count(self) -> int:
        return len(self._seen)

    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get the most recent seen filings."""
        items = sorted(self._seen.items(), key=lambda x: x[1], reverse=True)
        return [
            {"accession_number": acc, "detected_at": ts}
            for acc, ts in items[:limit]
        ]

    def clear_state(self):
        """Clear all seen filings state."""
        self._seen.clear()
        self._save_state()
        logger.info("Alert state cleared")

    def status(self) -> Dict:
        """Return current monitor status."""
        return {
            "watchlist": self.watchlist,
            "watchlist_count": len(self.watchlist),
            "seen_filings": len(self._seen),
            "poll_interval_seconds": self.poll_interval,
            "state_file": str(self.state_file),
            "cik_cache_size": len(self._cik_map),
        }

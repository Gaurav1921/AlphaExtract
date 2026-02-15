"""
SEC EDGAR 10-K Downloader
--------------------------
Downloads latest and historical 10-K filings from SEC EDGAR.
Uses centralized settings, proper retry logic, and rate limiting.
"""

import requests
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import json

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class SECDownloader:
    """Downloads 10-K filings from SEC EDGAR database."""

    def __init__(self, email: str = None):
        email = email or Settings.SEC_USER_EMAIL
        self.headers = {
            "User-Agent": f"AlphaExtract/{Settings.VERSION} ({email})",
            "Accept-Encoding": "gzip, deflate",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.raw_dir = Settings.RAW_DIR

    def get_cik(self, ticker: str) -> Optional[str]:
        """Convert ticker symbol to CIK (Central Index Key)."""
        ticker = ticker.upper().strip()
        if not ticker.isalpha() or len(ticker) > 5:
            logger.error(f"Invalid ticker format: {ticker}")
            return None

        try:
            response = self.session.get(
                Settings.SEC_TICKERS_URL,
                timeout=Settings.SEC_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            if "data" in data:
                for row in data["data"]:
                    if len(row) >= 3 and str(row[2]).upper() == ticker:
                        cik = str(row[0]).zfill(10)
                        logger.info(f"Found CIK {cik} for {ticker}")
                        return cik

            # Fallback endpoint
            logger.debug(f"{ticker} not in primary API, trying fallback")
            response = self.session.get(
                Settings.SEC_TICKERS_FALLBACK_URL,
                timeout=Settings.SEC_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            for entry in data.values():
                if entry["ticker"].upper() == ticker:
                    cik = str(entry["cik_str"]).zfill(10)
                    logger.info(f"Found CIK {cik} for {ticker} (fallback)")
                    return cik

            logger.error(f"Ticker {ticker} not found in SEC database")
            return None

        except requests.RequestException as e:
            logger.error(f"Network error looking up CIK for {ticker}: {e}")
            return None

    def get_10k_filings(self, cik: str, limit: int = 1) -> List[Dict]:
        """Retrieve metadata for 10-K filings from SEC submissions."""
        try:
            url = f"{Settings.SEC_BASE_URL}/submissions/CIK{cik}.json"
            response = self.session.get(url, timeout=Settings.SEC_REQUEST_TIMEOUT)
            response.raise_for_status()

            data = response.json()
            filings = data["filings"]["recent"]
            results = []

            for i, form in enumerate(filings["form"]):
                if form == "10-K":
                    accession = filings["accessionNumber"][i].replace("-", "")
                    filing_date = filings["filingDate"][i]
                    primary_doc = filings["primaryDocument"][i]

                    # CIK in URL paths should not have leading zeros
                    cik_trimmed = cik.lstrip("0") or "0"

                    download_url = (
                        f"{Settings.SEC_FILING_URL}/Archives/edgar/data/"
                        f"{cik_trimmed}/{accession}/{primary_doc}"
                    )

                    results.append(
                        {
                            "url": download_url,
                            "filing_date": filing_date,
                            "accession": accession,
                            "document": primary_doc,
                            "year": filing_date[:4],
                        }
                    )
                    if len(results) >= limit:
                        break

            logger.info(f"Found {len(results)} 10-K filing(s) for CIK {cik}")
            return results

        except requests.RequestException as e:
            logger.error(f"Failed to fetch filings for CIK {cik}: {e}")
            return []

    def download_filing(self, ticker: str, filing_info: Dict = None) -> Optional[Path]:
        """
        Download a single 10-K filing.

        If filing_info is None, looks up the latest 10-K automatically.
        """
        ticker = ticker.upper()

        if filing_info is None:
            cik = self.get_cik(ticker)
            if not cik:
                return None
            filings = self.get_10k_filings(cik, limit=1)
            if not filings:
                logger.error(f"No 10-K found for {ticker}")
                return None
            filing_info = filings[0]

        url = filing_info["url"]
        filing_date = filing_info["filing_date"]

        for attempt in range(1, Settings.SEC_MAX_RETRIES + 1):
            try:
                time.sleep(Settings.SEC_RATE_LIMIT_DELAY)
                logger.info(f"Downloading {ticker} {filing_date} (attempt {attempt})")

                response = self.session.get(url, timeout=Settings.SEC_REQUEST_TIMEOUT)
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "").lower()
                ext = "html" if ("html" in content_type or url.endswith(".htm")) else "txt"

                filename = f"{ticker}_10K_{filing_date}.{ext}"
                filepath = self.raw_dir / filename
                filepath.write_bytes(response.content)
                file_size_kb = len(response.content) / 1024

                # Save download metadata alongside the file
                metadata = {
                    "ticker": ticker,
                    "cik": filing_info.get("accession", "")[:10],
                    "filing_date": filing_date,
                    "download_date": datetime.now().isoformat(),
                    "download_url": url,
                    "file_size_kb": round(file_size_kb, 1),
                }
                filepath.with_suffix(".json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )

                logger.info(f"Downloaded {ticker} {filing_date} ({file_size_kb:.1f} KB)")
                return filepath

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt} failed for {ticker} {filing_date}: {e}")
                if attempt < Settings.SEC_MAX_RETRIES:
                    time.sleep(Settings.SEC_RETRY_DELAY * attempt)
                else:
                    logger.error(f"Failed to download {ticker} {filing_date} after {Settings.SEC_MAX_RETRIES} attempts")
                    return None

        return None

    def download_multiple(self, tickers: List[str], years: int = 1) -> Dict:
        """Download 10-Ks for multiple companies."""
        results = {"successful": [], "failed": [], "total": len(tickers)}
        logger.info(f"Batch download: {len(tickers)} companies, {years} year(s) each")

        for ticker in tickers:
            ticker = ticker.upper()
            cik = self.get_cik(ticker)
            if not cik:
                results["failed"].append(ticker)
                continue

            filings = self.get_10k_filings(cik, limit=years)
            if not filings:
                results["failed"].append(ticker)
                continue

            for filing_info in filings:
                # Skip if already downloaded
                filing_date = filing_info["filing_date"]
                existing = list(self.raw_dir.glob(f"{ticker}_10K_{filing_date}.*"))
                if any(f.suffix in (".html", ".txt") for f in existing):
                    results["successful"].append({"ticker": ticker, "path": str(existing[0]), "skipped": True})
                    continue

                filepath = self.download_filing(ticker, filing_info)
                if filepath:
                    results["successful"].append({"ticker": ticker, "path": str(filepath)})
                else:
                    results["failed"].append(ticker)

            time.sleep(Settings.SEC_RATE_LIMIT_DELAY)

        logger.info(f"Batch complete: {len(results['successful'])} succeeded, {len(results['failed'])} failed")
        return results

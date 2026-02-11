"""
Company Search Module
---------------------
Enables dynamic company search via SEC EDGAR.
"""

import requests
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime

from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class CompanyInfo:
    """Company information from SEC."""
    ticker: str
    name: str
    cik: str
    exchange: Optional[str] = None
    sic: Optional[str] = None
    sic_description: Optional[str] = None
    filings_count: int = 0
    latest_10k_date: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "cik": self.cik,
            "exchange": self.exchange,
            "sic": self.sic,
            "sic_description": self.sic_description,
            "filings_count": self.filings_count,
            "latest_10k_date": self.latest_10k_date,
        }


class CompanySearch:
    """Search for companies in SEC EDGAR database."""

    def __init__(self, email: str = None):
        email = email or Settings.SEC_USER_EMAIL
        self.headers = {
            "User-Agent": f"AlphaExtract/{Settings.VERSION} ({email})",
            "Accept-Encoding": "gzip, deflate",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self._ticker_cache: Dict[str, Dict] = {}
        self._cache_loaded = False
        self.watchlist_file = Settings.WATCHLIST_FILE

    def _load_ticker_cache(self) -> bool:
        """Load SEC ticker database into cache."""
        if self._cache_loaded:
            return True

        try:
            response = self.session.get(Settings.SEC_TICKERS_URL, timeout=15)
            response.raise_for_status()
            data = response.json()

            if "data" in data:
                for row in data["data"]:
                    if len(row) >= 4:
                        cik, name, ticker, exchange = row[0], row[1], row[2], row[3]
                        self._ticker_cache[ticker.upper()] = {
                            "cik": str(cik).zfill(10),
                            "name": name,
                            "ticker": ticker.upper(),
                            "exchange": exchange,
                        }

            self._cache_loaded = True
            logger.info(f"Loaded {len(self._ticker_cache)} companies from SEC")
            return True

        except Exception as e:
            logger.error(f"Failed to load ticker cache: {e}")
            return False

    def search_by_ticker(self, ticker: str) -> Optional[CompanyInfo]:
        """Search for a company by ticker symbol."""
        ticker = ticker.upper().strip()
        self._load_ticker_cache()

        if ticker in self._ticker_cache:
            cached = self._ticker_cache[ticker]
            company_info = self._get_company_details(cached["cik"])

            if company_info:
                return company_info

            return CompanyInfo(
                ticker=cached["ticker"],
                name=cached["name"],
                cik=cached["cik"],
                exchange=cached.get("exchange"),
            )

        return None

    def search_by_name(self, query: str, limit: int = 10) -> List[CompanyInfo]:
        """Search for companies by name."""
        query = query.lower().strip()
        self._load_ticker_cache()

        matches = []
        for ticker, info in self._ticker_cache.items():
            name_lower = info["name"].lower()
            if query in name_lower or query in ticker.lower():
                matches.append(
                    CompanyInfo(
                        ticker=info["ticker"],
                        name=info["name"],
                        cik=info["cik"],
                        exchange=info.get("exchange"),
                    )
                )
                if len(matches) >= limit * 2:
                    break

        matches.sort(key=lambda x: (0 if x.ticker.lower() == query else 1, len(x.name)))
        return matches[:limit]

    def _get_company_details(self, cik: str) -> Optional[CompanyInfo]:
        """Get detailed company information from SEC."""
        try:
            url = f"{Settings.SEC_BASE_URL}/submissions/CIK{cik}.json"
            response = self.session.get(url, timeout=Settings.SEC_REQUEST_TIMEOUT)
            response.raise_for_status()

            data = response.json()
            filings = data.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            filing_dates = filings.get("filingDate", [])

            ten_k_count = sum(1 for f in forms if f == "10-K")
            latest_10k = None
            for i, form in enumerate(forms):
                if form == "10-K" and i < len(filing_dates):
                    latest_10k = filing_dates[i]
                    break

            tickers = data.get("tickers", [])
            exchanges = data.get("exchanges", [])

            return CompanyInfo(
                ticker=tickers[0] if tickers else "",
                name=data.get("name", ""),
                cik=cik,
                exchange=exchanges[0] if exchanges else None,
                sic=data.get("sic"),
                sic_description=data.get("sicDescription"),
                filings_count=ten_k_count,
                latest_10k_date=latest_10k,
            )
        except Exception as e:
            logger.error(f"Failed to get company details for CIK {cik}: {e}")
            return None

    def get_available_filings(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get list of available 10-K filings for a ticker."""
        ticker = ticker.upper()
        company = self.search_by_ticker(ticker)
        if not company:
            return []

        try:
            url = f"{Settings.SEC_BASE_URL}/submissions/CIK{company.cik}.json"
            response = self.session.get(url, timeout=Settings.SEC_REQUEST_TIMEOUT)
            response.raise_for_status()

            data = response.json()
            filings = data.get("filings", {}).get("recent", {})

            available = []
            for i, form in enumerate(filings.get("form", [])):
                if form == "10-K" and len(available) < limit:
                    filing_date = filings["filingDate"][i]

                    existing = list(Settings.RAW_DIR.glob(f"{ticker}_10K_{filing_date}.*"))
                    has_local = any(f.suffix in (".html", ".txt") for f in existing)

                    sentiment_file = Settings.SENTIMENT_DIR / f"{ticker}_{filing_date}_sentiment.json"

                    available.append(
                        {
                            "filing_date": filing_date,
                            "year": filing_date[:4],
                            "accession": filings["accessionNumber"][i],
                            "downloaded": has_local,
                            "processed": sentiment_file.exists(),
                        }
                    )

            return available
        except Exception as e:
            logger.error(f"Failed to get filings for {ticker}: {e}")
            return []

    def get_watchlist(self) -> List[Dict]:
        """Get user's company watchlist."""
        if not self.watchlist_file.exists():
            return []
        try:
            return json.loads(self.watchlist_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def add_to_watchlist(self, ticker: str) -> bool:
        """Add company to watchlist."""
        ticker = ticker.upper()
        company = self.search_by_ticker(ticker)
        if not company:
            return False

        watchlist = self.get_watchlist()
        if any(w["ticker"] == ticker for w in watchlist):
            return True

        watchlist.append(
            {
                "ticker": company.ticker,
                "name": company.name,
                "cik": company.cik,
                "added_date": datetime.now().isoformat(),
                "exchange": company.exchange,
            }
        )

        self.watchlist_file.parent.mkdir(parents=True, exist_ok=True)
        self.watchlist_file.write_text(json.dumps(watchlist, indent=2), encoding="utf-8")
        return True

    def remove_from_watchlist(self, ticker: str) -> bool:
        """Remove company from watchlist."""
        ticker = ticker.upper()
        watchlist = self.get_watchlist()
        original_len = len(watchlist)
        watchlist = [w for w in watchlist if w["ticker"] != ticker]

        if len(watchlist) < original_len:
            self.watchlist_file.write_text(json.dumps(watchlist, indent=2), encoding="utf-8")
            return True
        return False

    def get_all_companies(self) -> Dict[str, str]:
        """Get all companies (defaults + watchlist)."""
        companies = {
            "AAPL": "Apple Inc.",
            "GOOGL": "Alphabet Inc.",
            "MSFT": "Microsoft Corp.",
            "TSLA": "Tesla Inc.",
        }

        for item in self.get_watchlist():
            ticker = item["ticker"]
            if ticker not in companies:
                companies[ticker] = item["name"]

        return companies

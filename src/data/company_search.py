"""
Company Search Module
---------------------
LOCATION: src/data/company_search.py

Enables dynamic company search via SEC EDGAR.
Users can search for ANY SEC-registered company by ticker or name.
"""

import requests
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
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
            'ticker': self.ticker,
            'name': self.name,
            'cik': self.cik,
            'exchange': self.exchange,
            'sic': self.sic,
            'sic_description': self.sic_description,
            'filings_count': self.filings_count,
            'latest_10k_date': self.latest_10k_date
        }


class CompanySearch:
    """
    Search for companies in SEC EDGAR database.
    
    Usage:
        search = CompanySearch()
        results = search.search_by_ticker("NVDA")
        results = search.search_by_name("NVIDIA")
    """
    
    def __init__(self, email: str = "student@example.com"):
        self.base_url = "https://data.sec.gov"
        self.headers = {
            "User-Agent": f"AlphaExtract/1.0 ({email})",
            "Accept-Encoding": "gzip, deflate"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self._ticker_cache: Dict[str, Dict] = {}
        self._cache_loaded = False
        
        self.watchlist_file = Path("data/watchlist.json")
    
    def _load_ticker_cache(self) -> bool:
        """Load SEC ticker database into cache."""
        if self._cache_loaded:
            return True
        
        try:
            url = "https://www.sec.gov/files/company_tickers_exchange.json"
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data:
                for row in data['data']:
                    if len(row) >= 4:
                        cik, name, ticker, exchange = row[0], row[1], row[2], row[3]
                        self._ticker_cache[ticker.upper()] = {
                            'cik': str(cik).zfill(10),
                            'name': name,
                            'ticker': ticker.upper(),
                            'exchange': exchange
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
            company_info = self._get_company_details(cached['cik'])
            
            if company_info:
                return company_info
            
            return CompanyInfo(
                ticker=cached['ticker'],
                name=cached['name'],
                cik=cached['cik'],
                exchange=cached.get('exchange')
            )
        
        return None
    
    def search_by_name(self, query: str, limit: int = 10) -> List[CompanyInfo]:
        """Search for companies by name."""
        query = query.lower().strip()
        self._load_ticker_cache()
        
        matches = []
        for ticker, info in self._ticker_cache.items():
            name_lower = info['name'].lower()
            if query in name_lower or query in ticker.lower():
                matches.append(CompanyInfo(
                    ticker=info['ticker'],
                    name=info['name'],
                    cik=info['cik'],
                    exchange=info.get('exchange')
                ))
                if len(matches) >= limit:
                    break
        
        matches.sort(key=lambda x: (0 if x.ticker.lower() == query else 1, len(x.name)))
        return matches
    
    def _get_company_details(self, cik: str) -> Optional[CompanyInfo]:
        """Get detailed company information from SEC."""
        try:
            url = f"{self.base_url}/submissions/CIK{cik}.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            filings = data.get('filings', {}).get('recent', {})
            forms = filings.get('form', [])
            filing_dates = filings.get('filingDate', [])
            
            ten_k_count = sum(1 for f in forms if f == '10-K')
            latest_10k = None
            for i, form in enumerate(forms):
                if form == '10-K':
                    latest_10k = filing_dates[i]
                    break
            
            return CompanyInfo(
                ticker=data.get('tickers', [''])[0] if data.get('tickers') else '',
                name=data.get('name', ''),
                cik=cik,
                exchange=data.get('exchanges', [''])[0] if data.get('exchanges') else None,
                sic=data.get('sic'),
                sic_description=data.get('sicDescription'),
                filings_count=ten_k_count,
                latest_10k_date=latest_10k
            )
        except Exception as e:
            logger.error(f"Failed to get company details: {e}")
            return None
    
    def get_available_filings(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get list of available 10-K filings for a ticker."""
        ticker = ticker.upper()
        company = self.search_by_ticker(ticker)
        if not company:
            return []
        
        try:
            url = f"{self.base_url}/submissions/CIK{company.cik}.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            filings = data.get('filings', {}).get('recent', {})
            
            available = []
            for i, form in enumerate(filings.get('form', [])):
                if form == '10-K' and len(available) < limit:
                    filing_date = filings['filingDate'][i]
                    
                    raw_dir = Path("data/raw")
                    existing = list(raw_dir.glob(f"{ticker}_10K_{filing_date}.*"))
                    has_local = any(f.suffix in ['.html', '.txt'] for f in existing)
                    
                    sentiment_file = Path("data/sentiment") / f"{ticker}_{filing_date}_sentiment.json"
                    has_sentiment = sentiment_file.exists()
                    
                    available.append({
                        'filing_date': filing_date,
                        'year': filing_date[:4],
                        'accession': filings['accessionNumber'][i],
                        'downloaded': has_local,
                        'processed': has_sentiment
                    })
            
            return available
        except Exception as e:
            logger.error(f"Failed to get filings: {e}")
            return []
    
    def get_watchlist(self) -> List[Dict]:
        """Get user's company watchlist."""
        if not self.watchlist_file.exists():
            return []
        try:
            with open(self.watchlist_file, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def add_to_watchlist(self, ticker: str) -> bool:
        """Add company to watchlist."""
        ticker = ticker.upper()
        company = self.search_by_ticker(ticker)
        if not company:
            return False
        
        watchlist = self.get_watchlist()
        if any(w['ticker'] == ticker for w in watchlist):
            return True
        
        watchlist.append({
            'ticker': company.ticker,
            'name': company.name,
            'cik': company.cik,
            'added_date': datetime.now().isoformat(),
            'exchange': company.exchange
        })
        
        self.watchlist_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.watchlist_file, 'w') as f:
            json.dump(watchlist, f, indent=2)
        
        return True
    
    def remove_from_watchlist(self, ticker: str) -> bool:
        """Remove company from watchlist."""
        ticker = ticker.upper()
        watchlist = self.get_watchlist()
        original_len = len(watchlist)
        watchlist = [w for w in watchlist if w['ticker'] != ticker]
        
        if len(watchlist) < original_len:
            with open(self.watchlist_file, 'w') as f:
                json.dump(watchlist, f, indent=2)
            return True
        return False
    
    def get_all_companies(self) -> Dict[str, str]:
        """Get all companies (default + watchlist)."""
        companies = {
            'AAPL': '🍎 Apple Inc.',
            'GOOGL': '🔍 Alphabet Inc.',
            'MSFT': '🪟 Microsoft Corp.',
            'TSLA': '⚡ Tesla Inc.'
        }
        
        emoji_map = {
            'A': '🅰️', 'B': '🅱️', 'C': '©️', 'D': '🔷', 'E': '📧',
            'F': '🎏', 'G': '🟢', 'H': '♓', 'I': 'ℹ️', 'J': '🎷',
            'K': '🔑', 'L': '🔷', 'M': 'Ⓜ️', 'N': '🔷', 'O': '⭕',
            'P': '🅿️', 'Q': '🔷', 'R': '®️', 'S': '💲', 'T': '✝️',
            'U': '⛎', 'V': '✅', 'W': '🔷', 'X': '❌', 'Y': '🔱', 'Z': '💤'
        }
        
        for item in self.get_watchlist():
            ticker = item['ticker']
            if ticker not in companies:
                emoji = emoji_map.get(ticker[0].upper(), '📊')
                companies[ticker] = f"{emoji} {item['name']}"
        
        return companies
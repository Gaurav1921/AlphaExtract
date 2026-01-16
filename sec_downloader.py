"""
SEC EDGAR 10-K Downloader
--------------------------
Fetches the latest 10-K filing for any public company.

Learning objectives:
- HTTP requests with proper headers (SEC requirement)
- JSON API consumption
- Error handling with retries
- File I/O and directory management
- Logging for production debugging
"""

import requests
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SECDownloader:
    """
    Downloads 10-K filings from SEC EDGAR database.
    
    Design principles:
    - Retry logic for network failures
    - Proper SEC User-Agent headers
    - Rate limiting (10 req/sec max)
    - No hardcoded ticker mappings
    """
    
    def __init__(self, email: str = "student@example.com"):
        """
        Initialize downloader with required SEC headers.
        
        Args:
            email: Your email (SEC requirement for identification)
        """
        self.base_url = "https://www.sec.gov"
        self.data_url = "https://data.sec.gov"  # For API endpoints
        self.headers = {
            "User-Agent": f"AlphaExtract/1.0 ({email})",
            "Accept-Encoding": "gzip, deflate"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Create directories
        self.raw_dir = Path("data/raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cik(self, ticker: str) -> Optional[str]:
        """
        Convert ticker symbol to CIK (Central Index Key).
        
        Uses SEC's company tickers JSON file for lookup.
        Falls back to alternative endpoint if primary fails.
        
        Args:
            ticker: Stock ticker (e.g., 'TSLA', 'AAPL')
            
        Returns:
            CIK number as string, or None if not found
        """
        try:
            # Try the Exchange API endpoint (more reliable)
            url = "https://www.sec.gov/files/company_tickers_exchange.json"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Format: {"fields": [...], "data": [[cik, name, ticker, exchange], ...]}
            if 'data' in data:
                for row in data['data']:
                    # row = [cik, company_name, ticker, exchange]
                    if len(row) >= 3 and row[2].upper() == ticker.upper():
                        cik = str(row[0]).zfill(10)
                        logger.info(f"Found CIK {cik} for ticker {ticker}")
                        return cik
            
            logger.warning(f"Ticker {ticker} not found in primary API, trying fallback...")
            
            # Fallback: Try the older company_tickers.json format
            fallback_url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(fallback_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            for entry in data.values():
                if entry['ticker'].upper() == ticker.upper():
                    cik = str(entry['cik_str']).zfill(10)
                    logger.info(f"Found CIK {cik} for ticker {ticker} (fallback)")
                    return cik
            
            logger.error(f"Ticker {ticker} not found in SEC database")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get CIK for {ticker}: {e}")
            logger.info("Tip: You can find CIKs manually at https://www.sec.gov/edgar/searchedgar/companysearch.html")
            return None
    
    def get_latest_10k_url(self, cik: str) -> Optional[dict]:
        """
        Find the most recent 10-K filing URL.
        
        Args:
            cik: Company's CIK number
            
        Returns:
            Dict with filing metadata or None if not found
        """
        try:
            # SEC's submissions endpoint returns ALL filings for a company
            # Use data.sec.gov for API calls
            url = f"{self.data_url}/submissions/CIK{cik}.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            filings = data['filings']['recent']
            
            # Find the first 10-K (not 10-K/A which is an amendment)
            for i, form in enumerate(filings['form']):
                if form == '10-K':
                    accession = filings['accessionNumber'][i].replace('-', '')
                    filing_date = filings['filingDate'][i]
                    primary_doc = filings['primaryDocument'][i]
                    
                    # Remove leading zeros from CIK for URL (SEC uses trimmed CIK in paths)
                    cik_trimmed = cik.lstrip('0')
                    
                    # Construct TWO URLs:
                    # 1. Raw HTML for programmatic download (what we actually download)
                    download_url = (
                        f"{self.base_url}/Archives/edgar/data/"
                        f"{cik_trimmed}/{accession}/{primary_doc}"
                    )
                    
                    # 2. Interactive viewer (for user reference, saved in metadata)
                    viewer_url = (
                        f"{self.base_url}/ix?doc=/Archives/edgar/data/"
                        f"{cik_trimmed}/{accession}/{primary_doc}"
                    )
                    
                    logger.info(f"Found 10-K filed on {filing_date}")
                    logger.info(f"Download URL: {download_url}")
                    
                    return {
                        'url': download_url,  # Use raw URL for download
                        'viewer_url': viewer_url,  # Save for reference
                        'filing_date': filing_date,
                        'accession': accession,
                        'document': primary_doc
                    }
            
            logger.error(f"No 10-K found for CIK {cik}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get 10-K URL for CIK {cik}: {e}")
            return None
            
            logger.error(f"No 10-K found for CIK {cik}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get 10-K URL for CIK {cik}: {e}")
            return None
    
    def download_filing(
        self, 
        ticker: str, 
        max_retries: int = 3,
        retry_delay: int = 2
    ) -> Optional[Path]:
        """
        Download the latest 10-K for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            max_retries: Number of retry attempts on failure
            retry_delay: Seconds to wait between retries
            
        Returns:
            Path to downloaded file, or None if failed
        """
        logger.info(f"Starting download for {ticker}")
        
        # Step 1: Get CIK
        cik = self.get_cik(ticker)
        if not cik:
            return None
        
        # Step 2: Get 10-K URL
        filing_info = self.get_latest_10k_url(cik)
        if not filing_info:
            return None
        
        # Step 3: Download with retries
        url = filing_info['url']
        filing_date = filing_info['filing_date']
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Downloading from {url} (attempt {attempt}/{max_retries})")
                
                # SEC rate limit: 10 requests/second max
                time.sleep(0.1)
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Determine file extension from Content-Type or URL
                content_type = response.headers.get('Content-Type', '')
                if 'html' in content_type or url.endswith('.htm'):
                    ext = 'html'
                else:
                    ext = 'txt'
                
                # Save file
                filename = f"{ticker}_10K_{filing_date}.{ext}"
                filepath = self.raw_dir / filename
                
                filepath.write_bytes(response.content)
                file_size = len(response.content) / 1024  # KB
                
                logger.info(
                    f"✓ Successfully downloaded {ticker} 10-K "
                    f"({file_size:.1f} KB) → {filepath}"
                )
                
                # Save metadata for later use
                metadata = {
                    'ticker': ticker,
                    'cik': cik,
                    'filing_date': filing_date,
                    'download_date': datetime.now().isoformat(),
                    'download_url': url,
                    'viewer_url': filing_info.get('viewer_url', url),
                    'file_size_kb': file_size
                }
                
                metadata_path = filepath.with_suffix('.json')
                metadata_path.write_text(json.dumps(metadata, indent=2))
                
                return filepath
                
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed for {ticker}: {e}"
                )
                
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to download {ticker} after {max_retries} attempts | "
                        f"URL: {url} | "
                        f"Error: {e} | "
                        f"Timestamp: {datetime.now()}"
                    )
                    return None
        
        return None
    
    def download_multiple(self, tickers: list[str]) -> dict:
        """
        Download 10-Ks for multiple companies.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dict with success/failure counts and paths
        """
        results = {
            'successful': [],
            'failed': [],
            'total': len(tickers)
        }
        
        logger.info(f"Starting batch download for {len(tickers)} companies")
        
        for ticker in tickers:
            filepath = self.download_filing(ticker)
            
            if filepath:
                results['successful'].append({'ticker': ticker, 'path': str(filepath)})
            else:
                results['failed'].append(ticker)
            
            # Be nice to SEC servers
            time.sleep(0.1)
        
        logger.info(
            f"Batch download complete: "
            f"{len(results['successful'])} succeeded, "
            f"{len(results['failed'])} failed"
        )
        
        return results


# ============================================================================
# TESTING CODE - Run this file directly to test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SEC EDGAR Downloader - Test Suite")
    print("=" * 70)
    
    # Initialize downloader
    downloader = SECDownloader(email="your.email@example.com")
    
    # Test 1: Single company download
    print("\n[TEST 1] Downloading Tesla (TSLA) 10-K...")
    tesla_path = downloader.download_filing("TSLA")
    
    if tesla_path:
        print(f"✓ Success! File saved to: {tesla_path}")
        print(f"  File size: {tesla_path.stat().st_size / 1024:.1f} KB")
    else:
        print("✗ Failed to download")
    
    # Test 2: Batch download
    print("\n[TEST 2] Downloading multiple companies...")
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    results = downloader.download_multiple(test_tickers)
    
    print(f"\nResults:")
    print(f"  Successful: {len(results['successful'])}")
    print(f"  Failed: {len(results['failed'])}")
    
    if results['successful']:
        print("\n  Downloaded files:")
        for item in results['successful']:
            print(f"    • {item['ticker']}: {item['path']}")
    
    if results['failed']:
        print(f"\n  Failed tickers: {', '.join(results['failed'])}")
    
    print("\n" + "=" * 70)
    print("Test complete! Check the data/raw/ directory for files.")
    print("=" * 70)
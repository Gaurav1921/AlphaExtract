"""
Enhanced SEC Downloader with Historical Support
-----------------------------------------------
Downloads multiple years of 10-K filings for a company.
"""

import requests
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedSECDownloader:
    """
    Downloads 10-K filings from SEC EDGAR with historical support.
    """
    
    def __init__(self, email: str = "student@example.com"):
        self.base_url = "https://data.sec.gov"
        self.headers = {
            "User-Agent": f"AlphaExtract/1.0 ({email})",
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.raw_dir = Path("data/raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cik(self, ticker: str) -> Optional[str]:
        """Convert ticker to CIK."""
        try:
            url = f"{self.base_url}/files/company_tickers.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            for entry in data.values():
                if entry['ticker'].upper() == ticker.upper():
                    cik = str(entry['cik_str']).zfill(10)
                    logger.info(f"Found CIK {cik} for ticker {ticker}")
                    return cik
            
            logger.error(f"Ticker {ticker} not found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get CIK for {ticker}: {e}")
            return None
    
    def get_all_10k_filings(self, cik: str, limit: int = None) -> List[Dict]:
        """
        Get all 10-K filings for a company.
        
        Args:
            cik: Company CIK
            limit: Maximum number of filings to return (None = all)
            
        Returns:
            List of filing metadata dicts
        """
        try:
            url = f"{self.base_url}/submissions/CIK{cik}.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            filings = data['filings']['recent']
            
            all_10ks = []
            
            for i, form in enumerate(filings['form']):
                if form == '10-K':  # Not amendments
                    accession = filings['accessionNumber'][i].replace('-', '')
                    filing_date = filings['filingDate'][i]
                    primary_doc = filings['primaryDocument'][i]
                    
                    doc_url = (
                        f"{self.base_url}/Archives/edgar/data/"
                        f"{cik}/{accession}/{primary_doc}"
                    )
                    
                    all_10ks.append({
                        'url': doc_url,
                        'filing_date': filing_date,
                        'accession': accession,
                        'document': primary_doc,
                        'year': filing_date[:4]
                    })
                    
                    if limit and len(all_10ks) >= limit:
                        break
            
            logger.info(f"Found {len(all_10ks)} 10-K filings for CIK {cik}")
            return all_10ks
            
        except Exception as e:
            logger.error(f"Failed to get filings for CIK {cik}: {e}")
            return []
    
    def download_filing(
        self,
        ticker: str,
        filing_info: Dict,
        max_retries: int = 3
    ) -> Optional[Path]:
        """Download a single filing."""
        
        url = filing_info['url']
        filing_date = filing_info['filing_date']
        
        for attempt in range(1, max_retries + 1):
            try:
                time.sleep(0.15)  # SEC rate limit: ~10 req/sec
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Determine extension
                content_type = response.headers.get('Content-Type', '')
                if 'html' in content_type or url.endswith('.htm'):
                    ext = 'html'
                else:
                    ext = 'txt'
                
                # Save file
                filename = f"{ticker}_10K_{filing_date}.{ext}"
                filepath = self.raw_dir / filename
                
                filepath.write_bytes(response.content)
                file_size = len(response.content) / 1024
                
                logger.info(f"✓ Downloaded {ticker} {filing_date} ({file_size:.1f} KB)")
                
                # Save metadata
                metadata = {
                    'ticker': ticker,
                    'filing_date': filing_date,
                    'download_date': datetime.now().isoformat(),
                    'url': url,
                    'file_size_kb': file_size
                }
                
                metadata_path = filepath.with_suffix('.json')
                metadata_path.write_text(json.dumps(metadata, indent=2))
                
                return filepath
                
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}")
                
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    logger.error(f"Failed to download {ticker} {filing_date}")
                    return None
        
        return None
    
    def download_historical(
        self,
        ticker: str,
        years: int = 5,
        progress_callback=None
    ) -> Dict:
        """
        Download multiple years of 10-K filings.
        
        Args:
            ticker: Stock ticker
            years: Number of years to download (5 = last 5 years)
            progress_callback: Function to call with progress updates
            
        Returns:
            Dict with results
        """
        logger.info(f"Starting historical download for {ticker} ({years} years)")
        
        # Get CIK
        cik = self.get_cik(ticker)
        if not cik:
            return {'success': False, 'error': 'Ticker not found'}
        
        # Get all 10-K filings
        all_filings = self.get_all_10k_filings(cik, limit=years)
        
        if not all_filings:
            return {'success': False, 'error': 'No filings found'}
        
        # Download each filing
        results = {
            'ticker': ticker,
            'total': len(all_filings),
            'successful': [],
            'failed': [],
            'skipped': []
        }
        
        for i, filing_info in enumerate(all_filings):
            filing_date = filing_info['filing_date']
            
            # Check if already downloaded
            existing = list(self.raw_dir.glob(f"{ticker}_10K_{filing_date}.*"))
            if existing and any(f.suffix in ['.html', '.txt'] for f in existing):
                logger.info(f"⏭️  Skipped {ticker} {filing_date} (already exists)")
                results['skipped'].append(filing_date)
            else:
                filepath = self.download_filing(ticker, filing_info)
                
                if filepath:
                    results['successful'].append({
                        'date': filing_date,
                        'path': str(filepath)
                    })
                else:
                    results['failed'].append(filing_date)
            
            # Progress callback
            if progress_callback:
                progress = (i + 1) / len(all_filings)
                progress_callback(progress, i + 1, len(all_filings))
        
        results['success'] = len(results['failed']) == 0
        
        logger.info(
            f"✓ Download complete: {len(results['successful'])} succeeded, "
            f"{len(results['skipped'])} skipped, {len(results['failed'])} failed"
        )
        
        return results
    
    def get_available_filings(self, ticker: str) -> List[Dict]:
        """Get list of available filings for a ticker (both local and remote)."""
        
        # Local files
        local_files = list(self.raw_dir.glob(f"{ticker}_10K_*.html")) + \
                     list(self.raw_dir.glob(f"{ticker}_10K_*.txt"))
        
        local_dates = set()
        for f in local_files:
            parts = f.stem.split('_')
            if len(parts) >= 3:
                local_dates.add(parts[2])  # The date part
        
        # Remote filings
        cik = self.get_cik(ticker)
        remote_filings = []
        
        if cik:
            remote_filings = self.get_all_10k_filings(cik)
        
        # Combine
        all_filings = []
        for filing in remote_filings:
            all_filings.append({
                'date': filing['filing_date'],
                'year': filing['year'],
                'downloaded': filing['filing_date'] in local_dates
            })
        
        return sorted(all_filings, key=lambda x: x['date'], reverse=True)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED SEC DOWNLOADER - HISTORICAL TEST")
    print("=" * 70)
    
    downloader = EnhancedSECDownloader(email="your.email@example.com")
    
    print("\n[TEST] Get available filings for AAPL...")
    filings = downloader.get_available_filings('AAPL')
    
    print(f"\nFound {len(filings)} filings:")
    for filing in filings[:5]:
        status = "✓ Downloaded" if filing['downloaded'] else "○ Available"
        print(f"  {status} - {filing['date']} ({filing['year']})")
    
    print("\n[TEST] Download last 3 years for AAPL...")
    
    def progress(pct, current, total):
        print(f"Progress: {pct*100:.0f}% ({current}/{total})")
    
    results = downloader.download_historical('AAPL', years=3, progress_callback=progress)
    
    print(f"\nResults:")
    print(f"  Successful: {len(results['successful'])}")
    print(f"  Skipped: {len(results['skipped'])}")
    print(f"  Failed: {len(results['failed'])}")
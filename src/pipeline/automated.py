"""
Automated Pipeline - Unified Module
------------------------------------
LOCATION: src/pipeline/automated.py

This REPLACES both:
- root/automated_pipeline.py (DELETE THIS)
- dashboard/automated_pipeline.py (DELETE THIS)

Features:
- One-click download and processing
- Progress tracking with callbacks
- Support for any SEC-listed company
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a pipeline operation."""
    success: bool
    message: str
    data: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass  
class FilingStatus:
    """Status of a single filing's processing stages."""
    filing_date: str
    year: str
    downloaded: bool = False
    parsed: bool = False
    sections_extracted: bool = False
    sentiment_analyzed: bool = False
    indexed: bool = False
    
    @property
    def is_complete(self) -> bool:
        return self.sentiment_analyzed
    
    def to_dict(self) -> Dict:
        return {
            'filing_date': self.filing_date,
            'year': self.year,
            'downloaded': self.downloaded,
            'parsed': self.parsed,
            'sections': self.sections_extracted,
            'sentiment': self.sentiment_analyzed,
            'complete': self.is_complete
        }


def get_filing_status(ticker: str) -> Dict[str, FilingStatus]:
    """Get complete status of all filings for a ticker."""
    ticker = ticker.upper()
    
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    sections_dir = Path("data/sections")
    sentiment_dir = Path("data/sentiment")
    
    # Ensure directories exist
    for d in [raw_dir, processed_dir, sections_dir, sentiment_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Downloaded raw files
    raw_files = list(raw_dir.glob(f"{ticker}_10K_*.html")) + \
                list(raw_dir.glob(f"{ticker}_10K_*.txt"))
    raw_dates = {f.stem.split('_')[2] for f in raw_files if len(f.stem.split('_')) >= 3}
    
    # Processed markdown files
    processed_files = list(processed_dir.glob(f"{ticker}_*.md"))
    processed_dates = {f.stem.split('_')[1] for f in processed_files if len(f.stem.split('_')) >= 2}
    
    # Section files
    section_files = list(sections_dir.glob(f"{ticker}_*_item_*.txt"))
    section_dates = {f.stem.split('_')[1] for f in section_files if len(f.stem.split('_')) >= 2}
    
    # Sentiment files
    sentiment_files = list(sentiment_dir.glob(f"{ticker}_*_sentiment.json"))
    sentiment_dates = {f.stem.split('_')[1] for f in sentiment_files if len(f.stem.split('_')) >= 2}
    
    # Combine all dates
    all_dates = raw_dates | processed_dates | section_dates | sentiment_dates
    
    status_by_date = {}
    for date in sorted(all_dates, reverse=True):
        year = date[:4] if len(date) >= 4 else "Unknown"
        status_by_date[date] = FilingStatus(
            filing_date=date,
            year=year,
            downloaded=date in raw_dates,
            parsed=date in processed_dates,
            sections_extracted=date in section_dates,
            sentiment_analyzed=date in sentiment_dates
        )
    
    return status_by_date


def get_available_years(ticker: str) -> List[str]:
    """Get list of years with sentiment data for a ticker."""
    sentiment_dir = Path("data/sentiment")
    sentiment_dir.mkdir(parents=True, exist_ok=True)
    
    sentiment_files = list(sentiment_dir.glob(f"{ticker.upper()}_*_sentiment.json"))
    
    years = []
    for f in sentiment_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            date = parts[1]
            year = date[:4] if len(date) >= 4 else None
            if year and year not in years:
                years.append(year)
    
    return sorted(years, reverse=True)


def get_sentiment_for_date(ticker: str, filing_date: str) -> Optional[Dict]:
    """Load sentiment data for a specific filing date."""
    sentiment_file = Path("data/sentiment") / f"{ticker.upper()}_{filing_date}_sentiment.json"
    
    if sentiment_file.exists():
        with open(sentiment_file, 'r') as f:
            return json.load(f)
    return None


def get_all_sentiment_data(ticker: str) -> List[Tuple[str, Dict]]:
    """Get all sentiment data for a ticker, sorted by date."""
    sentiment_dir = Path("data/sentiment")
    sentiment_dir.mkdir(parents=True, exist_ok=True)
    
    sentiment_files = list(sentiment_dir.glob(f"{ticker.upper()}_*_sentiment.json"))
    
    results = []
    for f in sorted(sentiment_files):
        parts = f.stem.split('_')
        if len(parts) >= 2:
            filing_date = parts[1]
            with open(f, 'r') as file:
                data = json.load(file)
                data['filing_date'] = filing_date
                results.append((filing_date, data))
    
    return sorted(results, key=lambda x: x[0])


class AutomatedPipeline:
    """
    Automated pipeline for processing 10-K filings.
    
    Usage:
        pipeline = AutomatedPipeline()
        result = pipeline.process_company("AAPL", years=3)
    """
    
    def __init__(self, email: str = "student@example.com"):
        self.email = email
        self.progress_callback: Optional[Callable] = None
        self._downloader = None
        
    @property
    def downloader(self):
        if self._downloader is None:
            from src.data.downloader import SECDownloader
            self._downloader = SECDownloader(email=self.email)
        return self._downloader
        
    def set_progress_callback(self, callback: Callable[[str, int], None]):
        """Set callback for progress updates: callback(message, percent)"""
        self.progress_callback = callback
    
    def _update_progress(self, message: str, percent: int):
        if self.progress_callback:
            self.progress_callback(message, percent)
        logger.info(f"[{percent}%] {message}")
    
    def download_filings(self, ticker: str, years: int = 5) -> PipelineResult:
        """Download multiple years of 10-K filings."""
        ticker = ticker.upper()
        self._update_progress(f"Starting download for {ticker}...", 0)
        
        # Get CIK
        cik = self.downloader.get_cik(ticker)
        if not cik:
            return PipelineResult(
                success=False,
                message=f"Ticker {ticker} not found in SEC database",
                errors=[f"CIK lookup failed for {ticker}"]
            )
        
        self._update_progress(f"Found CIK: {cik}", 5)
        
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = self.downloader.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            filings_data = data['filings']['recent']
            
            available_10ks = []
            for i, form in enumerate(filings_data['form']):
                if form == '10-K' and len(available_10ks) < years:
                    filing_date = filings_data['filingDate'][i]
                    accession = filings_data['accessionNumber'][i].replace('-', '')
                    primary_doc = filings_data['primaryDocument'][i]
                    
                    cik_trimmed = cik.lstrip('0')
                    doc_url = (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{cik_trimmed}/{accession}/{primary_doc}"
                    )
                    
                    available_10ks.append({
                        'url': doc_url,
                        'filing_date': filing_date,
                        'accession': accession,
                        'document': primary_doc
                    })
            
            if not available_10ks:
                return PipelineResult(
                    success=False,
                    message=f"No 10-K filings found for {ticker}",
                    errors=["No 10-K filings in SEC database"]
                )
            
            self._update_progress(f"Found {len(available_10ks)} filings", 10)
            
        except Exception as e:
            return PipelineResult(
                success=False,
                message=f"Failed to fetch filing list: {e}",
                errors=[str(e)]
            )
        
        # Download each filing
        downloaded = []
        errors = []
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, filing_info in enumerate(available_10ks):
            filing_date = filing_info['filing_date']
            progress = 10 + int((idx + 1) / len(available_10ks) * 30)
            
            existing = list(raw_dir.glob(f"{ticker}_10K_{filing_date}.*"))
            if existing and any(f.suffix in ['.html', '.txt'] for f in existing):
                self._update_progress(f"Skipped {filing_date} (exists)", progress)
                downloaded.append({'date': filing_date, 'path': str(existing[0]), 'skipped': True})
                continue
            
            self._update_progress(f"Downloading {filing_date}...", progress)
            
            try:
                time.sleep(0.15)
                response = self.downloader.session.get(filing_info['url'], timeout=30)
                response.raise_for_status()
                
                ext = 'html' if 'html' in filing_info['document'].lower() else 'txt'
                filename = f"{ticker}_10K_{filing_date}.{ext}"
                filepath = raw_dir / filename
                filepath.write_bytes(response.content)
                
                metadata = {
                    'ticker': ticker,
                    'filing_date': filing_date,
                    'download_date': datetime.now().isoformat(),
                    'url': filing_info['url'],
                    'file_size_kb': len(response.content) / 1024
                }
                metadata_path = filepath.with_suffix('.json')
                metadata_path.write_text(json.dumps(metadata, indent=2))
                
                downloaded.append({'date': filing_date, 'path': str(filepath), 'skipped': False})
                
            except Exception as e:
                errors.append(f"Failed to download {filing_date}: {e}")
        
        return PipelineResult(
            success=len(downloaded) > 0,
            message=f"Downloaded {len(downloaded)} filings ({len(errors)} errors)",
            data={'downloaded': downloaded, 'total': len(available_10ks)},
            errors=errors
        )
    
    def parse_filing(self, ticker: str, filing_date: str) -> PipelineResult:
        """Parse a downloaded filing with Docling."""
        ticker = ticker.upper()
        self._update_progress(f"Parsing {ticker} {filing_date}...", 0)
        
        raw_dir = Path("data/raw")
        raw_files = list(raw_dir.glob(f"{ticker}_10K_{filing_date}.*"))
        raw_files = [f for f in raw_files if f.suffix in ['.html', '.txt', '.htm']]
        
        if not raw_files:
            return PipelineResult(
                success=False,
                message=f"Raw file not found: {ticker}_10K_{filing_date}",
                errors=["File not found"]
            )
        
        raw_file = raw_files[0]
        
        try:
            from docling.document_converter import DocumentConverter
            
            self._update_progress("Running Docling parser...", 30)
            
            converter = DocumentConverter()
            result = converter.convert(str(raw_file))
            
            output_dir = Path("data/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{ticker}_{filing_date}.md"
            markdown_content = result.document.export_to_markdown()
            output_file.write_text(markdown_content, encoding='utf-8')
            
            self._update_progress("Parsing complete", 100)
            
            return PipelineResult(
                success=True,
                message=f"Parsed successfully → {output_file.name}",
                data={'output_path': str(output_file), 'word_count': len(markdown_content.split())}
            )
            
        except Exception as e:
            return PipelineResult(success=False, message=f"Parse failed: {e}", errors=[str(e)])
    
    def extract_sections(self, ticker: str, filing_date: str) -> PipelineResult:
        """Extract sections from parsed filing."""
        ticker = ticker.upper()
        self._update_progress(f"Extracting sections...", 0)
        
        sections_dir = Path("data/sections")
        sections_dir.mkdir(parents=True, exist_ok=True)
        
        existing = list(sections_dir.glob(f"{ticker}_{filing_date}_item_*.txt"))
        if existing:
            return PipelineResult(
                success=True,
                message=f"Sections already extracted: {len(existing)} files",
                data={'sections': [f.name for f in existing], 'skipped': True}
            )
        
        parsed_file = Path("data/processed") / f"{ticker}_{filing_date}.md"
        if not parsed_file.exists():
            return PipelineResult(
                success=False,
                message=f"Parsed file not found: {parsed_file}",
                errors=["Must parse filing first"]
            )
        
        try:
            from src.data.splitter import SectionSplitter
            splitter = SectionSplitter()
            result = splitter.process_file(parsed_file)
            
            if result['success']:
                return PipelineResult(
                    success=True,
                    message=f"Extracted {len(result['stats']['sections_found'])} sections",
                    data={'sections': result['stats']['sections_found']}
                )
            else:
                return PipelineResult(success=True, message="No sections found", data={'sections': []})
                
        except Exception as e:
            return PipelineResult(success=True, message=f"Section extraction note: {e}", errors=[str(e)])
    
    def analyze_sentiment(self, ticker: str, filing_date: str) -> PipelineResult:
        """Run sentiment analysis on sections."""
        ticker = ticker.upper()
        self._update_progress("Analyzing sentiment...", 0)
        
        sentiment_dir = Path("data/sentiment")
        sentiment_dir.mkdir(parents=True, exist_ok=True)
        
        sentiment_file = sentiment_dir / f"{ticker}_{filing_date}_sentiment.json"
        if sentiment_file.exists():
            return PipelineResult(success=True, message="Sentiment already analyzed", data={'skipped': True})
        
        sections_dir = Path("data/sections")
        section_files = list(sections_dir.glob(f"{ticker}_{filing_date}_item_*.txt"))
        
        if not section_files:
            return PipelineResult(
                success=False,
                message="No section files found",
                errors=["Extract sections first"]
            )
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            self._update_progress("Loading FinBERT model...", 10)
            
            model_name = "ProsusAI/finbert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
            
            results = {'sections': {}, 'overall': {}}
            all_scores = []
            
            for i, section_file in enumerate(section_files):
                section_name = section_file.stem.split('_')[-1]
                progress = 10 + int((i + 1) / len(section_files) * 80)
                self._update_progress(f"Analyzing {section_name}...", progress)
                
                text = section_file.read_text(encoding='utf-8')
                words = text.split()
                
                # Chunk for FinBERT
                chunks = [' '.join(words[j:j+450]) for j in range(0, len(words), 450)]
                
                chunk_scores = []
                for chunk in chunks[:10]:
                    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    neg, neu, pos = predictions[0].tolist()
                    chunk_scores.append(pos - neg)
                
                avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0
                
                if avg_score >= 0.5: signal = 'STRONG_BUY'
                elif avg_score >= 0.2: signal = 'BUY'
                elif avg_score >= -0.2: signal = 'HOLD'
                elif avg_score >= -0.5: signal = 'SELL'
                else: signal = 'STRONG_SELL'
                
                results['sections'][section_name] = {
                    'scores': {'compound': avg_score},
                    'signal': signal,
                    'word_count': len(words)
                }
                all_scores.append(avg_score)
            
            overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
            
            if overall_score >= 0.5: overall_signal = 'STRONG_BUY'
            elif overall_score >= 0.2: overall_signal = 'BUY'
            elif overall_score >= -0.2: overall_signal = 'HOLD'
            elif overall_score >= -0.5: overall_signal = 'SELL'
            else: overall_signal = 'STRONG_SELL'
            
            results['overall'] = {'compound': overall_score, 'signal': overall_signal}
            
            sentiment_file.write_text(json.dumps(results, indent=2))
            
            return PipelineResult(
                success=True,
                message=f"Sentiment: {overall_signal} ({overall_score:+.3f})",
                data={'signal': overall_signal, 'score': overall_score}
            )
            
        except Exception as e:
            return PipelineResult(success=False, message=f"Sentiment analysis failed: {e}", errors=[str(e)])
    
    def process_filing(self, ticker: str, filing_date: str) -> PipelineResult:
        """Run complete pipeline for a single filing."""
        ticker = ticker.upper()
        steps_completed = []
        errors = []
        
        # Step 1: Parse
        self._update_progress(f"Step 1/3: Parsing {filing_date}...", 0)
        parse_result = self.parse_filing(ticker, filing_date)
        if parse_result.success or Path(f"data/processed/{ticker}_{filing_date}.md").exists():
            steps_completed.append('parse')
        else:
            return PipelineResult(success=False, message="Pipeline failed at parsing", errors=parse_result.errors)
        
        # Step 2: Sections
        self._update_progress(f"Step 2/3: Extracting sections...", 35)
        sections_result = self.extract_sections(ticker, filing_date)
        if sections_result.success:
            steps_completed.append('sections')
        
        # Step 3: Sentiment
        self._update_progress(f"Step 3/3: Analyzing sentiment...", 70)
        sentiment_result = self.analyze_sentiment(ticker, filing_date)
        if sentiment_result.success:
            steps_completed.append('sentiment')
        else:
            errors.extend(sentiment_result.errors)
        
        self._update_progress("Pipeline complete", 100)
        
        return PipelineResult(
            success='sentiment' in steps_completed,
            message=f"Completed: {', '.join(steps_completed)}",
            data={'steps_completed': steps_completed, 'sentiment': sentiment_result.data if sentiment_result.success else None},
            errors=errors
        )
    
    def process_company(self, ticker: str, years: int = 5, auto_process: bool = True) -> PipelineResult:
        """Download and process multiple years of filings."""
        ticker = ticker.upper()
        
        download_result = self.download_filings(ticker, years)
        if not download_result.success:
            return download_result
        
        if not auto_process:
            return download_result
        
        downloaded = download_result.data.get('downloaded', [])
        processed = []
        failed = []
        
        for i, filing in enumerate(downloaded):
            filing_date = filing['date']
            self._update_progress(f"Processing {filing_date} ({i+1}/{len(downloaded)})", 40 + int((i+1)/len(downloaded)*60))
            
            result = self.process_filing(ticker, filing_date)
            if result.success:
                processed.append(filing_date)
            else:
                failed.append(filing_date)
        
        return PipelineResult(
            success=len(processed) > 0,
            message=f"Processed {len(processed)}/{len(downloaded)} filings",
            data={'processed': processed, 'failed': failed, 'total': len(downloaded)},
            errors=download_result.errors + [f"Failed: {d}" for d in failed]
        )
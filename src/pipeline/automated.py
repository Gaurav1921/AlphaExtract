"""
Automated Pipeline
-------------------
Orchestrates the full 10-K processing pipeline:
download -> parse -> split sections -> sentiment analysis.

Uses the actual module classes instead of duplicating logic.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.config.settings import Settings

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

    @property
    def is_complete(self) -> bool:
        return self.sentiment_analyzed

    def to_dict(self) -> Dict:
        return {
            "filing_date": self.filing_date,
            "year": self.year,
            "downloaded": self.downloaded,
            "parsed": self.parsed,
            "sections": self.sections_extracted,
            "sentiment": self.sentiment_analyzed,
            "complete": self.is_complete,
        }


def get_filing_status(ticker: str) -> Dict[str, FilingStatus]:
    """Get complete status of all filings for a ticker."""
    ticker = ticker.upper()

    raw_dir = Settings.RAW_DIR
    processed_dir = Settings.PROCESSED_DIR
    sections_dir = Settings.SECTIONS_DIR
    sentiment_dir = Settings.SENTIMENT_DIR

    def _extract_dates(files, date_index=2):
        dates = set()
        for f in files:
            parts = f.stem.split("_")
            if len(parts) > date_index:
                dates.add(parts[date_index])
        return dates

    raw_files = list(raw_dir.glob(f"{ticker}_10K_*.*"))
    raw_files = [f for f in raw_files if f.suffix in (".html", ".txt", ".htm")]
    raw_dates = _extract_dates(raw_files, date_index=2)

    processed_dates = {f.stem.split("_")[1] for f in processed_dir.glob(f"{ticker}_*.md") if len(f.stem.split("_")) >= 2}
    section_dates = {f.stem.split("_")[1] for f in sections_dir.glob(f"{ticker}_*_item_*.txt") if len(f.stem.split("_")) >= 2}
    sentiment_dates = {f.stem.split("_")[1] for f in sentiment_dir.glob(f"{ticker}_*_sentiment.json") if len(f.stem.split("_")) >= 2}

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
            sentiment_analyzed=date in sentiment_dates,
        )

    return status_by_date


def get_available_years(ticker: str) -> List[str]:
    """Get list of years with sentiment data for a ticker."""
    sentiment_dir = Settings.SENTIMENT_DIR
    files = list(sentiment_dir.glob(f"{ticker.upper()}_*_sentiment.json"))

    years = set()
    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            year = parts[1][:4]
            if len(year) == 4:
                years.add(year)

    return sorted(years, reverse=True)


def get_sentiment_for_date(ticker: str, filing_date: str) -> Optional[Dict]:
    """Load sentiment data for a specific filing date."""
    sentiment_file = Settings.SENTIMENT_DIR / f"{ticker.upper()}_{filing_date}_sentiment.json"
    if sentiment_file.exists():
        try:
            return json.loads(sentiment_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load sentiment for {ticker} {filing_date}: {e}")
    return None


def get_all_sentiment_data(ticker: str) -> List[Tuple[str, Dict]]:
    """Get all sentiment data for a ticker, sorted by date."""
    sentiment_dir = Settings.SENTIMENT_DIR
    files = sorted(sentiment_dir.glob(f"{ticker.upper()}_*_sentiment.json"))

    results = []
    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            filing_date = parts[1]
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                data["filing_date"] = filing_date
                results.append((filing_date, data))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Skipping {f.name}: {e}")

    return sorted(results, key=lambda x: x[0])


class AutomatedPipeline:
    """
    Orchestrates download -> parse -> split -> sentiment for 10-K filings.

    Uses the actual SentimentAnalyzer class instead of duplicating FinBERT logic.
    """

    def __init__(self, email: str = None):
        self.email = email or Settings.SEC_USER_EMAIL
        self.progress_callback: Optional[Callable] = None
        self._downloader = None
        self._analyzer = None

    @property
    def downloader(self):
        if self._downloader is None:
            from src.data.downloader import SECDownloader
            self._downloader = SECDownloader(email=self.email)
        return self._downloader

    @property
    def analyzer(self):
        if self._analyzer is None:
            from src.models.sentiment import SentimentAnalyzer
            self._analyzer = SentimentAnalyzer()
        return self._analyzer

    def set_progress_callback(self, callback: Callable[[str, int], None]):
        self.progress_callback = callback

    def _update_progress(self, message: str, percent: int):
        if self.progress_callback:
            self.progress_callback(message, percent)
        logger.info(f"[{percent}%] {message}")

    def download_filings(self, ticker: str, years: int = 5) -> PipelineResult:
        """Download multiple years of 10-K filings."""
        ticker = ticker.upper()
        self._update_progress(f"Starting download for {ticker}...", 0)

        cik = self.downloader.get_cik(ticker)
        if not cik:
            return PipelineResult(success=False, message=f"Ticker {ticker} not found", errors=["CIK lookup failed"])

        self._update_progress(f"Found CIK: {cik}", 5)

        filings = self.downloader.get_10k_filings(cik, limit=years)
        if not filings:
            return PipelineResult(success=False, message=f"No 10-K filings found for {ticker}", errors=["No filings"])

        self._update_progress(f"Found {len(filings)} filings", 10)

        downloaded = []
        errors = []
        for idx, filing_info in enumerate(filings):
            filing_date = filing_info["filing_date"]
            progress = 10 + int((idx + 1) / len(filings) * 30)

            existing = list(Settings.RAW_DIR.glob(f"{ticker}_10K_{filing_date}.*"))
            if any(f.suffix in (".html", ".txt") for f in existing):
                self._update_progress(f"Skipped {filing_date} (exists)", progress)
                downloaded.append({"date": filing_date, "path": str(existing[0]), "skipped": True})
                continue

            self._update_progress(f"Downloading {filing_date}...", progress)
            filepath = self.downloader.download_filing(ticker, filing_info)
            if filepath:
                downloaded.append({"date": filing_date, "path": str(filepath)})
            else:
                errors.append(f"Failed to download {filing_date}")

        return PipelineResult(
            success=len(downloaded) > 0,
            message=f"Downloaded {len(downloaded)} filings ({len(errors)} errors)",
            data={"downloaded": downloaded, "total": len(filings)},
            errors=errors,
        )

    def parse_filing(self, ticker: str, filing_date: str) -> PipelineResult:
        """Parse a downloaded filing with Docling."""
        ticker = ticker.upper()
        self._update_progress(f"Parsing {ticker} {filing_date}...", 0)

        raw_files = list(Settings.RAW_DIR.glob(f"{ticker}_10K_{filing_date}.*"))
        raw_files = [f for f in raw_files if f.suffix in (".html", ".txt", ".htm")]
        if not raw_files:
            return PipelineResult(success=False, message=f"Raw file not found for {ticker} {filing_date}")

        try:
            from src.data.parser import TenKParser
            parser = TenKParser()
            self._update_progress("Running Docling parser...", 30)
            result = parser.process_filing(raw_files[0])

            if result.get("success"):
                self._update_progress("Parsing complete", 100)
                return PipelineResult(success=True, message=f"Parsed {filing_date}", data=result)
            else:
                return PipelineResult(success=False, message=f"Parse failed: {result.get('error')}")
        except Exception as e:
            return PipelineResult(success=False, message=f"Parse failed: {e}", errors=[str(e)])

    def extract_sections(self, ticker: str, filing_date: str) -> PipelineResult:
        """Extract sections from parsed filing."""
        ticker = ticker.upper()
        self._update_progress("Extracting sections...", 0)

        existing = list(Settings.SECTIONS_DIR.glob(f"{ticker}_{filing_date}_item_*.txt"))
        if existing:
            return PipelineResult(success=True, message=f"Sections already extracted ({len(existing)} files)", data={"skipped": True})

        parsed_file = Settings.PROCESSED_DIR / f"{ticker}_{filing_date}.md"
        if not parsed_file.exists():
            return PipelineResult(success=False, message="Parsed file not found — parse first")

        try:
            from src.data.splitter import SectionSplitter
            splitter = SectionSplitter()
            result = splitter.process_file(parsed_file)
            if result["success"]:
                return PipelineResult(success=True, message=f"Extracted {len(result['stats']['sections_found'])} sections")
            return PipelineResult(success=True, message="No sections found")
        except Exception as e:
            return PipelineResult(success=False, message=f"Section extraction failed: {e}", errors=[str(e)])

    def analyze_sentiment(self, ticker: str, filing_date: str) -> PipelineResult:
        """Run sentiment analysis using SentimentAnalyzer (not duplicated logic)."""
        ticker = ticker.upper()
        self._update_progress("Analyzing sentiment...", 0)

        sentiment_file = Settings.SENTIMENT_DIR / f"{ticker}_{filing_date}_sentiment.json"
        if sentiment_file.exists():
            return PipelineResult(success=True, message="Sentiment already analyzed", data={"skipped": True})

        section_files = list(Settings.SECTIONS_DIR.glob(f"{ticker}_{filing_date}_item_*.txt"))
        if not section_files:
            return PipelineResult(success=False, message="No section files found — extract sections first")

        try:
            self._update_progress("Loading FinBERT model...", 10)
            result = self.analyzer.analyze_and_save(ticker, filing_date)
            if result:
                signal = result["overall"]["signal"]
                score = result["overall"]["compound"]
                return PipelineResult(
                    success=True,
                    message=f"Sentiment: {signal} ({score:+.3f})",
                    data={"signal": signal, "score": score},
                )
            return PipelineResult(success=False, message="Sentiment analysis returned no results")
        except Exception as e:
            return PipelineResult(success=False, message=f"Sentiment analysis failed: {e}", errors=[str(e)])

    def score_ensemble(self, ticker: str, filing_date: str) -> PipelineResult:
        """Run ensemble scoring (FinBERT + Keywords + LLM)."""
        ticker = ticker.upper()
        self._update_progress("Ensemble scoring...", 0)

        ensemble_file = Settings.SENTIMENT_DIR / f"{ticker}_{filing_date}_ensemble.json"
        if ensemble_file.exists():
            return PipelineResult(success=True, message="Ensemble already scored", data={"skipped": True})

        sentiment_file = Settings.SENTIMENT_DIR / f"{ticker}_{filing_date}_sentiment.json"
        if not sentiment_file.exists():
            return PipelineResult(success=False, message="Sentiment data required before ensemble scoring")

        try:
            import json
            from src.models.ensemble import EnsembleScorer, load_section_texts, load_historical_texts

            sentiment_data = json.loads(sentiment_file.read_text(encoding="utf-8"))
            section_texts = load_section_texts(ticker, filing_date)
            historical_texts = load_historical_texts(ticker, exclude_date=filing_date)

            scorer = EnsembleScorer()
            result = scorer.score_filing(
                ticker=ticker,
                filing_date=filing_date,
                sentiment_data=sentiment_data,
                section_texts=section_texts,
                historical_texts=historical_texts,
            )
            scorer.save_result(result)

            signal = result["ensemble"]["signal"]
            score = result["ensemble"]["score"]
            return PipelineResult(
                success=True,
                message=f"Ensemble: {signal} ({score:+.3f})",
                data={"signal": signal, "score": score},
            )
        except Exception as e:
            logger.warning(f"Ensemble scoring failed for {ticker} {filing_date}: {e}")
            return PipelineResult(success=False, message=f"Ensemble scoring failed: {e}", errors=[str(e)])

    def process_filing(self, ticker: str, filing_date: str) -> PipelineResult:
        """Run complete pipeline for a single filing."""
        ticker = ticker.upper()
        steps_completed = []
        errors = []

        # Step 1: Parse
        self._update_progress(f"Step 1/4: Parsing {filing_date}...", 0)
        parse_result = self.parse_filing(ticker, filing_date)
        if parse_result.success or (Settings.PROCESSED_DIR / f"{ticker}_{filing_date}.md").exists():
            steps_completed.append("parse")
        else:
            return PipelineResult(success=False, message="Pipeline failed at parsing", errors=parse_result.errors)

        # Step 2: Sections
        self._update_progress("Step 2/4: Extracting sections...", 25)
        sections_result = self.extract_sections(ticker, filing_date)
        if sections_result.success:
            steps_completed.append("sections")

        # Step 3: Sentiment
        self._update_progress("Step 3/4: Analyzing sentiment...", 50)
        sentiment_result = self.analyze_sentiment(ticker, filing_date)
        if sentiment_result.success:
            steps_completed.append("sentiment")
        else:
            errors.extend(sentiment_result.errors)

        # Step 4: Ensemble scoring
        self._update_progress("Step 4/4: Ensemble scoring...", 75)
        ensemble_result = self.score_ensemble(ticker, filing_date)
        if ensemble_result.success:
            steps_completed.append("ensemble")
        else:
            # Ensemble failure is non-fatal — FinBERT result is still usable
            logger.info(f"Ensemble scoring skipped for {ticker} {filing_date}: {ensemble_result.message}")

        self._update_progress("Pipeline complete", 100)

        return PipelineResult(
            success="sentiment" in steps_completed,
            message=f"Completed: {', '.join(steps_completed)}",
            data={
                "steps_completed": steps_completed,
                "sentiment": sentiment_result.data if sentiment_result.success else None,
                "ensemble": ensemble_result.data if ensemble_result.success else None,
            },
            errors=errors,
        )

    def process_company(self, ticker: str, years: int = 5, auto_process: bool = True) -> PipelineResult:
        """Download and process multiple years of filings."""
        ticker = ticker.upper()

        download_result = self.download_filings(ticker, years)
        if not download_result.success:
            return download_result

        if not auto_process:
            return download_result

        downloaded = download_result.data.get("downloaded", [])
        processed = []
        failed = []

        for i, filing in enumerate(downloaded):
            filing_date = filing["date"]
            self._update_progress(f"Processing {filing_date} ({i+1}/{len(downloaded)})", 40 + int((i+1)/len(downloaded)*60))

            result = self.process_filing(ticker, filing_date)
            if result.success:
                processed.append(filing_date)
            else:
                failed.append(filing_date)

        return PipelineResult(
            success=len(processed) > 0,
            message=f"Processed {len(processed)}/{len(downloaded)} filings",
            data={"processed": processed, "failed": failed, "total": len(downloaded)},
            errors=download_result.errors + [f"Failed: {d}" for d in failed],
        )

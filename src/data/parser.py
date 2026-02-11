"""
10-K Document Parser with Docling
----------------------------------
Extracts structured content from SEC 10-K filings using Docling.
"""

from pathlib import Path
import json
import logging
from typing import Dict, List
from datetime import datetime
import pandas as pd

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class TenKParser:
    """Parses 10-K filings and extracts structured content via Docling."""

    def __init__(self):
        from docling.document_converter import DocumentConverter
        self.converter = DocumentConverter()
        self.processed_dir = Settings.PROCESSED_DIR

    def parse_document(self, filepath: Path) -> Dict:
        """Parse a 10-K HTML/TXT file using Docling."""
        logger.info(f"Parsing {filepath.name}")
        start_time = datetime.now()

        try:
            result = self.converter.convert(str(filepath))
            parse_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Parsed {filepath.name} in {parse_time:.2f}s")

            return {
                "success": True,
                "document": result.document,
                "parse_time": parse_time,
                "filepath": str(filepath),
            }
        except Exception as e:
            logger.error(f"Failed to parse {filepath.name}: {e}")
            return {"success": False, "error": str(e), "filepath": str(filepath)}

    def extract_sections(self, doc) -> Dict[str, str]:
        """Extract key sections from parsed document."""
        full_markdown = doc.export_to_markdown()
        sections = {"full_document": full_markdown}
        sections["metadata"] = {
            "total_chars": len(full_markdown),
            "total_words": len(full_markdown.split()),
            "extraction_date": datetime.now().isoformat(),
        }
        return sections

    def extract_tables(self, doc) -> List[Dict]:
        """Extract all tables from document as DataFrames."""
        tables = []
        if not hasattr(doc, "tables") or not doc.tables:
            return tables

        for i, table in enumerate(doc.tables):
            try:
                df = table.export_to_dataframe()
                tables.append({"index": i, "dataframe": df, "rows": len(df), "cols": len(df.columns)})
                logger.debug(f"Table {i}: {len(df)} rows x {len(df.columns)} cols")
            except Exception as e:
                logger.warning(f"Could not convert table {i}: {e}")

        return tables

    def _parse_filing_metadata(self, filepath: Path) -> Dict[str, str]:
        """Safely extract ticker and filing_date from filename."""
        parts = filepath.stem.split("_")
        ticker = parts[0] if parts else "UNKNOWN"
        # Filing date is after the last known prefix (e.g. TICKER_10K_YYYY-MM-DD or TICKER_YYYY-MM-DD)
        filing_date = "unknown"
        for part in parts[1:]:
            if len(part) >= 8 and "-" in part:
                filing_date = part
                break
        if filing_date == "unknown" and len(parts) >= 2:
            filing_date = parts[-1]
        return {"ticker": ticker, "filing_date": filing_date}

    def process_filing(self, filepath: Path, save_output: bool = True) -> Dict:
        """Complete processing pipeline for a single 10-K filing."""
        parse_result = self.parse_document(filepath)
        if not parse_result["success"]:
            return parse_result

        doc = parse_result["document"]
        sections = self.extract_sections(doc)
        tables = self.extract_tables(doc)

        meta = self._parse_filing_metadata(filepath)
        result = {
            "ticker": meta["ticker"],
            "filing_date": meta["filing_date"],
            "parse_time": parse_result["parse_time"],
            "sections": sections,
            "tables": {
                "count": len(tables),
                "summary": [{"index": t["index"], "rows": t["rows"], "cols": t["cols"]} for t in tables],
            },
            "metadata": sections["metadata"],
            "success": True,
        }

        if save_output:
            self._save_results(result, tables)

        logger.info(
            f"Processed {filepath.name}: {len(sections)} sections, "
            f"{len(tables)} tables, {sections['metadata']['total_words']:,} words"
        )
        return result

    def _save_results(self, result: Dict, tables: List[Dict]):
        """Save extracted content to files."""
        ticker = result["ticker"]
        filing_date = result["filing_date"]

        md_path = self.processed_dir / f"{ticker}_{filing_date}.md"
        md_path.write_text(result["sections"]["full_document"], encoding="utf-8")

        json_result = {k: v for k, v in result.items() if k != "sections"}
        json_result["sections"] = {k: v for k, v in result["sections"].items() if k != "full_document"}

        json_path = self.processed_dir / f"{ticker}_{filing_date}_metadata.json"
        json_path.write_text(json.dumps(json_result, indent=2, default=str), encoding="utf-8")

        for table_info in tables:
            table_path = self.processed_dir / f"{ticker}_{filing_date}_table_{table_info['index']}.csv"
            table_info["dataframe"].to_csv(table_path, index=False)

    def batch_process(self, filepaths: List[Path]) -> Dict:
        """Process multiple 10-K filings."""
        results = {"successful": [], "failed": [], "total": len(filepaths)}

        for filepath in filepaths:
            try:
                result = self.process_filing(filepath)
                if result.get("success"):
                    results["successful"].append(result["ticker"])
                else:
                    results["failed"].append(filepath.stem.split("_")[0])
            except Exception as e:
                logger.error(f"Failed to process {filepath.name}: {e}")
                results["failed"].append(filepath.stem.split("_")[0])

        logger.info(f"Batch complete: {len(results['successful'])} ok, {len(results['failed'])} failed")
        return results


def parse_all_filings() -> Dict:
    """Parse all unprocessed filings in raw directory."""
    parser = TenKParser()
    raw_files = sorted(Settings.RAW_DIR.glob("*_10K_*.html")) + sorted(Settings.RAW_DIR.glob("*_10K_*.txt"))

    # Filter out already-processed
    processed = {f.stem.replace("_metadata", "") for f in Settings.PROCESSED_DIR.glob("*.md")}
    to_process = []
    for f in raw_files:
        meta = parser._parse_filing_metadata(f)
        key = f"{meta['ticker']}_{meta['filing_date']}"
        if key not in processed:
            to_process.append(f)

    if not to_process:
        logger.info("All filings already parsed")
        return {"success": 0, "total": 0}

    result = parser.batch_process(to_process)
    return {"success": len(result["successful"]), "total": result["total"]}

"""
Section Splitter for 10-K Documents
------------------------------------
Extracts specific sections (Item 1A, 7, 8) from parsed markdown files.
"""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import logging

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class SectionSplitter:
    """Extracts key sections from 10-K markdown files."""

    SECTION_PATTERNS = {
        "item_1a": [
            r"Item\s+1A\.?\s*[\-—:]?\s*Risk\s+Factors",
            r"ITEM\s+1A\.?\s*[\-—:]?\s*RISK\s+FACTORS",
        ],
        "item_7": [
            r"Item\s+7\.?\s*[\-—:]?\s*Management.?s\s+Discussion\s+and\s+Analysis",
            r"ITEM\s+7\.?\s*[\-—:]?\s*MANAGEMENT.?S\s+DISCUSSION\s+AND\s+ANALYSIS",
        ],
        "item_8": [
            r"Item\s+8\.?\s*[\-—:]?\s*Financial\s+Statements",
            r"ITEM\s+8\.?\s*[\-—:]?\s*FINANCIAL\s+STATEMENTS",
        ],
    }

    # Patterns to detect the *next* item boundary
    NEXT_ITEM_PATTERN = r"(?:^|\n)\s*I[Tt][Ee][Mm]\s+\d+[A-Za-z]?\."

    def __init__(self):
        self.output_dir = Settings.SECTIONS_DIR

    def find_section_boundaries(self, text: str, section_key: str) -> Optional[Tuple[int, int]]:
        """Find start and end positions of a section in the text."""
        patterns = self.SECTION_PATTERNS.get(section_key, [])
        all_matches = []
        for pattern in patterns:
            all_matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        if not all_matches:
            return None

        all_matches.sort(key=lambda m: m.start())

        # Find the actual content start (skip table-of-contents entries).
        # TOC entries are short; real sections have substantial text after the header.
        start_pos = None
        for match in all_matches:
            pos = match.start()
            after_header = text[pos + len(match.group()) : pos + 1000].strip()
            if len(after_header) > 200:
                start_pos = pos
                break

        if start_pos is None:
            # If only one match, use it regardless
            if len(all_matches) == 1:
                start_pos = all_matches[0].start()
            else:
                return None

        # Find end: the next Item header at least 1000 chars after start
        search_start = start_pos + 1000
        for match in re.finditer(self.NEXT_ITEM_PATTERN, text[search_start:]):
            candidate = search_start + match.start()
            snippet = text[candidate : candidate + 500].strip()
            if len(snippet) > 200:
                return (start_pos, candidate)

        # No next item found — cap at 50000 chars
        return (start_pos, min(len(text), start_pos + 50000))

    def extract_section(self, text: str, section_key: str) -> Optional[str]:
        """Extract a specific section from the text."""
        boundaries = self.find_section_boundaries(text, section_key)
        if boundaries is None:
            logger.debug(f"Section {section_key} not found")
            return None

        start_pos, end_pos = boundaries
        section_text = text[start_pos:end_pos].strip()
        section_text = self._clean_section_text(section_text)

        logger.info(f"Extracted {section_key}: {len(section_text.split())} words")
        return section_text

    @staticmethod
    def _clean_section_text(text: str) -> str:
        """Clean extracted section text."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\n[A-Za-z\s]+\|\s*\d{4}\s*Form\s*10-K\s*\|\s*\d+\n", "\n", text)
        text = re.sub(r"\|[\s\-\|]+\|", "", text)
        return text.strip()

    def split_document(self, filepath: Path) -> Tuple[Dict[str, str], Dict]:
        """Split a 10-K markdown file into sections."""
        logger.info(f"Splitting: {filepath.name}")
        text = filepath.read_text(encoding="utf-8")

        sections = {}
        for section_key in ["item_1a", "item_7", "item_8"]:
            section_text = self.extract_section(text, section_key)
            if section_text:
                sections[section_key] = section_text

        parts = filepath.stem.split("_")
        ticker = parts[0] if parts else "UNKNOWN"
        filing_date = parts[1] if len(parts) >= 2 else "unknown"

        stats = {
            "ticker": ticker,
            "filing_date": filing_date,
            "sections_found": list(sections.keys()),
            "section_stats": {
                key: {"chars": len(txt), "words": len(txt.split()), "lines": len(txt.splitlines())}
                for key, txt in sections.items()
            },
        }

        logger.info(f"Found {len(sections)} sections for {ticker} {filing_date}")
        return sections, stats

    def save_sections(self, sections: Dict[str, str], stats: Dict):
        """Save extracted sections to individual files."""
        ticker = stats["ticker"]
        filing_date = stats["filing_date"]

        for section_key, content in sections.items():
            section_path = self.output_dir / f"{ticker}_{filing_date}_{section_key}.txt"
            section_path.write_text(content, encoding="utf-8")
            logger.debug(f"Saved {section_path.name}")

        stats_path = self.output_dir / f"{ticker}_{filing_date}_sections.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    def process_file(self, filepath: Path) -> Dict:
        """Complete processing pipeline for a single file."""
        sections, stats = self.split_document(filepath)
        self.save_sections(sections, stats)
        return {"success": len(sections) > 0, "sections": sections, "stats": stats}

    def batch_process(self, directory: Path) -> Dict:
        """Process all markdown files in a directory."""
        md_files = sorted(directory.glob("*.md"))
        if not md_files:
            logger.warning(f"No markdown files in {directory}")
            return {"total": 0, "successful": [], "failed": []}

        results = {"total": len(md_files), "successful": [], "failed": []}

        for filepath in md_files:
            try:
                result = self.process_file(filepath)
                bucket = "successful" if result["success"] else "failed"
                results[bucket].append(result["stats"]["ticker"])
            except Exception as e:
                logger.error(f"Failed to process {filepath.name}: {e}")
                results["failed"].append(filepath.stem.split("_")[0])

        logger.info(f"Batch split: {len(results['successful'])} ok, {len(results['failed'])} failed")
        return results

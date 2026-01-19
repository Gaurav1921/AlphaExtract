"""
Section Splitter for 10-K Documents
------------------------------------
Extracts specific sections (Item 1A, 7, 8) from parsed markdown files.

Key challenges:
- Section headers vary in format
- Need to handle both table of contents links and actual content
- Must stop at the next section header
- XBRL metadata at top should be skipped
"""

import re
from pathlib import Path
from typing import Dict, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SectionSplitter:
    """
    Extracts key sections from 10-K markdown files.
    
    Targets:
    - Item 1A: Risk Factors
    - Item 7: Management's Discussion and Analysis (MD&A)
    - Item 8: Financial Statements
    """
    
    # Regex patterns for different section header formats
    SECTION_PATTERNS = {
        'item_1a': [
            r'Item\s+1A\.\s+Ris\s*k\s+Factors',         # Handle "RIS K"
            r'ITEM\s+1A\.\s+RIS\s*K\s+FACTORS',
            r'Item\s+1A\.\s+Risk\s+Factors',            # Normal
            r'ITEM\s+1A\.\s+RISK\s+FACTORS',
        ],
        'item_7': [
            r'Item\s+7\.\s+Management.?s\s+Discussion\s+and\s+Analysis',
            r'ITEM\s+7\.\s+MANAGEMENT.?S\s+DISCUSSION\s+AND\s+ANALYSIS',
        ],
        'item_8': [
            r'Item\s+8\.\s+Financial\s+State\s*ments',  # Handle "STATE MENTS"
            r'ITEM\s+8\.\s+FINANCIAL\s+STATE\s*MENTS',
            r'Item\s+8\.\s+Financial\s+Statements',     # Normal spacing
            r'ITEM\s+8\.\s+FINANCIAL\s+STATEMENTS',
        ],
        'item_1b': [
            r'Item\s+1B\.\s+Unresolved\s+Staff\s+Comments',
        ],
        'item_9': [
            r'Item\s+9\.\s+Changes\s+in\s+and\s+Disagreements',
        ]
    }
    
    def __init__(self):
        """Initialize the section splitter."""
        self.output_dir = Path("data/sections")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def find_section_boundaries(self, text: str, section_key: str) -> Optional[tuple]:
        """
        Find start and end positions of a section in the text.
        
        Strategy: The first occurrence is usually Table of Contents,
        the second (or last) occurrence is the actual section content.
        
        Args:
            text: Full markdown text
            section_key: Key from SECTION_PATTERNS (e.g., 'item_1a')
            
        Returns:
            Tuple of (start_pos, end_pos) or None if not found
        """
        patterns = self.SECTION_PATTERNS.get(section_key, [])
        
        # Find all matches
        all_matches = []
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            all_matches.extend(matches)
        
        if not all_matches:
            return None
        
        # Sort by position
        all_matches.sort(key=lambda m: m.start())
        
        # Strategy: Try matches starting from the second one
        # (first is usually TOC)
        start_pos = None
        
        for match in all_matches[1:]:  # Skip first match (likely TOC)
            pos = match.start()
            
            # Verify this looks like a real section
            # Real sections have substantial text after them
            context_after = text[pos:min(len(text), pos + 1000)]
            
            # Count non-whitespace characters in next 500 chars
            content_after = context_after[len(match.group()):500]
            if len(content_after.strip()) > 200:  # Has substantial content
                start_pos = pos
                break
        
        # Fallback: if we only have one match, use it
        if start_pos is None and len(all_matches) == 1:
            start_pos = all_matches[0].start()
        
        if start_pos is None:
            return None
        
        # Find section end (next major section header that's NOT in TOC)
        # Look for "Item X." that has content after it
        search_start = start_pos + 1000
        next_item_pattern = r'(?:^|\n)I[Tt][Ee][Mm]\s+\d+[A-Z]?\.'
        
        potential_ends = []
        for match in re.finditer(next_item_pattern, text[search_start:]):
            pos = search_start + match.start()
            
            # Check if this next item has content (not just TOC)
            snippet = text[pos:pos + 500]
            if len(snippet.strip()) > 200:
                potential_ends.append(pos)
        
        if potential_ends:
            end_pos = potential_ends[0]
        else:
            # No clear next section, take a reasonable chunk
            end_pos = min(len(text), start_pos + 50000)  # Max 50k chars
        
        return (start_pos, end_pos)
    
    def extract_section(self, text: str, section_key: str) -> Optional[str]:
        """
        Extract a specific section from the text.
        
        Args:
            text: Full markdown text
            section_key: Section identifier (e.g., 'item_1a')
            
        Returns:
            Section text or None if not found
        """
        boundaries = self.find_section_boundaries(text, section_key)
        
        if boundaries is None:
            logger.warning(f"Section {section_key} not found")
            return None
        
        start_pos, end_pos = boundaries
        section_text = text[start_pos:end_pos].strip()
        
        # Clean up the text
        section_text = self._clean_section_text(section_text)
        
        logger.info(
            f"Extracted {section_key}: "
            f"{len(section_text)} chars, "
            f"{len(section_text.split())} words"
        )
        
        return section_text
    
    def _clean_section_text(self, text: str) -> str:
        """
        Clean extracted section text.
        
        Args:
            text: Raw section text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove page numbers/footers (e.g., "Apple Inc. | 2025 Form 10-K | 5")
        text = re.sub(r'\n[A-Za-z\s]+\|\s*\d{4}\s*Form\s*10-K\s*\|\s*\d+\n', '\n', text)
        
        # Remove markdown table artifacts if they snuck in
        text = re.sub(r'\|[\s\-\|]+\|', '', text)
        
        return text.strip()
    
    def split_document(self, filepath: Path) -> Dict[str, str]:
        """
        Split a 10-K markdown file into sections.
        
        Args:
            filepath: Path to markdown file
            
        Returns:
            Dict mapping section names to content
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"SPLITTING SECTIONS: {filepath.name}")
        logger.info(f"{'='*80}")
        
        # Read the markdown file
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Extract key sections
        sections = {}
        
        for section_key in ['item_1a', 'item_7', 'item_8']:
            section_text = self.extract_section(text, section_key)
            if section_text:
                sections[section_key] = section_text
        
        # Generate summary stats
        stats = {
            'ticker': filepath.stem.split('_')[0],
            'filing_date': filepath.stem.split('_')[-1],
            'sections_found': list(sections.keys()),
            'section_stats': {
                key: {
                    'chars': len(text),
                    'words': len(text.split()),
                    'lines': len(text.splitlines())
                }
                for key, text in sections.items()
            }
        }
        
        logger.info(f"\n✓ Extraction complete!")
        logger.info(f"  Sections found: {len(sections)}")
        for key, text in sections.items():
            logger.info(f"    {key}: {len(text.split())} words")
        
        return sections, stats
    
    def save_sections(self, sections: Dict[str, str], stats: Dict, filepath: Path):
        """
        Save extracted sections to individual files.
        
        Args:
            sections: Dict of section name → content
            stats: Statistics dict
            filepath: Original markdown filepath
        """
        ticker = stats['ticker']
        filing_date = stats['filing_date']
        
        # Save each section as a separate file
        for section_key, content in sections.items():
            section_path = self.output_dir / f"{ticker}_{filing_date}_{section_key}.txt"
            section_path.write_text(content, encoding='utf-8')
            logger.info(f"  Saved: {section_path.name}")
        
        # Save stats
        stats_path = self.output_dir / f"{ticker}_{filing_date}_sections.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding='utf-8')
        logger.info(f"  Saved: {stats_path.name}")
    
    def process_file(self, filepath: Path) -> Dict:
        """
        Complete processing pipeline for a single file.
        
        Args:
            filepath: Path to markdown file
            
        Returns:
            Dict with sections and stats
        """
        sections, stats = self.split_document(filepath)
        self.save_sections(sections, stats, filepath)
        
        return {
            'success': len(sections) > 0,
            'sections': sections,
            'stats': stats
        }
    
    def batch_process(self, directory: Path) -> Dict:
        """
        Process all markdown files in a directory.
        
        Args:
            directory: Path to directory with markdown files
            
        Returns:
            Summary of processing results
        """
        md_files = list(directory.glob("*.md"))
        
        if not md_files:
            logger.warning(f"No markdown files found in {directory}")
            return {'total': 0, 'successful': [], 'failed': []}
        
        results = {
            'total': len(md_files),
            'successful': [],
            'failed': []
        }
        
        for filepath in md_files:
            try:
                result = self.process_file(filepath)
                if result['success']:
                    results['successful'].append(result['stats']['ticker'])
                else:
                    results['failed'].append(result['stats']['ticker'])
            except Exception as e:
                logger.error(f"Failed to process {filepath.name}: {e}")
                results['failed'].append(filepath.stem.split('_')[0])
        
        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH PROCESSING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"  Total files: {results['total']}")
        logger.info(f"  Successful: {len(results['successful'])}")
        logger.info(f"  Failed: {len(results['failed'])}")
        
        return results


# ============================================================================
# TESTING & USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SECTION SPLITTER - Extract Items 1A, 7, 8")
    print("=" * 80)
    
    splitter = SectionSplitter()
    
    # Find processed markdown files
    processed_dir = Path("data/processed")
    md_files = list(processed_dir.glob("*.md"))
    
    if not md_files:
        print("\n❌ No markdown files found in data/processed/")
        print("   Run parse_10k.py first!")
    else:
        print(f"\nFound {len(md_files)} markdown files")
        
        # Process first file as test
        print("\n[TEST] Processing first file in detail...")
        result = splitter.process_file(md_files[0])
        
        if result['success']:
            print(f"\n✓ Success!")
            print(f"  Sections extracted: {', '.join(result['stats']['sections_found'])}")
            print(f"  Output directory: data/sections/")
            
            # Show sample from Item 7
            if 'item_7' in result['sections']:
                sample = result['sections']['item_7'][:500]
                print(f"\n📄 ITEM 7 (MD&A) PREVIEW:")
                print("-" * 80)
                print(sample)
                print("...")
        
        # Batch process all files
        if len(md_files) > 1:
            response = input(f"\nProcess all {len(md_files)} files? (y/n): ")
            if response.lower() == 'y':
                splitter.batch_process(processed_dir)
    
    print("\n" + "=" * 80)
    print("Next step: Sentiment analysis with FinBERT!")
    print("=" * 80)
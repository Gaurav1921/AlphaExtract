"""
10-K Document Parser with Docling
----------------------------------
Extracts structured content from SEC 10-K filings using Docling's
document intelligence capabilities.

Key features:
- XBRL-aware parsing
- Section extraction (Item 1A, 7, 8)
- Table extraction to DataFrames
- Clean text output
- Structured JSON export
"""

from docling.document_converter import DocumentConverter

from pathlib import Path
import json
import logging
from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TenKParser:
    """
    Parses 10-K filings and extracts structured content.
    
    Why Docling over BeautifulSoup:
    - Understands document layout (headers, tables, paragraphs)
    - Handles XBRL inline format
    - Preserves table structure
    - Better text extraction (removes boilerplate)
    """
    
    def __init__(self):
        """Initialize Docling converter with optimized settings."""
        
        # Simple converter initialization - Docling 2.x handles settings automatically
        self.converter = DocumentConverter()
        
        # Output directories
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_document(self, filepath: Path) -> Dict:
        """
        Parse a 10-K HTML file using Docling.
        
        Args:
            filepath: Path to HTML file
            
        Returns:
            Dict with parsed document and metadata
        """
        logger.info(f"Parsing {filepath.name} with Docling...")
        
        start_time = datetime.now()
        
        try:
            # Convert document
            result = self.converter.convert(str(filepath))
            
            parse_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✓ Parsed in {parse_time:.2f} seconds")
            
            return {
                'success': True,
                'document': result.document,
                'parse_time': parse_time,
                'filepath': str(filepath)
            }
            
        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            return {
                'success': False,
                'error': str(e),
                'filepath': str(filepath)
            }
    
    def extract_sections(self, doc) -> Dict[str, str]:
        """
        Extract key sections from parsed document.
        
        Args:
            doc: Docling document object
            
        Returns:
            Dict mapping section names to content
        """
        sections = {}
        
        # Export full document as markdown
        full_markdown = doc.export_to_markdown()
        
        # Section patterns to extract
        # These are the key sections for AlphaExtract
        section_patterns = {
            'business_overview': r'Item\s+1\.?\s+Business',
            'risk_factors': r'Item\s+1A\.?\s+Risk\s+Factors',
            'mda': r'Item\s+7\.?\s+Management',
            'financials': r'Item\s+8\.?\s+Financial\s+Statements',
        }
        
        # For now, return full markdown
        # We'll add section splitting in the next iteration
        sections['full_document'] = full_markdown
        
        # Extract basic stats
        sections['metadata'] = {
            'total_chars': len(full_markdown),
            'total_words': len(full_markdown.split()),
            'extraction_date': datetime.now().isoformat()
        }
        
        return sections
    
    def extract_tables(self, doc) -> List[pd.DataFrame]:
        """
        Extract all tables from document as DataFrames.
        
        Args:
            doc: Docling document object
            
        Returns:
            List of pandas DataFrames
        """
        tables = []
        
        # Docling stores tables in document.tables
        if hasattr(doc, 'tables') and doc.tables:
            for i, table in enumerate(doc.tables):
                try:
                    # Convert Docling table to DataFrame
                    # (Exact method depends on Docling version)
                    df = table.export_to_dataframe()
                    tables.append({
                        'index': i,
                        'dataframe': df,
                        'rows': len(df),
                        'cols': len(df.columns)
                    })
                    logger.info(f"  Table {i}: {len(df)} rows × {len(df.columns)} cols")
                except Exception as e:
                    logger.warning(f"  Could not convert table {i}: {e}")
        
        return tables
    
    def process_filing(self, filepath: Path, save_output: bool = True) -> Dict:
        """
        Complete processing pipeline for a 10-K filing.
        
        Args:
            filepath: Path to 10-K HTML file
            save_output: Whether to save extracted content
            
        Returns:
            Dict with all extracted content
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING: {filepath.name}")
        logger.info(f"{'='*80}")
        
        # Step 1: Parse with Docling
        parse_result = self.parse_document(filepath)
        
        if not parse_result['success']:
            return parse_result
        
        doc = parse_result['document']
        
        # Step 2: Extract sections
        logger.info("Extracting sections...")
        sections = self.extract_sections(doc)
        
        # Step 3: Extract tables
        logger.info("Extracting tables...")
        tables = self.extract_tables(doc)
        
        # Step 4: Compile results
        result = {
            'ticker': filepath.stem.split('_')[0],
            'filing_date': filepath.stem.split('_')[-1],
            'parse_time': parse_result['parse_time'],
            'sections': sections,
            'tables': {
                'count': len(tables),
                'summary': [
                    {'index': t['index'], 'rows': t['rows'], 'cols': t['cols']}
                    for t in tables
                ]
            },
            'metadata': sections['metadata']
        }
        
        # Step 5: Save outputs
        if save_output:
            self._save_results(result, tables, filepath)
        
        logger.info(f"\n✓ Processing complete!")
        logger.info(f"  Sections extracted: {len(sections)}")
        logger.info(f"  Tables found: {len(tables)}")
        logger.info(f"  Total words: {sections['metadata']['total_words']:,}")
        
        return result
    
    def _save_results(self, result: Dict, tables: List, filepath: Path):
        """Save extracted content to files."""
        
        ticker = result['ticker']
        filing_date = result['filing_date']
        
        # Save markdown
        md_path = self.processed_dir / f"{ticker}_{filing_date}.md"
        md_path.write_text(result['sections']['full_document'], encoding='utf-8')
        logger.info(f"  Saved markdown: {md_path}")
        
        # Save metadata as JSON
        json_result = {k: v for k, v in result.items() if k != 'sections'}
        json_result['sections'] = {k: v for k, v in result['sections'].items() if k != 'full_document'}
        
        json_path = self.processed_dir / f"{ticker}_{filing_date}_metadata.json"
        json_path.write_text(json.dumps(json_result, indent=2), encoding='utf-8')
        logger.info(f"  Saved metadata: {json_path}")
        
        # Save tables as CSV
        for table_info in tables:
            table_path = self.processed_dir / f"{ticker}_{filing_date}_table_{table_info['index']}.csv"
            table_info['dataframe'].to_csv(table_path, index=False)
        
        if tables:
            logger.info(f"  Saved {len(tables)} tables as CSV")
    
    def batch_process(self, filepaths: List[Path]) -> Dict:
        """
        Process multiple 10-K filings.
        
        Args:
            filepaths: List of HTML file paths
            
        Returns:
            Summary statistics
        """
        results = {
            'successful': [],
            'failed': [],
            'total': len(filepaths)
        }
        
        for filepath in filepaths:
            result = self.process_filing(filepath)
            
            if result.get('success', True):
                results['successful'].append(result['ticker'])
            else:
                results['failed'].append(result['ticker'])
        
        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH PROCESSING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"  Successful: {len(results['successful'])}")
        logger.info(f"  Failed: {len(results['failed'])}")
        
        return results


# ============================================================================
# TESTING & USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("10-K PARSER - Docling Edition")
    print("=" * 80)
    
    # Initialize parser
    parser = TenKParser()
    
    # Find downloaded 10-Ks
    raw_dir = Path("data/raw")
    html_files = sorted(raw_dir.glob("*_10K_*.html"))
    
    if not html_files:
        print("\n❌ No 10-K files found in data/raw/")
        print("   Run sec_downloader.py first!")
    else:
        print(f"\nFound {len(html_files)} 10-K files")
        
        # Process first file as test
        print("\n[TEST] Processing first file in detail...")
        result = parser.process_filing(html_files[0])
        
        if result.get('success', True):
            print(f"\n✓ Success!")
            print(f"  Output saved to: data/processed/")
            print(f"  Markdown file: {result['ticker']}_{result['filing_date']}.md")
            print(f"  Metadata JSON: {result['ticker']}_{result['filing_date']}_metadata.json")
            
            # Show preview of extracted content
            full_text = result['sections']['full_document']
            print(f"\n📄 CONTENT PREVIEW (first 500 chars):")
            print("-" * 80)
            print(full_text[:500])
            print("...")
        
        # Batch process all files
        if len(html_files) > 1:
            response = input(f"\nProcess all {len(html_files)} files? (y/n): ")
            if response.lower() == 'y':
                parser.batch_process(html_files)
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print("1. Check data/processed/ for extracted content")
    print("2. Review the markdown files - are sections clear?")
    print("3. Check table CSVs - do numbers look correct?")
    print("4. Ready for Phase 2: Sentiment Analysis!")
    print("=" * 80)
"""
10-K Structure Inspector
------------------------
Explores the structure of a downloaded 10-K to understand:
- What sections exist
- How much text is in each
- Where tables are located
- Sample content from key sections

This helps us decide what to extract and prioritize.
"""

from pathlib import Path
from bs4 import BeautifulSoup
import re
from collections import defaultdict


class TenKInspector:
    """
    Quick analysis tool for understanding 10-K structure.
    """
    
    def __init__(self, filepath: Path):
        """Load and parse the 10-K HTML."""
        self.filepath = filepath
        with open(filepath, 'r', encoding='utf-8') as f:
            self.html = f.read()
        self.soup = BeautifulSoup(self.html, 'html.parser')
    
    def find_sections(self):
        """
        Find major sections in the 10-K.
        
        Returns:
            dict: Section headers and their locations
        """
        sections = {}
        
        # Common patterns for section headers
        patterns = [
            r'Item\s+(\d+[A-Z]?)\.\s*(.+?)(?=\n|$)',  # "Item 1A. Risk Factors"
            r'ITEM\s+(\d+[A-Z]?)\.\s*(.+?)(?=\n|$)',  # Uppercase variant
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, self.html, re.IGNORECASE)
            for match in matches:
                item_num = match.group(1)
                title = match.group(2).strip()
                
                # Clean up title (remove HTML tags)
                title = re.sub(r'<[^>]+>', '', title)
                
                if item_num not in sections:  # Keep first occurrence
                    sections[item_num] = {
                        'title': title,
                        'position': match.start()
                    }
        
        return dict(sorted(sections.items(), key=lambda x: x[1]['position']))
    
    def extract_section_text(self, start_pos: int, end_pos: int) -> str:
        """
        Extract text between two positions.
        
        Args:
            start_pos: Start position in HTML
            end_pos: End position in HTML
            
        Returns:
            Clean text content
        """
        section_html = self.html[start_pos:end_pos]
        section_soup = BeautifulSoup(section_html, 'html.parser')
        
        # Remove script and style elements
        for element in section_soup(['script', 'style']):
            element.decompose()
        
        # Get text and clean it
        text = section_soup.get_text()
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines()]
        text = '\n'.join(line for line in lines if line)
        
        return text
    
    def count_tables(self):
        """Count how many tables are in the document."""
        return len(self.soup.find_all('table'))
    
    def get_section_stats(self):
        """
        Analyze each section's content.
        
        Returns:
            dict: Statistics for each section
        """
        sections = self.find_sections()
        section_items = list(sections.items())
        stats = {}
        
        for i, (item_num, info) in enumerate(section_items):
            # Determine end position (next section or end of document)
            if i < len(section_items) - 1:
                end_pos = section_items[i + 1][1]['position']
            else:
                end_pos = len(self.html)
            
            # Extract section text
            text = self.extract_section_text(info['position'], end_pos)
            
            # Calculate stats
            stats[item_num] = {
                'title': info['title'],
                'char_count': len(text),
                'word_count': len(text.split()),
                'preview': text[:300] + '...' if len(text) > 300 else text,
                'tables_estimate': text.count('Table') + text.count('TABLE')
            }
        
        return stats
    
    def print_report(self):
        """Print a formatted analysis report."""
        print("=" * 80)
        print(f"10-K STRUCTURE ANALYSIS: {self.filepath.name}")
        print("=" * 80)
        
        # Overall stats
        total_tables = self.count_tables()
        total_chars = len(self.html)
        
        print(f"\n📊 OVERALL STATISTICS")
        print(f"   Total size: {total_chars:,} characters ({total_chars / 1024 / 1024:.2f} MB)")
        print(f"   Total tables: {total_tables}")
        
        # Section breakdown
        print(f"\n📑 SECTION BREAKDOWN")
        print("-" * 80)
        
        stats = self.get_section_stats()
        
        # Sort by size (descending)
        sorted_stats = sorted(
            stats.items(), 
            key=lambda x: x[1]['word_count'], 
            reverse=True
        )
        
        for item_num, info in sorted_stats[:10]:  # Show top 10
            print(f"\nItem {item_num}: {info['title']}")
            print(f"   Words: {info['word_count']:,}")
            print(f"   Characters: {info['char_count']:,}")
            print(f"   Estimated tables: {info['tables_estimate']}")
            print(f"   Preview: {info['preview'][:150]}...")
        
        print("\n" + "=" * 80)
        
        # Key sections for analysis
        print(f"\n🎯 KEY SECTIONS FOR ALPHAEXTRACT")
        print("-" * 80)
        
        key_items = {
            '1A': 'Risk Factors (sentiment + red flags)',
            '7': 'MD&A (management outlook + sentiment)',
            '8': 'Financial Statements (tables + numbers)'
        }
        
        for item, description in key_items.items():
            if item in stats:
                info = stats[item]
                print(f"\n✓ Item {item}: {description}")
                print(f"   Size: {info['word_count']:,} words")
                print(f"   Priority: {'HIGH' if info['word_count'] > 5000 else 'MEDIUM'}")
            else:
                print(f"\n✗ Item {item}: {description}")
                print(f"   Status: NOT FOUND")
        
        print("\n" + "=" * 80)
        
        return stats


def compare_companies(filepaths: list[Path]):
    """
    Compare section sizes across multiple companies.
    
    Args:
        filepaths: List of 10-K HTML files
    """
    print("\n" + "=" * 80)
    print("CROSS-COMPANY COMPARISON")
    print("=" * 80)
    
    results = {}
    for filepath in filepaths:
        inspector = TenKInspector(filepath)
        stats = inspector.get_section_stats()
        
        # Extract ticker from filename (e.g., "TSLA_10K_2025-01-30.html")
        ticker = filepath.stem.split('_')[0]
        results[ticker] = stats
    
    # Compare key sections
    key_sections = ['1A', '7', '8']
    
    for section in key_sections:
        print(f"\n📊 Item {section} Word Counts:")
        print("-" * 40)
        
        for ticker, stats in results.items():
            if section in stats:
                word_count = stats[section]['word_count']
                print(f"   {ticker:6s}: {word_count:6,} words")
            else:
                print(f"   {ticker:6s}: NOT FOUND")
    
    print("\n" + "=" * 80)


# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    # Find all downloaded 10-Ks
    raw_dir = Path("data/raw")
    html_files = list(raw_dir.glob("*_10K_*.html"))
    
    if not html_files:
        print("❌ No 10-K files found in data/raw/")
        print("   Run sec_downloader.py first!")
    else:
        print(f"Found {len(html_files)} 10-K files\n")
        
        # Analyze first file in detail
        print("Analyzing first file in detail...")
        inspector = TenKInspector(html_files[0])
        stats = inspector.print_report()
        
        # If multiple files, compare them
        if len(html_files) > 1:
            print("\n\nComparing all files...")
            compare_companies(html_files)
        
        print("\n💡 RECOMMENDATION:")
        print("-" * 80)
        print("Based on this analysis, we should prioritize:")
        print("1. Item 7 (MD&A) - Usually largest, most narrative content")
        print("2. Item 1A (Risk Factors) - Red flag detection")
        print("3. Item 8 (Financials) - Tables for quantitative signals")
        print("\nNext: Build Docling parser to extract these sections cleanly!")
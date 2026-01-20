"""
Parse Wrapper for Pipeline
--------------------------
Simple wrapper to parse a single 10-K file.
Usage: python parse_wrapper.py <path_to_file>
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def parse_single_file(filepath: Path) -> bool:
    """Parse a single 10-K file."""
    try:
        # Import the actual parse_10k.py functionality
        # We'll use Docling directly since we know it's installed
        from docling.document_converter import DocumentConverter
        
        print(f"Parsing {filepath}...")
        
        # Convert using Docling
        converter = DocumentConverter()
        result = converter.convert(str(filepath))
        
        # Save as markdown
        ticker = filepath.stem.split('_')[0]
        filing_date = filepath.stem.split('_')[2]
        
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{ticker}_{filing_date}.md"
        output_file.write_text(result.document.export_to_markdown(), encoding='utf-8')
        
        print(f"✓ Parsed successfully → {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ Parse failed: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_wrapper.py <path_to_file>")
        sys.exit(1)
    
    filepath = Path(sys.argv[1])
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    success = parse_single_file(filepath)
    sys.exit(0 if success else 1)
"""
Section Splitter Wrapper
------------------------
Thin wrapper that calls the main SectionSplitter class.
Kept for backward compatibility with existing scripts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.splitter import SectionSplitter


if __name__ == "__main__":
    print("=" * 80)
    print("SECTION SPLITTER - Extract Items 1A, 7, 8")
    print("=" * 80)
    
    splitter = SectionSplitter()
    
    processed_dir = Path("data/processed")
    md_files = list(processed_dir.glob("*.md"))
    
    if not md_files:
        print("\n❌ No markdown files found in data/processed/")
        print("   Run the parser first!")
    else:
        print(f"\nFound {len(md_files)} markdown files")
        
        print("\n[PROCESSING] Processing all files...")
        results = splitter.batch_process(processed_dir)
        
        print(f"\n✓ Processing complete!")
        print(f"  Successful: {len(results['successful'])}")
        print(f"  Failed: {len(results['failed'])}")
    
    print("\n" + "=" * 80)
    print("Next step: Sentiment analysis with FinBERT!")
    print("=" * 80)
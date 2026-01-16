"""
Section Header Finder
---------------------
Diagnoses why sections aren't being extracted by finding all Item X headers.
"""

import re
from pathlib import Path


def find_all_item_headers(filepath: Path):
    """Find all Item X.Y headers in a markdown file."""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"\n{'='*80}")
    print(f"FILE: {filepath.name}")
    print(f"{'='*80}")
    
    # Find all "Item X" patterns
    pattern = r'.{0,50}(Item\s+\d+[A-Z]?\.[^\n]{0,100})'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # Deduplicate and filter
    seen = set()
    items = []
    
    for match in matches:
        clean = match.strip()
        if clean not in seen and 'Item' in clean:
            seen.add(clean)
            items.append(clean)
    
    # Show unique items
    print(f"\nFound {len(items)} unique Item headers:\n")
    
    for i, item in enumerate(items[:20], 1):  # Show first 20
        # Clean up for display
        item = re.sub(r'\s+', ' ', item)
        print(f"{i:2}. {item[:120]}")
    
    if len(items) > 20:
        print(f"\n... and {len(items) - 20} more")
    
    # Look specifically for Item 7
    print(f"\n{'='*80}")
    print("SEARCHING FOR ITEM 7 (MD&A):")
    print(f"{'='*80}")
    
    item7_pattern = r'.{0,100}Item\s+7\..{0,150}'
    item7_matches = re.findall(item7_pattern, text, re.IGNORECASE)
    
    if item7_matches:
        print(f"Found {len(item7_matches)} occurrences:\n")
        for i, match in enumerate(item7_matches[:5], 1):
            clean = re.sub(r'\s+', ' ', match).strip()
            print(f"{i}. {clean}\n")
    else:
        print("❌ Item 7 NOT FOUND!")
        print("\nSearching for 'Management' keyword:")
        mgmt_pattern = r'.{0,50}Management.{0,100}'
        mgmt_matches = re.findall(mgmt_pattern, text, re.IGNORECASE)[:5]
        for match in mgmt_matches:
            clean = re.sub(r'\s+', ' ', match).strip()
            print(f"  - {clean}")


if __name__ == "__main__":
    processed_dir = Path("data/processed")
    md_files = sorted(processed_dir.glob("*.md"))
    
    if not md_files:
        print("❌ No markdown files found")
    else:
        print(f"Analyzing {len(md_files)} files...")
        
        for md_file in md_files:
            find_all_item_headers(md_file)
    
    print(f"\n{'='*80}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*80}")
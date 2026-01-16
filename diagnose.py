"""
HTML Structure Diagnostic
--------------------------
Examines the raw HTML to understand how sections are actually formatted.
"""

from pathlib import Path
from bs4 import BeautifulSoup
import re


def diagnose_html(filepath: Path):
    """Inspect actual HTML structure."""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        html = f.read()
    
    soup = BeautifulSoup(html, 'html.parser')
    
    print("=" * 80)
    print(f"DIAGNOSTIC REPORT: {filepath.name}")
    print("=" * 80)
    
    # 1. Find all text containing "Item 1A" or "Item 7"
    print("\n🔍 SEARCHING FOR SECTION HEADERS...")
    print("-" * 80)
    
    test_items = ["Item 1A", "Item 7", "Item 8", "ITEM 1A", "Risk Factors", 
                  "Management's Discussion", "Financial Statements"]
    
    for test in test_items:
        # Case-insensitive search
        matches = re.findall(f'.{{0,50}}{re.escape(test)}.{{0,50}}', html, re.IGNORECASE)
        if matches:
            print(f"\n✓ Found '{test}': {len(matches)} occurrences")
            print(f"   Sample: {matches[0][:100]}")
        else:
            print(f"\n✗ '{test}' not found")
    
    # 2. Look for common header tags
    print("\n\n📋 HTML STRUCTURE ANALYSIS")
    print("-" * 80)
    
    # Check for different header patterns
    for tag in ['h1', 'h2', 'h3', 'h4', 'p', 'div', 'span', 'td']:
        elements = soup.find_all(tag)
        
        # Filter for elements that might be section headers
        potential_headers = []
        for elem in elements[:50]:  # Check first 50
            text = elem.get_text().strip()
            if re.search(r'Item\s+\d+[A-Z]?', text, re.IGNORECASE):
                potential_headers.append({
                    'tag': tag,
                    'text': text[:100],
                    'attrs': elem.attrs
                })
        
        if potential_headers:
            print(f"\n<{tag}> tags with 'Item X' pattern: {len(potential_headers)} found")
            for i, header in enumerate(potential_headers[:3]):  # Show first 3
                print(f"   Example {i+1}:")
                print(f"      Text: {header['text']}")
                print(f"      Attrs: {header['attrs']}")
    
    # 3. Show sample of raw HTML around "Item 1A"
    print("\n\n📄 RAW HTML SAMPLE (around 'Item 1A')")
    print("-" * 80)
    
    match = re.search(r'.{200}Item\s+1A.{200}', html, re.IGNORECASE | re.DOTALL)
    if match:
        print(match.group(0))
    else:
        print("Could not find 'Item 1A' in raw HTML")
    
    # 4. Check for XBRL/iXBRL tags (inline XBRL)
    print("\n\n🏷️  XBRL TAG CHECK")
    print("-" * 80)
    
    xbrl_tags = soup.find_all(lambda tag: ':' in tag.name)[:10]
    if xbrl_tags:
        print(f"Found {len(xbrl_tags)} XBRL tags (this is inline XBRL format)")
        print("Sample tags:")
        for tag in xbrl_tags[:5]:
            print(f"   {tag.name}")
        print("\n⚠️  This document uses inline XBRL - needs special parsing!")
    else:
        print("No XBRL tags found - standard HTML")
    
    # 5. Check document type
    print("\n\n📑 DOCUMENT TYPE")
    print("-" * 80)
    
    if 'ix:' in html or 'xbrl' in html.lower():
        print("🟡 Inline XBRL Document")
        print("   Sections are embedded in XBRL tags")
        print("   Recommendation: Use Docling's XBRL parser")
    elif '<table' in html:
        print("🟢 Standard HTML with tables")
        print("   Regular parsing should work")
    else:
        print("🔴 Unknown format")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    html_files = list(raw_dir.glob("*_10K_*.html"))
    
    if html_files:
        # Diagnose Tesla file
        diagnose_html(html_files[0])
    else:
        print("No files found in data/raw/")
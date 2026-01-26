"""
One-Time Script: Fix Sentiment JSON Keys
-----------------------------------------
Renames section keys from "1a", "7", "8" to "item_1a", "item_7", "item_8"
for consistency across all sentiment files.

LOCATION: Run from project root
USAGE: python fix_json_keys.py

This is a one-time migration script. Safe to run multiple times.
"""

import json
from pathlib import Path
from datetime import datetime


def fix_sentiment_file(filepath: Path) -> dict:
    """
    Fix section keys in a single sentiment JSON file.
    
    Args:
        filepath: Path to sentiment JSON file
        
    Returns:
        Dict with status and changes made
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Track changes
    changes = []
    
    # Check if sections exist
    if 'sections' not in data:
        return {'status': 'skipped', 'reason': 'no sections key', 'changes': []}
    
    sections = data['sections']
    new_sections = {}
    
    # Key mapping
    key_map = {
        '1a': 'item_1a',
        '7': 'item_7',
        '8': 'item_8',
        '1': 'item_1',
        '1b': 'item_1b',
        '9': 'item_9',
    }
    
    for old_key, section_data in sections.items():
        if old_key in key_map:
            new_key = key_map[old_key]
            new_sections[new_key] = section_data
            changes.append(f"'{old_key}' → '{new_key}'")
        elif old_key.startswith('item_'):
            # Already in correct format
            new_sections[old_key] = section_data
        else:
            # Unknown key, keep as is
            new_sections[old_key] = section_data
    
    if not changes:
        return {'status': 'already_fixed', 'reason': 'keys already in item_X format', 'changes': []}
    
    # Update sections
    data['sections'] = new_sections
    
    # Add migration metadata
    data['_migrated'] = {
        'date': datetime.now().isoformat(),
        'changes': changes
    }
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    return {'status': 'fixed', 'changes': changes}


def main():
    print("=" * 70)
    print("SENTIMENT JSON KEY MIGRATION")
    print("=" * 70)
    print("\nThis script renames section keys:")
    print("  '1a' → 'item_1a'")
    print("  '7'  → 'item_7'")
    print("  '8'  → 'item_8'")
    print()
    
    sentiment_dir = Path("data/sentiment")
    
    if not sentiment_dir.exists():
        print(f"❌ Directory not found: {sentiment_dir}")
        return
    
    json_files = list(sentiment_dir.glob("*_sentiment.json"))
    
    if not json_files:
        print(f"❌ No sentiment JSON files found in {sentiment_dir}")
        return
    
    print(f"Found {len(json_files)} sentiment files\n")
    print("-" * 70)
    
    stats = {
        'fixed': 0,
        'already_fixed': 0,
        'skipped': 0,
        'errors': 0
    }
    
    for filepath in sorted(json_files):
        try:
            result = fix_sentiment_file(filepath)
            
            if result['status'] == 'fixed':
                stats['fixed'] += 1
                print(f"✓ FIXED: {filepath.name}")
                for change in result['changes']:
                    print(f"    {change}")
            
            elif result['status'] == 'already_fixed':
                stats['already_fixed'] += 1
                print(f"○ OK: {filepath.name} (already correct)")
            
            else:
                stats['skipped'] += 1
                print(f"⏭ SKIP: {filepath.name} ({result['reason']})")
        
        except Exception as e:
            stats['errors'] += 1
            print(f"❌ ERROR: {filepath.name} - {e}")
    
    print("-" * 70)
    print("\nSUMMARY:")
    print(f"  Fixed:         {stats['fixed']}")
    print(f"  Already OK:    {stats['already_fixed']}")
    print(f"  Skipped:       {stats['skipped']}")
    print(f"  Errors:        {stats['errors']}")
    print(f"  Total:         {len(json_files)}")
    print()
    
    if stats['fixed'] > 0:
        print("✅ Migration complete! All files now use 'item_X' format.")
    else:
        print("✅ No changes needed. All files already consistent.")


if __name__ == "__main__":
    main()
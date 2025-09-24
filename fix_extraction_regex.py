#!/usr/bin/env python3
"""
Fix the regex pattern for state extraction
"""

import pandas as pd
import re

def test_regex_patterns():
    """Test different regex patterns for state extraction."""
    print("🔍 Testing Regex Patterns for State Extraction")
    print("=" * 60)
    
    # Test string from the file
    test_string = '"Report" for : ANDHRA PRADESH for year 2022-2023'
    print(f"📄 Test string: {test_string}")
    
    # Test different regex patterns
    patterns = [
        r'for\s*:\s*([^f]+?)\s*for\s*year',  # Original pattern
        r'for\s*:\s*([^f]+?)\s*for\s*year',  # Same as above
        r'for\s*:\s*([A-Z\s]+?)\s*for\s*year',  # Only uppercase and spaces
        r'for\s*:\s*([^f]+?)\s*for\s*year',  # Original
        r'for\s*:\s*([^f]+?)\s*for\s*year',  # Original
    ]
    
    for i, pattern in enumerate(patterns):
        print(f"\n🔍 Pattern {i+1}: {pattern}")
        match = re.search(pattern, test_string, re.IGNORECASE)
        if match:
            state_name = match.group(1).strip()
            print(f"   ✅ Match found: '{state_name}'")
        else:
            print(f"   ❌ No match")
    
    # Test the correct pattern
    print(f"\n🎯 Testing correct pattern:")
    correct_pattern = r'for\s*:\s*([A-Z\s]+?)\s*for\s*year'
    match = re.search(correct_pattern, test_string, re.IGNORECASE)
    if match:
        state_name = match.group(1).strip()
        print(f"   ✅ Correct extraction: '{state_name}'")
    else:
        print(f"   ❌ Still no match")

def fix_extraction_with_correct_regex():
    """Fix the extraction with the correct regex pattern."""
    print("\n🔧 Fixing Extraction with Correct Regex")
    print("=" * 50)
    
    def extract_state_from_filename_fixed(file_path):
        """Extract state name from the first row with correct regex."""
        try:
            # Read just the first few rows
            df = pd.read_excel(file_path, header=None, nrows=5)
            
            # Look for state name in the first row
            first_row = df.iloc[0].tolist()
            for cell in first_row:
                if pd.notna(cell):
                    cell_str = str(cell)
                    # Look for pattern: "Report" for : STATE NAME for year
                    if 'for :' in cell_str and 'for year' in cell_str:
                        # Use the correct regex pattern
                        match = re.search(r'for\s*:\s*([A-Z\s]+?)\s*for\s*year', cell_str, re.IGNORECASE)
                        if match:
                            state_name = match.group(1).strip()
                            print(f"   ✅ Extracted state: {state_name}")
                            return state_name
            
            return "Unknown"
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return "Unknown"
    
    # Test with the problematic file
    test_file = "INGRIS DATASETS/INGRIS DATASETS/2022-2023 dataset/'2.xlsx"
    result = extract_state_from_filename_fixed(test_file)
    print(f"📊 Result: {result}")

if __name__ == "__main__":
    test_regex_patterns()
    fix_extraction_with_correct_regex()

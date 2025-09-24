#!/usr/bin/env python3
"""
Check all rows to find where the state data is
"""

import pandas as pd
import re

def check_all_rows():
    """Check all rows to find the state data."""
    print("🔍 Checking All Rows for State Data")
    print("=" * 50)
    
    test_file = "INGRIS DATASETS/INGRIS DATASETS/2022-2023 dataset/'2.xlsx"
    
    try:
        print(f"📄 Reading file: {test_file}")
        df = pd.read_excel(test_file, header=None, nrows=10)
        print(f"📊 Shape: {df.shape}")
        
        print("\n📋 All rows (first 10):")
        for i in range(min(10, len(df))):
            row_data = df.iloc[i].tolist()
            non_null = [str(x) for x in row_data if pd.notna(x)]
            print(f"   Row {i}: {non_null[:3]}...")
            
            # Check if this row contains state data
            for cell in row_data:
                if pd.notna(cell):
                    cell_str = str(cell)
                    if 'for :' in cell_str and 'for year' in cell_str:
                        print(f"     🎯 FOUND STATE DATA in row {i}: {cell_str}")
                        # Extract state
                        match = re.search(r'for\s*:\s*([A-Z\s]+?)\s*for\s*year', cell_str, re.IGNORECASE)
                        if match:
                            state_name = match.group(1).strip()
                            print(f"     ✅ Extracted state: '{state_name}'")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_all_rows()

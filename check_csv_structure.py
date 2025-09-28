#!/usr/bin/env python3
"""
Check CSV structure and find Chhattisgarh data
"""

import pandas as pd

def check_csv_structure():
    """Check CSV structure and find Chhattisgarh"""
    print("Checking CSV structure...")
    
    try:
        # Load CSV data
        df = pd.read_csv("ingris_rag_ready_complete.csv", low_memory=False)
        print(f"CSV loaded: {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        
        # Check each column for state-like data
        for col in df.columns:
            if 'state' in col.lower() or 'STATE' in col:
                print(f"\nColumn '{col}':")
                unique_values = df[col].dropna().unique()
                print(f"  Unique values: {len(unique_values)}")
                print(f"  Sample values: {list(unique_values[:10])}")
                
                # Check for Chhattisgarh
                chhattisgarh_mask = df[col].astype(str).str.contains('CHHATTISGARH|CHATTISGARH', case=False, na=False)
                chhattisgarh_count = chhattisgarh_mask.sum()
                print(f"  Chhattisgarh records: {chhattisgarh_count}")
                
                if chhattisgarh_count > 0:
                    chhattisgarh_data = df[chhattisgarh_mask]
                    print(f"  Sample Chhattisgarh records:")
                    for i, (idx, row) in enumerate(chhattisgarh_data.head(3).iterrows()):
                        print(f"    Row {i+1}: {dict(row)}")
        
        # Also check district column for Chhattisgarh districts
        print(f"\nChecking district column...")
        if 'district' in df.columns:
            district_chhattisgarh = df[df['district'].astype(str).str.contains('CHHATTISGARH|CHATTISGARH', case=False, na=False)]
            print(f"Districts with Chhattisgarh: {len(district_chhattisgarh)}")
            if len(district_chhattisgarh) > 0:
                print(f"Sample districts: {district_chhattisgarh['district'].unique()[:5]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_csv_structure()

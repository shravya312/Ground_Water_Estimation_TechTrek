#!/usr/bin/env python3
"""
Check the original data files to see what's available
"""

import pandas as pd
import glob

def check_original_data():
    """Check what data is in the original Excel files."""
    print("🔍 Checking Original Data Files")
    print("=" * 50)
    
    files = glob.glob('datasets123/datasets/*.xlsx')
    print(f"📁 Found {len(files)} Excel files")
    
    all_states = set()
    all_districts = set()
    all_years = set()
    
    for i, file in enumerate(files[:3]):  # Check first 3 files
        print(f"\n📄 File {i+1}: {file.split('/')[-1]}")
        try:
            df = pd.read_excel(file)
            print(f"   Columns: {list(df.columns)}")
            
            # Check for STATE column
            if "STATE" in df.columns:
                states = df["STATE"].dropna().unique()
                print(f"   States: {list(states)[:10]}")
                all_states.update(states)
            else:
                print("   No STATE column found")
            
            # Check for DISTRICT column
            if "DISTRICT" in df.columns:
                districts = df["DISTRICT"].dropna().unique()
                print(f"   Districts: {list(districts)[:10]}")
                all_districts.update(districts)
            else:
                print("   No DISTRICT column found")
            
            # Check for Assessment_Year column
            if "Assessment_Year" in df.columns:
                years = df["Assessment_Year"].dropna().unique()
                print(f"   Years: {list(years)}")
                all_years.update(years)
            else:
                print("   No Assessment_Year column found")
                
            print(f"   Total rows: {len(df)}")
            
        except Exception as e:
            print(f"   Error reading file: {e}")
    
    print(f"\n📊 Overall Summary:")
    print(f"   All States: {sorted(list(all_states))}")
    print(f"   All Districts: {sorted(list(all_districts))[:20]}...")
    print(f"   All Years: {sorted(list(all_years))}")
    
    # Check specifically for Karnataka
    karnataka_found = any("karnataka" in str(state).lower() for state in all_states)
    print(f"\n🔍 Karnataka Found: {karnataka_found}")

if __name__ == "__main__":
    check_original_data()

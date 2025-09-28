#!/usr/bin/env python3
"""
Simple CSV data check
"""

import pandas as pd

def check_csv_data():
    """Check CSV data structure"""
    print("Checking CSV Data Structure")
    print("=" * 50)
    
    try:
        # Load CSV
        df = pd.read_csv('ingris_rag_ready_complete.csv', low_memory=False)
        print(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Check state data
        if 'state' in df.columns:
            print(f"\nStates in CSV:")
            states = df['state'].dropna().unique()
            print(f"Unique states: {len(states)}")
            
            # Check Karnataka specifically
            karnataka_data = df[df['state'].str.upper() == 'KARNATAKA']
            print(f"\nKarnataka records in CSV: {len(karnataka_data)}")
            
            if len(karnataka_data) > 0:
                print("Sample Karnataka records:")
                sample = karnataka_data[['state', 'district', 'year']].head(5)
                for idx, row in sample.iterrows():
                    print(f"  {row['state']} - {row['district']} - {row['year']}")
        else:
            print("No 'state' column found in CSV")
        
        # Check year data
        if 'year' in df.columns:
            df['year_numeric'] = pd.to_numeric(df['year'], errors='coerce')
            valid_years = df['year_numeric'].dropna()
            if len(valid_years) > 0:
                print(f"\nYear range: {valid_years.min()} - {valid_years.max()}")
            else:
                print("\nNo valid year data found")
        
        # Check for combined_text column
        if 'combined_text' in df.columns:
            print(f"\nCombined_text column found: {len(df['combined_text'].dropna())} non-null values")
        else:
            print("\nNo combined_text column found")
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_csv_data()

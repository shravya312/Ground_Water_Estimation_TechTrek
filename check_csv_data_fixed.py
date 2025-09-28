#!/usr/bin/env python3
"""
Check CSV data structure - fixed version
"""

import pandas as pd
import numpy as np

def check_csv_data():
    """Check CSV data structure"""
    print("Checking CSV Data Structure")
    print("=" * 50)
    
    try:
        # Load CSV
        df = pd.read_csv('ingris_rag_ready_complete.csv', low_memory=False)
        print(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Check column names
        print(f"\nColumn names: {list(df.columns)}")
        
        # Check state data
        if 'state' in df.columns:
            print(f"\nStates in CSV:")
            states = df['state'].unique()
            print(f"Unique states: {len(states)}")
            print(f"States: {sorted(states)}")
            
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
        
        # Check year data - handle mixed types
        if 'year' in df.columns:
            # Convert to numeric, coercing errors
            df['year_numeric'] = pd.to_numeric(df['year'], errors='coerce')
            valid_years = df['year_numeric'].dropna()
            if len(valid_years) > 0:
                print(f"\nYear range: {valid_years.min()} - {valid_years.max()}")
            else:
                print("\nNo valid year data found")
        elif 'Assessment_Year' in df.columns:
            df['Assessment_Year_numeric'] = pd.to_numeric(df['Assessment_Year'], errors='coerce')
            valid_years = df['Assessment_Year_numeric'].dropna()
            if len(valid_years) > 0:
                print(f"\nAssessment_Year range: {valid_years.min()} - {valid_years.max()}")
            else:
                print("\nNo valid Assessment_Year data found")
        
        # Check for combined_text column
        if 'combined_text' in df.columns:
            print(f"\nCombined_text column found: {len(df['combined_text'].dropna())} non-null values")
        else:
            print("\nNo combined_text column found")
        
        # Check data types
        print(f"\nData types:")
        print(df.dtypes.value_counts())
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_csv_data()

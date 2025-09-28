#!/usr/bin/env python3
"""
Check trend-related fields in the data
"""

import pandas as pd

def check_trend_fields():
    """Check what trend-related fields are available"""
    print("Checking Trend Fields in Data")
    print("=" * 40)
    
    try:
        # Load the CSV data
        df = pd.read_csv("ingris_rag_ready_complete.csv", low_memory=False)
        print(f"CSV loaded: {len(df)} records")
        
        # Get all column names
        columns = df.columns.tolist()
        print(f"Total columns: {len(columns)}")
        
        # Find trend-related columns
        trend_columns = [col for col in columns if 'trend' in col.lower() or 'monsoon' in col.lower()]
        print(f"\nTrend-related columns: {trend_columns}")
        
        # Check for specific trend fields
        specific_fields = [
            'pre_monsoon_of_gw_trend',
            'post_monsoon_of_gw_trend',
            'pre_monsoon_gw_trend',
            'post_monsoon_gw_trend',
            'pre_monsoon_trend',
            'post_monsoon_trend'
        ]
        
        print(f"\nChecking specific trend fields:")
        for field in specific_fields:
            if field in columns:
                print(f"  FOUND: {field}")
                # Show sample values
                sample_values = df[field].dropna().unique()[:5]
                print(f"    Sample values: {list(sample_values)}")
            else:
                print(f"  NOT FOUND: {field}")
        
        # Show all columns that might be related
        print(f"\nAll columns containing 'trend' or 'monsoon':")
        for col in trend_columns:
            print(f"  - {col}")
            if col in df.columns:
                sample_values = df[col].dropna().unique()[:3]
                print(f"    Sample values: {list(sample_values)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_trend_fields()

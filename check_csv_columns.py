#!/usr/bin/env python3
"""
Check CSV column names and sample data
"""

import pandas as pd

def check_csv_structure():
    """Check CSV structure and columns"""
    print("🔍 Checking CSV Structure")
    print("=" * 40)
    
    try:
        # Load the CSV data
        print("🔄 Loading data from ingris_rag_ready_complete.csv...")
        df = pd.read_csv("ingris_rag_ready_complete.csv", nrows=5)  # Load only first 5 rows
        print(f"✅ Loaded sample data")
        
        # Show column names
        print(f"\n📊 Total columns: {len(df.columns)}")
        print("\n📋 Column names:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        # Check for state-related columns
        print("\n🔍 Looking for state-related columns:")
        state_columns = [col for col in df.columns if 'state' in col.lower() or 'STATE' in col]
        if state_columns:
            print("Found state columns:")
            for col in state_columns:
                print(f"   - {col}")
        else:
            print("No obvious state columns found")
        
        # Show sample data
        print("\n📄 Sample data (first row):")
        for col in df.columns[:10]:  # Show first 10 columns
            value = df.iloc[0][col]
            print(f"{col}: {value}")
        
        return df.columns.tolist()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    check_csv_structure()

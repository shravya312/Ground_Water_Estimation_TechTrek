#!/usr/bin/env python3
"""
Check ingris_rag_ready_complete.csv data completeness
"""

import pandas as pd

def check_csv_data():
    print("Checking ingris_rag_ready_complete.csv Data")
    print("=" * 50)
    
    try:
        # Read CSV
        print("1. Reading CSV file...")
        df = pd.read_csv('ingris_rag_ready_complete.csv')
        print(f"[OK] CSV loaded successfully")
        
        # Basic stats
        print(f"\n2. Basic Statistics:")
        print(f"   Total records: {len(df):,}")
        print(f"   Total columns: {len(df.columns)}")
        print(f"   File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # State analysis
        print(f"\n3. State Analysis:")
        print(f"   Unique states: {df['state'].nunique()}")
        print(f"   Karnataka records: {len(df[df['state'] == 'KARNATAKA']):,}")
        
        # Top 10 states
        print(f"\n   Top 10 states by record count:")
        state_counts = df['state'].value_counts().head(10)
        for state, count in state_counts.items():
            print(f"     {state}: {count:,} records")
        
        # Check column names
        print(f"\n4. Column Names:")
        print(f"   Available columns: {list(df.columns)}")
        
        # Find year column
        year_col = None
        for col in df.columns:
            if col.lower() == 'year' or 'assessment' in col.lower():
                year_col = col
                break
        
        if year_col:
            print(f"   Year column found: {year_col}")
            # Handle mixed data types
            year_data = df[year_col].dropna()
            try:
                # Try to convert to numeric
                year_data = pd.to_numeric(year_data, errors='coerce').dropna()
                years = sorted(year_data.unique())
                print(f"   Years covered: {years}")
                print(f"   Year range: {min(years)} - {max(years)}")
            except:
                print(f"   Year data (first 10): {year_data.head(10).tolist()}")
        else:
            print("   [WARNING] No year column found")
        
        # District analysis
        print(f"\n5. District Analysis:")
        print(f"   Unique districts: {df['district'].nunique()}")
        
        # Karnataka specific analysis
        print(f"\n6. Karnataka Specific Analysis:")
        karnataka_df = df[df['state'] == 'KARNATAKA']
        if len(karnataka_df) > 0:
            print(f"   Karnataka records: {len(karnataka_df):,}")
            print(f"   Karnataka districts: {karnataka_df['district'].nunique()}")
            if year_col:
                try:
                    karnataka_years = karnataka_df[year_col].dropna()
                    karnataka_years = pd.to_numeric(karnataka_years, errors='coerce').dropna()
                    print(f"   Karnataka years: {sorted(karnataka_years.unique())}")
                except:
                    print(f"   Karnataka year data: {karnataka_df[year_col].head(10).tolist()}")
            
            print(f"\n   Top 10 Karnataka districts:")
            karnataka_districts = karnataka_df['district'].value_counts().head(10)
            for district, count in karnataka_districts.items():
                print(f"     {district}: {count} records")
        else:
            print("   [WARNING] No Karnataka records found!")
        
        # Data completeness
        print(f"\n7. Data Completeness:")
        print(f"   Records with missing state: {df['state'].isna().sum()}")
        print(f"   Records with missing district: {df['district'].isna().sum()}")
        if year_col:
            print(f"   Records with missing year: {df[year_col].isna().sum()}")
        
        # Sample data
        print(f"\n8. Sample Data (first 3 records):")
        sample_cols = ['serial_number', 'state', 'district']
        if year_col:
            sample_cols.append(year_col)
        sample_cols.extend(['rainfall_mm', 'ground_water_recharge_ham'])
        
        # Only include columns that exist
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(3).to_string(index=False))
        
        print(f"\n[SUCCESS] CSV data analysis completed!")
        print(f"The file contains {len(df):,} complete groundwater records.")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to analyze CSV: {e}")
        return False

if __name__ == "__main__":
    check_csv_data()

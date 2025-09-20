#!/usr/bin/env python3
"""
Check quality data in the CSV file
"""

import pandas as pd

def check_quality_data():
    """Check what quality data is available in the CSV"""
    print("🔍 Checking Quality Data in CSV")
    print("=" * 50)
    
    # Load the dataset
    df = pd.read_csv('master_groundwater_data.csv', low_memory=False)
    
    # Find quality-related columns
    quality_cols = [col for col in df.columns if 'quality' in col.lower() or 'tagging' in col.lower()]
    
    print(f"📊 Found {len(quality_cols)} quality-related columns:")
    for i, col in enumerate(quality_cols, 1):
        print(f"   {i}. {col}")
    
    print(f"\n📋 Sample data from quality columns:")
    if quality_cols:
        sample_data = df[quality_cols].head(10)
        print(sample_data.to_string())
        
        print(f"\n📈 Data types:")
        for col in quality_cols:
            print(f"   {col}: {df[col].dtype}")
        
        print(f"\n🔢 Non-null values:")
        for col in quality_cols:
            non_null_count = df[col].notna().sum()
            total_count = len(df)
            print(f"   {col}: {non_null_count}/{total_count} ({non_null_count/total_count*100:.1f}%)")
        
        print(f"\n📝 Unique values in quality columns:")
        for col in quality_cols:
            unique_vals = df[col].dropna().unique()
            print(f"   {col}: {len(unique_vals)} unique values")
            if len(unique_vals) <= 20:  # Show all if not too many
                print(f"      Values: {list(unique_vals)}")
            else:  # Show first 10 if too many
                print(f"      First 10 values: {list(unique_vals[:10])}")
    else:
        print("❌ No quality-related columns found!")
        
        # Check for other potential quality indicators
        print("\n🔍 Looking for other potential quality indicators...")
        potential_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                          ['arsenic', 'fluoride', 'iron', 'manganese', 'nitrate', 'salinity', 'tds', 'ph'])]
        
        if potential_cols:
            print(f"Found {len(potential_cols)} potential quality indicator columns:")
            for col in potential_cols:
                print(f"   - {col}")
        else:
            print("No obvious quality indicator columns found")

if __name__ == "__main__":
    check_quality_data()

#!/usr/bin/env python3
"""
Check storage-related fields in the data
"""

import pandas as pd

def check_storage_fields():
    """Check what storage-related fields are available"""
    print("Checking Storage Fields in Data")
    print("=" * 40)
    
    try:
        # Load the CSV data
        df = pd.read_csv("ingris_rag_ready_complete.csv", low_memory=False)
        print(f"CSV loaded: {len(df)} records")
        
        # Get all column names
        columns = df.columns.tolist()
        
        # Find storage-related columns
        storage_keywords = ['storage', 'confined', 'unconfined', 'semi', 'aquifer', 'ground_water_availability']
        storage_columns = [col for col in columns if any(keyword in col.lower() for keyword in storage_keywords)]
        
        print(f"\nStorage-related columns ({len(storage_columns)}):")
        for col in storage_columns:
            print(f"  - {col}")
            if col in df.columns:
                # Show sample values
                sample_values = df[col].dropna().unique()[:5]
                print(f"    Sample values: {list(sample_values)}")
                # Show data type and non-null count
                non_null_count = df[col].notna().sum()
                print(f"    Non-null values: {non_null_count}/{len(df)} ({non_null_count/len(df)*100:.1f}%)")
        
        # Check specific fields that should be in the storage section
        specific_storage_fields = [
            'instorage_unconfined_ground_water_resourcesham',
            'total_ground_water_availability_in_unconfined_aquifier_ham',
            'dynamic_confined_ground_water_resourcesham',
            'instorage_confined_ground_water_resourcesham',
            'total_confined_ground_water_resources_ham',
            'dynamic_semi_confined_ground_water_resources_ham',
            'instorage_semi_confined_ground_water_resources_ham',
            'total_semiconfined_ground_water_resources_ham',
            'total_ground_water_availability_in_the_area_ham'
        ]
        
        print(f"\nChecking specific storage fields:")
        for field in specific_storage_fields:
            if field in columns:
                print(f"  FOUND: {field}")
                sample_values = df[field].dropna().unique()[:5]
                print(f"    Sample values: {list(sample_values)}")
                non_null_count = df[field].notna().sum()
                print(f"    Non-null values: {non_null_count}/{len(df)} ({non_null_count/len(df)*100:.1f}%)")
            else:
                print(f"  NOT FOUND: {field}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_storage_fields()

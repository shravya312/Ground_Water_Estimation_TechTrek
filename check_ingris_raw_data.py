#!/usr/bin/env python3
"""
Check INGRIS Raw Data - Examine original Excel files to understand state data
"""

import pandas as pd
import glob
import os

def check_ingris_raw_data():
    """Check the original INGRIS Excel files to understand the data structure."""
    print("🔍 Checking INGRIS Raw Data Structure")
    print("=" * 60)
    
    # Find all Excel files
    files = glob.glob('INGRIS DATASETS/INGRIS DATASETS/*/*.xlsx')
    print(f"📁 Found {len(files)} Excel files")
    
    if not files:
        print("❌ No Excel files found in INGRIS DATASETS folder")
        return
    
    # Check first few files
    print("\n📋 Examining first 5 files:")
    for i, file_path in enumerate(files[:5]):
        print(f"\n📄 File {i+1}: {os.path.basename(file_path)}")
        try:
            # Read Excel file
            df = pd.read_excel(file_path, header=None)
            print(f"   Shape: {df.shape}")
            
            # Look for state information in first few rows
            print("   First 10 rows:")
            for j in range(min(10, len(df))):
                row_data = df.iloc[j].tolist()
                # Filter out NaN values
                row_data = [str(x) for x in row_data if pd.notna(x)]
                if row_data:
                    print(f"     Row {j}: {row_data[:5]}...")  # Show first 5 non-null values
            
            # Look for patterns that might indicate state names
            print("   Looking for state patterns...")
            for j in range(min(20, len(df))):
                for k in range(min(5, len(df.columns))):
                    cell_value = df.iloc[j, k]
                    if pd.notna(cell_value):
                        cell_str = str(cell_value).upper()
                        if any(state in cell_str for state in ['ANDHRA', 'ARUNACHAL', 'ANDAMAN', 'STATE', 'PRADESH']):
                            print(f"     Found potential state info at row {j}, col {k}: {cell_value}")
            
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    # Check a specific file that might have state info
    print(f"\n🔍 Detailed analysis of first file: {files[0]}")
    try:
        df = pd.read_excel(files[0], header=None)
        print(f"   Full shape: {df.shape}")
        
        # Look for state information more systematically
        print("   Searching for state-related content...")
        state_keywords = ['STATE', 'ANDHRA', 'PRADESH', 'ARUNACHAL', 'ANDAMAN', 'NICOBAR']
        
        for i in range(min(50, len(df))):
            for j in range(min(10, len(df.columns))):
                cell_value = df.iloc[i, j]
                if pd.notna(cell_value):
                    cell_str = str(cell_value).upper()
                    if any(keyword in cell_str for keyword in state_keywords):
                        print(f"     Row {i}, Col {j}: {cell_value}")
        
        # Check if there are any headers or metadata rows
        print("\n   Checking for header/metadata rows...")
        for i in range(min(10, len(df))):
            row_data = df.iloc[i].tolist()
            non_null_count = sum(1 for x in row_data if pd.notna(x))
            if non_null_count > 0:
                print(f"     Row {i}: {non_null_count} non-null values")
                if non_null_count < 5:  # Likely a header/metadata row
                    print(f"       Content: {[x for x in row_data if pd.notna(x)]}")
        
    except Exception as e:
        print(f"   ❌ Error in detailed analysis: {e}")
    
    # Check if there are any summary files or metadata files
    print(f"\n🔍 Looking for summary/metadata files...")
    summary_files = glob.glob('INGRIS DATASETS/INGRIS DATASETS/*/*.xlsx')
    for file_path in summary_files:
        filename = os.path.basename(file_path).lower()
        if any(keyword in filename for keyword in ['summary', 'metadata', 'index', 'list']):
            print(f"   Found potential summary file: {file_path}")
            try:
                df = pd.read_excel(file_path, header=None)
                print(f"     Shape: {df.shape}")
                print(f"     First few rows: {df.head(3).values.tolist()}")
            except Exception as e:
                print(f"     Error reading: {e}")

if __name__ == "__main__":
    check_ingris_raw_data()

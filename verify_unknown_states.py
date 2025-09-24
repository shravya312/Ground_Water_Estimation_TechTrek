#!/usr/bin/env python3
"""
Verify Unknown States - Check if records marked as "Unknown" actually have state data in original files
"""

import pandas as pd
import glob
import os
import re

def check_unknown_records():
    """Check if records marked as Unknown actually have state data."""
    print("🔍 Verifying Unknown State Records")
    print("=" * 60)
    
    # Load the current CSV
    df = pd.read_csv('ingris_rag_ready.csv', low_memory=False)
    unknown_records = df[df['state'] == 'Unknown']
    print(f"📊 Records marked as 'Unknown': {len(unknown_records)}")
    
    # Get unique source files for unknown records
    unknown_files = unknown_records['source_file'].unique()
    print(f"📁 Files with 'Unknown' states: {len(unknown_files)}")
    print(f"   Files: {list(unknown_files)[:10]}...")  # Show first 10
    
    # Check a few of these files manually
    print(f"\n🔍 Checking original Excel files for state data...")
    
    # Find the actual Excel files
    excel_files = glob.glob('INGRIS DATASETS/INGRIS DATASETS/*/*.xlsx')
    print(f"📁 Total Excel files found: {len(excel_files)}")
    
    # Check a few files that might have "Unknown" states
    files_to_check = []
    for file_path in excel_files:
        filename = os.path.basename(file_path)
        if filename in unknown_files:
            files_to_check.append(file_path)
    
    print(f"📋 Files to check: {len(files_to_check)}")
    
    # Check first 5 files
    for i, file_path in enumerate(files_to_check[:5]):
        print(f"\n📄 File {i+1}: {os.path.basename(file_path)}")
        try:
            # Read Excel file
            df_excel = pd.read_excel(file_path, header=None)
            
            # Look for state information in first few rows
            print("   First 5 rows:")
            for j in range(min(5, len(df_excel))):
                row_data = df_excel.iloc[j].tolist()
                non_null = [str(x) for x in row_data if pd.notna(x)]
                if non_null:
                    print(f"     Row {j}: {non_null[:3]}...")  # Show first 3 non-null values
            
            # Look for STATE column
            print("   Looking for STATE column...")
            for j in range(min(20, len(df_excel))):
                for k in range(min(10, len(df_excel.columns))):
                    cell_value = df_excel.iloc[j, k]
                    if pd.notna(cell_value):
                        cell_str = str(cell_value).upper()
                        if 'STATE' in cell_str:
                            print(f"     Found 'STATE' at row {j}, col {k}: {cell_value}")
                            
                            # Check if there are actual state values in this column
                            if j < len(df_excel) - 1:
                                print(f"     Checking values in STATE column...")
                                for row_idx in range(j+1, min(j+10, len(df_excel))):
                                    state_value = df_excel.iloc[row_idx, k]
                                    if pd.notna(state_value) and str(state_value).strip():
                                        print(f"       Row {row_idx}: {state_value}")
                                        break
            
            # Look for state name in first row (filename pattern)
            first_row = df_excel.iloc[0].tolist()
            for cell in first_row:
                if pd.notna(cell):
                    cell_str = str(cell)
                    if 'for :' in cell_str and 'for year' in cell_str:
                        print(f"   State from filename: {cell_str}")
                        break
            
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    # Check if the issue is with our extraction logic
    print(f"\n🔍 Checking extraction logic...")
    
    # Look at a specific file that should have state data
    sample_file = None
    for file_path in excel_files:
        filename = os.path.basename(file_path)
        if filename in unknown_files:
            sample_file = file_path
            break
    
    if sample_file:
        print(f"📄 Detailed analysis of: {os.path.basename(sample_file)}")
        try:
            df_excel = pd.read_excel(sample_file, header=None)
            print(f"   Shape: {df_excel.shape}")
            
            # Find header row
            header_row = -1
            for i in range(min(10, len(df_excel))):
                if any(isinstance(cell, str) and 'S.No' in str(cell) for cell in df_excel.iloc[i]):
                    header_row = i
                    break
            
            if header_row >= 0:
                print(f"   Header row found at: {header_row}")
                headers = df_excel.iloc[header_row].tolist()
                print(f"   Headers: {[h for h in headers if pd.notna(h)][:10]}...")
                
                # Check if STATE is in headers
                state_col_idx = None
                for idx, header in enumerate(headers):
                    if pd.notna(header) and 'STATE' in str(header).upper():
                        state_col_idx = idx
                        break
                
                if state_col_idx is not None:
                    print(f"   STATE column found at index: {state_col_idx}")
                    # Check actual state values
                    print(f"   State values in first 10 data rows:")
                    for i in range(header_row + 1, min(header_row + 11, len(df_excel))):
                        state_value = df_excel.iloc[i, state_col_idx]
                        if pd.notna(state_value):
                            print(f"     Row {i}: {state_value}")
                else:
                    print(f"   ❌ No STATE column found in headers")
            else:
                print(f"   ❌ No header row found")
                
        except Exception as e:
            print(f"   ❌ Error in detailed analysis: {e}")

if __name__ == "__main__":
    check_unknown_records()

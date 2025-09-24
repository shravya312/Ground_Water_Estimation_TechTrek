#!/usr/bin/env python3
"""
Investigate original data structure to find missing records
"""

import pandas as pd
import glob
import os
import re

def investigate_original_data():
    """Investigate the original data structure to understand why we're missing records."""
    print("🔍 Investigating Original Data Structure")
    print("=" * 60)
    
    # Check all Excel files in the INGRIS datasets
    excel_files = glob.glob('INGRIS DATASETS/INGRIS DATASETS/*/*.xlsx')
    print(f"📁 Found {len(excel_files)} Excel files")
    
    # Analyze a few files to understand the structure
    print(f"\n📋 Analyzing file structures...")
    
    total_records_found = 0
    files_processed = 0
    
    for i, file_path in enumerate(excel_files[:10]):  # Check first 10 files
        print(f"\n📄 File {i+1}: {os.path.basename(file_path)}")
        try:
            # Read Excel file
            df = pd.read_excel(file_path, header=None)
            print(f"   Shape: {df.shape}")
            
            # Find header row
            header_row = -1
            for j in range(min(15, len(df))):
                if any(isinstance(cell, str) and 'S.No' in str(cell) for cell in df.iloc[j]):
                    header_row = j
                    break
            
            if header_row >= 0:
                print(f"   Header row: {header_row}")
                headers = df.iloc[header_row].tolist()
                print(f"   Headers: {[h for h in headers if pd.notna(h)][:10]}...")
                
                # Count data rows
                data_rows = len(df) - header_row - 1
                print(f"   Data rows: {data_rows}")
                total_records_found += data_rows
                files_processed += 1
                
                # Check for state information
                state_found = False
                for k in range(min(5, len(df))):
                    for cell in df.iloc[k]:
                        if pd.notna(cell) and 'for :' in str(cell) and 'for year' in str(cell):
                            print(f"   State info: {str(cell)[:100]}...")
                            state_found = True
                            break
                    if state_found:
                        break
                
                if not state_found:
                    print(f"   ⚠️ No state info found in first 5 rows")
            else:
                print(f"   ❌ No header row found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Analysis Summary:")
    print(f"   Files processed: {files_processed}")
    print(f"   Total records found: {total_records_found:,}")
    print(f"   Average records per file: {total_records_found/files_processed if files_processed > 0 else 0:.0f}")
    
    # Estimate total records across all files
    estimated_total = (total_records_found / files_processed) * len(excel_files) if files_processed > 0 else 0
    print(f"   Estimated total records: {estimated_total:,.0f}")
    
    return total_records_found, files_processed

def check_extraction_process():
    """Check what our extraction process is actually doing."""
    print(f"\n🔍 Checking Our Extraction Process")
    print("=" * 50)
    
    # Check what our final extraction script actually processes
    if os.path.exists('final_fixed_extraction.py'):
        print("📄 Our extraction script processes these datasets:")
        datasets = [
            ("2016-2017 dataset", "2016"),
            ("2019-2020 datset", "2019"),
            ("2021-2022 dataset", "2021"),
            ("2022-2023 dataset", "2022"),
            ("2023-2024 datset", "2023"),
            ("2024- 2025 dataset", "2024")
        ]
        
        total_expected = 0
        for dataset_name, year in datasets:
            dataset_path = f"INGRIS DATASETS/INGRIS DATASETS/{dataset_name}"
            if os.path.exists(dataset_path):
                excel_files = glob.glob(f"{dataset_path}/*.xlsx")
                print(f"   {dataset_name}: {len(excel_files)} files")
                total_expected += len(excel_files)
            else:
                print(f"   {dataset_name}: ❌ Not found")
        
        print(f"\n📊 Expected files: {total_expected}")
        
        # Check if we're missing any datasets
        all_excel_files = glob.glob('INGRIS DATASETS/INGRIS DATASETS/*/*.xlsx')
        print(f"📊 Actual files found: {len(all_excel_files)}")
        
        if len(all_excel_files) > total_expected:
            print(f"⚠️ We might be missing some datasets in our extraction!")
            
            # Find which datasets we're not processing
            processed_datasets = [d[0] for d in datasets]
            actual_datasets = set()
            for file_path in all_excel_files:
                dataset_name = file_path.split('/')[-2]
                actual_datasets.add(dataset_name)
            
            missing_datasets = actual_datasets - set(processed_datasets)
            if missing_datasets:
                print(f"📋 Datasets we're not processing:")
                for dataset in missing_datasets:
                    print(f"   - {dataset}")

def check_column_differences():
    """Check if there are column name differences causing data loss."""
    print(f"\n🔍 Checking Column Name Differences")
    print("=" * 50)
    
    # Check a few files to see if column names are different
    excel_files = glob.glob('INGRIS DATASETS/INGRIS DATASETS/*/*.xlsx')
    
    all_headers = set()
    
    for i, file_path in enumerate(excel_files[:5]):  # Check first 5 files
        try:
            df = pd.read_excel(file_path, header=None)
            
            # Find header row
            header_row = -1
            for j in range(min(15, len(df))):
                if any(isinstance(cell, str) and 'S.No' in str(cell) for cell in df.iloc[j]):
                    header_row = j
                    break
            
            if header_row >= 0:
                headers = df.iloc[header_row].tolist()
                clean_headers = [str(h).strip().lower() for h in headers if pd.notna(h)]
                all_headers.update(clean_headers)
                print(f"📄 {os.path.basename(file_path)}: {len(clean_headers)} columns")
                print(f"   Sample headers: {clean_headers[:5]}...")
        
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print(f"\n📊 All unique headers found: {len(all_headers)}")
    print(f"📋 Sample headers: {list(all_headers)[:20]}...")

if __name__ == "__main__":
    # Investigate original data
    total_records, files_processed = investigate_original_data()
    
    # Check extraction process
    check_extraction_process()
    
    # Check column differences
    check_column_differences()
    
    print(f"\n🎯 Conclusion:")
    print(f"   If we found {total_records:,} records in {files_processed} files,")
    print(f"   we should expect much more than 63,374 records total.")
    print(f"   There might be an issue with our extraction process or dataset coverage.")

#!/usr/bin/env python3
"""
Final Fix for INGRIS Extraction - Extract state from filename when STATE column is missing
"""

import pandas as pd
import glob
import os
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def clean_header(header):
    """Clean and standardize header names."""
    if pd.isna(header) or header == '':
        return 'unknown_column'
    
    # Convert to string and clean
    header_str = str(header).strip()
    
    # Remove special characters and replace with underscores
    header_str = re.sub(r'[^a-zA-Z0-9\s]', '', header_str)
    header_str = re.sub(r'\s+', '_', header_str)
    header_str = header_str.lower()
    
    # Handle common variations
    header_mappings = {
        's_no': 'serial_number',
        'sno': 'serial_number',
        'state': 'state',
        'district': 'district',
        'assessment_unit': 'assessment_unit',
        'year': 'year',
        'assessment_year': 'year'
    }
    
    return header_mappings.get(header_str, header_str)

def extract_state_from_filename(file_path):
    """Extract state name from the first row of the Excel file."""
    try:
        # Read just the first few rows to get the state name
        df = pd.read_excel(file_path, header=None, nrows=5)
        
        # Look for state name in the first row
        first_row = df.iloc[0].tolist()
        for cell in first_row:
            if pd.notna(cell):
                cell_str = str(cell)
                # Look for pattern: "Report" for : STATE NAME for year
                if 'for :' in cell_str and 'for year' in cell_str:
                    # Extract state name between "for :" and "for year"
                    match = re.search(r'for\s*:\s*([^f]+?)\s*for\s*year', cell_str, re.IGNORECASE)
                    if match:
                        state_name = match.group(1).strip()
                        logger.info(f"Extracted state from filename: {state_name}")
                        return state_name
        
        return "Unknown"
    except Exception as e:
        logger.warning(f"Could not extract state from {file_path}: {e}")
        return "Unknown"

def extract_data_from_excel_final(file_path, year=None):
    """Extract data from a single Excel file with proper state extraction."""
    try:
        # First, extract state from the file
        state_from_filename = extract_state_from_filename(file_path)
        
        # Read Excel file
        df = pd.read_excel(file_path, header=None)
        
        # Find the header row (look for "S.No" or similar)
        header_row_index = -1
        for i in range(min(df.shape[0], 10)):  # Check first 10 rows
            if any(isinstance(cell, str) and re.search(r'S\.?\s*No\.?', str(cell), re.IGNORECASE) for cell in df.iloc[i]):
                header_row_index = i
                break
        
        if header_row_index == -1:
            logger.warning(f"Could not find header row in {file_path}")
            return []
        
        # Use the identified row as header
        headers = df.iloc[header_row_index].fillna(method='ffill').fillna('')
        df = df[header_row_index + 1:].copy()
        df.columns = [clean_header(h) for h in headers]
        
        # Drop rows where all key columns are NaN
        df.dropna(how='all', inplace=True)
        
        # Add metadata
        df['source_file'] = os.path.basename(file_path)
        df['year'] = year if year else 'Unknown'
        
        # Handle STATE column
        if 'state' in df.columns:
            # Keep the original STATE column values
            logger.info(f"Found STATE column in {file_path}, using actual values")
        else:
            # If no STATE column, use the extracted state for all rows
            df['state'] = state_from_filename
            logger.info(f"No STATE column in {file_path}, using extracted state: {state_from_filename}")
        
        # Convert to list of dictionaries
        data = df.to_dict('records')
        
        logger.info(f"Extracted {len(data)} rows from {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return []

def process_ingris_datasets_final():
    """Process all INGRIS datasets with final state extraction fix."""
    print("🔧 Final INGRIS Processing - Extract State from Filename")
    print("=" * 60)
    
    all_data = []
    
    # Process each year dataset
    year_datasets = [
        ("2016-2017 dataset", "2016"),
        ("2019-2020 datset", "2019"),
        ("2021-2022 dataset", "2021"),
        ("2022-2023 dataset", "2022"),
        ("2023-2024 datset", "2023"),
        ("2024- 2025 dataset", "2024")
    ]
    
    for dataset_name, year in year_datasets:
        print(f"\n📁 Processing {dataset_name}...")
        dataset_path = f"INGRIS DATASETS/INGRIS DATASETS/{dataset_name}"
        
        if not os.path.exists(dataset_path):
            print(f"   ⚠️ Dataset path not found: {dataset_path}")
            continue
        
        # Get all Excel files in this dataset
        excel_files = glob.glob(f"{dataset_path}/*.xlsx")
        print(f"   📄 Found {len(excel_files)} Excel files")
        
        for file_path in excel_files:
            data = extract_data_from_excel_final(file_path, year)
            all_data.extend(data)
    
    if all_data:
        # Create DataFrame
        df = pd.DataFrame(all_data)
        print(f"\n📊 Total extracted: {len(df)} rows")
        
        # Check state distribution
        if 'state' in df.columns:
            print("\n📋 State distribution:")
            state_counts = df['state'].value_counts()
            print(state_counts.head(15))
            
            # Check how many are still Unknown
            unknown_count = state_counts.get('Unknown', 0)
            valid_count = len(df) - unknown_count
            print(f"\n✅ Valid states: {valid_count} records")
            print(f"❌ Unknown states: {unknown_count} records")
            print(f"📊 Success rate: {(valid_count/len(df)*100):.1f}%")
        
        # Save the final CSV
        output_file = "ingris_rag_ready_final.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Final CSV saved: {output_file}")
        
        # Also update the original file
        df.to_csv("ingris_rag_ready.csv", index=False)
        print("✅ Updated original ingris_rag_ready.csv")
        
        return df
    else:
        print("❌ No data extracted")
        return None

if __name__ == "__main__":
    df = process_ingris_datasets_final()
    if df is not None:
        print(f"\n🎉 Final extraction complete! {len(df)} rows with proper state data.")

#!/usr/bin/env python3
"""
Extract ALL datasets - fix the missing data issue
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

def extract_state_from_filename_fixed(file_path):
    """Extract state name from the first few rows of the Excel file."""
    try:
        # Read the first 10 rows to find state data
        df = pd.read_excel(file_path, header=None, nrows=10)
        
        # Look for state name in the first few rows
        for row_idx in range(min(10, len(df))):
            row_data = df.iloc[row_idx].tolist()
            for cell in row_data:
                if pd.notna(cell):
                    cell_str = str(cell)
                    # Look for pattern: "Report" for : STATE NAME for year
                    if 'for :' in cell_str and 'for year' in cell_str:
                        # Extract state name between "for :" and "for year"
                        match = re.search(r'for\s*:\s*([A-Z\s]+?)\s*for\s*year', cell_str, re.IGNORECASE)
                        if match:
                            state_name = match.group(1).strip()
                            logger.info(f"Extracted state from row {row_idx}: {state_name}")
                            return state_name
        
        return "Unknown"
    except Exception as e:
        logger.warning(f"Could not extract state from {file_path}: {e}")
        return "Unknown"

def extract_data_from_excel_final_fixed(file_path, year=None):
    """Extract data from a single Excel file with proper state extraction."""
    try:
        # First, extract state from the file
        state_from_filename = extract_state_from_filename_fixed(file_path)
        
        # Read Excel file
        df = pd.read_excel(file_path, header=None)
        
        # Find the header row (look for "S.No" or similar)
        header_row_index = -1
        for i in range(min(df.shape[0], 15)):  # Check first 15 rows
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

def extract_all_datasets():
    """Extract data from ALL available datasets."""
    print("🚀 Extracting ALL Available Datasets")
    print("=" * 60)
    
    all_data = []
    
    # Find ALL dataset directories
    base_path = "INGRIS DATASETS/INGRIS DATASETS"
    if not os.path.exists(base_path):
        print(f"❌ Base path not found: {base_path}")
        return None
    
    # Get all subdirectories (datasets)
    all_datasets = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print(f"📁 Found {len(all_datasets)} datasets:")
    for dataset in all_datasets:
        print(f"   - {dataset}")
    
    # Process each dataset
    for dataset_name in all_datasets:
        print(f"\n📁 Processing dataset: {dataset_name}")
        dataset_path = os.path.join(base_path, dataset_name)
        
        # Try to extract year from dataset name
        year_match = re.search(r'(\d{4})', dataset_name)
        current_year = year_match.group(1) if year_match else 'Unknown'
        
        # Get all Excel files in this dataset
        excel_files = glob.glob(os.path.join(dataset_path, '*.xlsx'))
        print(f"   📄 Found {len(excel_files)} Excel files")
        
        if not excel_files:
            print(f"   ⚠️ No Excel files found in {dataset_name}")
            continue
        
        # Process each Excel file
        for file_path in excel_files:
            data = extract_data_from_excel_final_fixed(file_path, year=current_year)
            all_data.extend(data)
    
    if all_data:
        # Create DataFrame
        df = pd.DataFrame(all_data)
        print(f"\n📊 Total extracted: {len(df):,} records")
        
        # Check state distribution
        if 'state' in df.columns:
            print("\n📋 State distribution:")
            state_counts = df['state'].value_counts()
            print(state_counts.head(15))
            
            # Check how many are still Unknown
            unknown_count = state_counts.get('Unknown', 0)
            valid_count = len(df) - unknown_count
            print(f"\n✅ Valid states: {valid_count:,} records")
            print(f"❌ Unknown states: {unknown_count:,} records")
            print(f"📊 Success rate: {(valid_count/len(df)*100):.1f}%")
        
        # Create combined text for RAG
        print("🔄 Creating combined text...")
        def create_combined_text(row):
            parts = []
            for col, value in row.items():
                if pd.notna(value) and value != '' and col not in ['combined_text', 'source_file']:
                    parts.append(f"{col}: {value}")
            return " | ".join(parts)
        
        df['combined_text'] = df.apply(create_combined_text, axis=1)
        
        # Remove duplicates
        print("🔄 Removing duplicates...")
        initial_count = len(df)
        df = df.drop_duplicates(subset=['combined_text'])
        final_count = len(df)
        print(f"📊 Deduplication: {initial_count:,} → {final_count:,} records ({initial_count - final_count:,} duplicates removed)")
        
        # Save the complete dataset
        output_file = "ingris_rag_ready_complete.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Complete dataset saved: {output_file}")
        
        # Also save as the main upload file
        df.to_csv("ingris_rag_ready.csv", index=False)
        print("✅ Updated ingris_rag_ready.csv with complete data")
        
        return df
    else:
        print("❌ No data extracted")
        return None

if __name__ == "__main__":
    df = extract_all_datasets()
    if df is not None:
        print(f"\n🎉 Complete extraction finished!")
        print(f"📈 Total records: {len(df):,}")
        print("🚀 You can now upload the complete dataset!")

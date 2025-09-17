#!/usr/bin/env python3
"""
Extract all INGRIS Excel files into a clean, unified CSV file
This is much faster than processing individual files for RAG
"""

import pandas as pd
import os
import json
import re
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def clean_header(header):
    """Clean and standardize column headers."""
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

def extract_data_from_excel(file_path, year=None, state=None):
    """Extract data from a single Excel file."""
    try:
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
        df['state'] = state if state else 'Unknown'
        
        # Convert to list of dictionaries
        data = df.to_dict('records')
        
        logger.info(f"Extracted {len(data)} rows from {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return []

def process_all_ingris_excel_files():
    """Process all INGRIS Excel files and create a unified CSV."""
    base_path = Path("INGRIS DATASETS/INGRIS DATASETS")
    all_data = []
    
    if not base_path.exists():
        logger.error(f"INGRIS DATASETS directory not found at {base_path}")
        return None
    
    logger.info("🔍 Scanning for Excel files...")
    
    # Process STATE WISE datasets
    state_wise_path = base_path / "STATE WISE"
    if state_wise_path.exists():
        logger.info("📁 Processing state-wise datasets...")
        for file_path in state_wise_path.glob("*.xlsx"):
            year_match = re.search(r'(\d{4})', file_path.name)
            year = int(year_match.group(1)) if year_match else None
            logger.info(f"Processing {file_path.name} (Year: {year})")
            all_data.extend(extract_data_from_excel(file_path, year=year))
    
    # Process year-specific datasets
    year_folders = [f for f in base_path.iterdir() if f.is_dir() and "dataset" in f.name.lower()]
    for year_folder in year_folders:
        year_match = re.search(r'(\d{4})', year_folder.name)
        year = int(year_match.group(1)) if year_match else None
        logger.info(f"Processing year dataset: {year_folder.name}")
        
        for file_path in year_folder.glob("*.xlsx"):
            all_data.extend(extract_data_from_excel(file_path, year=year))
    
    logger.info(f"📊 Total extracted: {len(all_data)} rows")
    return all_data

def create_clean_csv(data, output_file="ingris_clean_data.csv"):
    """Create a clean CSV file from extracted data."""
    if not data:
        logger.error("No data to save")
        return False
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Clean up the data
        df = df.fillna('')
        
        # Remove completely empty rows
        df = df[df.astype(str).ne('').any(axis=1)]
        
        # Save to CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"✅ Clean CSV saved: {output_file}")
        logger.info(f"📊 Rows: {len(df)}, Columns: {len(df.columns)}")
        
        # Show sample of the data
        logger.info("📋 Sample data:")
        print(df.head(3).to_string())
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating CSV: {e}")
        return False

def create_rag_ready_csv(csv_file="ingris_clean_data.csv", output_file="ingris_rag_ready.csv"):
    """Create a RAG-ready CSV with combined text content."""
    try:
        df = pd.read_csv(csv_file)
        
        # Create combined text for RAG
        def create_combined_text(row):
            parts = []
            for col, value in row.items():
                if pd.notna(value) and str(value).strip() != '':
                    parts.append(f"{col}: {value}")
            return " | ".join(parts)
        
        df['combined_text'] = df.apply(create_combined_text, axis=1)
        
        # Save RAG-ready CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"✅ RAG-ready CSV saved: {output_file}")
        logger.info(f"📊 Rows: {len(df)}, Columns: {len(df.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating RAG-ready CSV: {e}")
        return False

def main():
    """Main function to extract Excel files to CSV."""
    print("🚀 Starting INGRIS Excel to CSV extraction...")
    
    # Extract data from Excel files
    data = process_all_ingris_excel_files()
    
    if not data:
        print("❌ No data extracted")
        return
    
    # Create clean CSV
    if create_clean_csv(data):
        print("✅ Clean CSV created successfully")
        
        # Create RAG-ready CSV
        if create_rag_ready_csv():
            print("✅ RAG-ready CSV created successfully")
            print("\n🎉 Extraction complete!")
            print("📁 Files created:")
            print("   • ingris_clean_data.csv - Raw extracted data")
            print("   • ingris_rag_ready.csv - RAG-ready with combined text")
            print("\n💡 You can now use these CSV files for faster processing!")
        else:
            print("❌ Failed to create RAG-ready CSV")
    else:
        print("❌ Failed to create clean CSV")

if __name__ == "__main__":
    main()

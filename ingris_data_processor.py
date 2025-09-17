#!/usr/bin/env python3
"""
INGRIS Data Processor for RAG System
Processes INGRIS (Integrated Groundwater Resource Information System) datasets
and prepares them for vector search and RAG applications.
"""

import os
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class INGRISDataProcessor:
    """Processes INGRIS datasets for RAG applications."""
    
    def __init__(self, data_dir: str = "INGRIS DATASETS/INGRIS DATASETS"):
        self.data_dir = Path(data_dir)
        self.processed_data = []
        self.metadata = {}
        
    def process_all_datasets(self) -> List[Dict[str, Any]]:
        """Process all INGRIS datasets and return structured data."""
        logger.info("Starting INGRIS data processing...")
        
        # Process state-wise datasets first (they have better structure)
        state_wise_dir = self.data_dir / "STATE WISE"
        if state_wise_dir.exists():
            self._process_state_wise_datasets(state_wise_dir)
        
        # Process individual year datasets
        for year_dir in self.data_dir.iterdir():
            if year_dir.is_dir() and year_dir.name != "STATE WISE" and not year_dir.name.startswith("__"):
                self._process_year_dataset(year_dir)
        
        logger.info(f"Processed {len(self.processed_data)} data entries")
        return self.processed_data
    
    def _process_state_wise_datasets(self, state_wise_dir: Path):
        """Process state-wise consolidated datasets."""
        logger.info("Processing state-wise datasets...")
        
        for file_path in state_wise_dir.glob("*.xlsx"):
            if file_path.name.startswith("."):
                continue
                
            year = self._extract_year_from_filename(file_path.name)
            logger.info(f"Processing {file_path.name} (Year: {year})")
            
            try:
                df = pd.read_excel(file_path, header=None)
                processed_data = self._parse_ingris_dataframe(df, year, file_path.name)
                self.processed_data.extend(processed_data)
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
    
    def _process_year_dataset(self, year_dir: Path):
        """Process individual year datasets."""
        logger.info(f"Processing year dataset: {year_dir.name}")
        
        year = self._extract_year_from_filename(year_dir.name)
        
        for file_path in year_dir.glob("*.xlsx"):
            if file_path.name.startswith("."):
                continue
                
            try:
                df = pd.read_excel(file_path, header=None)
                processed_data = self._parse_ingris_dataframe(df, year, file_path.name)
                self.processed_data.extend(processed_data)
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
    
    def _extract_year_from_filename(self, filename: str) -> str:
        """Extract year from filename."""
        year_match = re.search(r'(20\d{2})', filename)
        if year_match:
            return year_match.group(1)
        return "Unknown"
    
    def _parse_ingris_dataframe(self, df: pd.DataFrame, year: str, filename: str) -> List[Dict[str, Any]]:
        """Parse INGRIS dataframe and extract structured data."""
        processed_entries = []
        
        # Find the header row (usually around row 7-9)
        header_row = self._find_header_row(df)
        if header_row is None:
            logger.warning(f"Could not find header row in {filename}")
            return processed_entries
        
        # Extract headers
        headers = self._extract_headers(df, header_row)
        
        # Find data start row
        data_start_row = self._find_data_start_row(df, header_row)
        if data_start_row is None:
            logger.warning(f"Could not find data start row in {filename}")
            return processed_entries
        
        # Extract state name from the report title
        state_name = self._extract_state_name(df)
        
        # Process data rows
        for idx in range(data_start_row, len(df)):
            row = df.iloc[idx]
            if self._is_valid_data_row(row):
                entry = self._create_data_entry(row, headers, year, state_name, filename)
                if entry:
                    processed_entries.append(entry)
        
        return processed_entries
    
    def _find_header_row(self, df: pd.DataFrame) -> Optional[int]:
        """Find the row containing column headers."""
        for idx in range(min(15, len(df))):
            row = df.iloc[idx]
            if any(str(cell).strip().lower() in ['s.no', 'state', 'district', 'assessment unit'] for cell in row if pd.notna(cell)):
                return idx
        return None
    
    def _extract_headers(self, df: pd.DataFrame, header_row: int) -> List[str]:
        """Extract and clean column headers."""
        headers = []
        for col in df.columns:
            cell_value = str(df.iloc[header_row, col]).strip()
            if cell_value and cell_value != 'nan':
                headers.append(cell_value)
            else:
                headers.append(f"Column_{col}")
        return headers
    
    def _find_data_start_row(self, df: pd.DataFrame, header_row: int) -> Optional[int]:
        """Find the row where actual data starts."""
        for idx in range(header_row + 1, min(header_row + 10, len(df))):
            row = df.iloc[idx]
            if self._is_valid_data_row(row):
                return idx
        return None
    
    def _is_valid_data_row(self, row: pd.Series) -> bool:
        """Check if a row contains valid data."""
        # Check if row has non-null values and looks like data
        non_null_count = row.notna().sum()
        if non_null_count < 3:  # At least 3 non-null values
            return False
        
        # Check if first few columns look like data (not headers)
        first_cell = str(row.iloc[0]).strip()
        if first_cell.lower() in ['s.no', 'state', 'district', 'assessment unit', 'nan', '']:
            return False
        
        return True
    
    def _extract_state_name(self, df: pd.DataFrame) -> str:
        """Extract state name from the report title."""
        for idx in range(min(5, len(df))):
            row = df.iloc[idx]
            for cell in row:
                if pd.notna(cell):
                    cell_str = str(cell).strip()
                    if "report for :" in cell_str.lower():
                        # Extract state name from "Report for : STATE NAME for year YYYY-YYYY"
                        match = re.search(r'report for\s*:\s*([^f]+?)\s+for year', cell_str, re.IGNORECASE)
                        if match:
                            return match.group(1).strip()
        return "Unknown State"
    
    def _create_data_entry(self, row: pd.Series, headers: List[str], year: str, state_name: str, filename: str) -> Optional[Dict[str, Any]]:
        """Create a structured data entry from a row."""
        try:
            entry = {
                'year': year,
                'state': state_name,
                'filename': filename,
                'raw_data': {},
                'text_content': '',
                'metadata': {}
            }
            
            # Map data to headers
            for i, header in enumerate(headers):
                if i < len(row):
                    value = row.iloc[i]
                    if pd.notna(value):
                        entry['raw_data'][header] = str(value).strip()
            
            # Extract key fields
            entry['district'] = entry['raw_data'].get('DISTRICT', 'Unknown')
            entry['assessment_unit'] = entry['raw_data'].get('ASSESSMENT UNIT', 'Unknown')
            entry['s_no'] = entry['raw_data'].get('S.No', 'Unknown')
            
            # Create searchable text content
            text_parts = []
            text_parts.append(f"State: {state_name}")
            text_parts.append(f"District: {entry['district']}")
            text_parts.append(f"Assessment Unit: {entry['assessment_unit']}")
            text_parts.append(f"Year: {year}")
            
            # Add key groundwater parameters
            key_params = [
                'Rainfall (mm)', 'Ground Water Recharge (ham)', 'Annual Ground water Recharge (ham)',
                'Annual Extractable Ground water Resource (ham)', 'Ground Water Extraction for all uses (ha.m)',
                'Stage of Ground Water Extraction (%)', 'Net Annual Ground Water Availability for Future Use (ham)',
                'Total Ground Water Availability in the area (ham)'
            ]
            
            for param in key_params:
                if param in entry['raw_data']:
                    value = entry['raw_data'][param]
                    if value and str(value).strip() != 'nan':
                        text_parts.append(f"{param}: {value}")
            
            entry['text_content'] = " | ".join(text_parts)
            
            # Create metadata for filtering
            entry['metadata'] = {
                'state': state_name,
                'district': entry['district'],
                'year': year,
                'assessment_unit': entry['assessment_unit'],
                'data_type': 'ingris_groundwater_assessment'
            }
            
            return entry
            
        except Exception as e:
            logger.error(f"Error creating data entry: {str(e)}")
            return None
    
    def save_processed_data(self, output_file: str = "ingris_processed_data.json"):
        """Save processed data to JSON file."""
        output_path = Path(output_file)
        
        data_to_save = {
            'metadata': {
                'total_entries': len(self.processed_data),
                'processed_at': pd.Timestamp.now().isoformat(),
                'data_source': 'INGRIS DATASETS'
            },
            'data': self.processed_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed data to {output_path}")
        return output_path
    
    def load_processed_data(self, input_file: str = "ingris_processed_data.json") -> List[Dict[str, Any]]:
        """Load processed data from JSON file."""
        input_path = Path(input_file)
        
        if not input_path.exists():
            logger.warning(f"Processed data file {input_path} not found")
            return []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.processed_data = data.get('data', [])
        logger.info(f"Loaded {len(self.processed_data)} entries from {input_path}")
        return self.processed_data

def main():
    """Main function to process INGRIS datasets."""
    processor = INGRISDataProcessor()
    
    # Process all datasets
    processed_data = processor.process_all_datasets()
    
    # Save processed data
    output_file = processor.save_processed_data()
    
    print(f"✅ Processed {len(processed_data)} INGRIS data entries")
    print(f"📁 Saved to: {output_file}")
    
    # Show sample data
    if processed_data:
        print("\n📊 Sample processed entry:")
        sample = processed_data[0]
        print(f"State: {sample['state']}")
        print(f"District: {sample['district']}")
        print(f"Year: {sample['year']}")
        print(f"Text Content: {sample['text_content'][:200]}...")

if __name__ == "__main__":
    main()

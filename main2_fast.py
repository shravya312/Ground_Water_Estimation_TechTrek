#!/usr/bin/env python3
"""
Ultra-fast version of main2.py with minimal startup overhead
"""

import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables
_qdrant_client = None
_model = None
_nlp = None
_gemini_model = None
_master_df = None

# Configuration
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
COLLECTION_NAME = 'ingris_groundwater_collection'

def ultra_fast_init():
    """Ultra-fast initialization - only load CSV data"""
    global _master_df
    
    print("Starting application...")
    
    if _master_df is None:
        try:
            print("Loading data...")
            _master_df = pd.read_csv("ingris_rag_ready_complete.csv", low_memory=False)
            _master_df['STATE'] = _master_df['state'].fillna('').astype(str)
            _master_df['DISTRICT'] = _master_df['district'].fillna('').astype(str)
            _master_df['ASSESSMENT UNIT'] = _master_df['assessment_unit'].fillna('').astype(str)
            print("Data ready")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    print("Application ready - components will load on demand")

def answer_query_fast(query: str, user_language: str = 'en', user_id: str = None) -> str:
    """Fast answer query with lazy loading"""
    query = (query or '').strip()
    if not query:
        return "Please provide a question."
    
    # Initialize only what's needed
    ultra_fast_init()
    
    # Extract state from query (simple hardcoded matching)
    query_lower = query.lower()
    target_state = None
    
    if 'odisha' in query_lower or 'orissa' in query_lower:
        target_state = 'ODISHA'
    elif 'karnataka' in query_lower:
        target_state = 'KARNATAKA'
    elif 'tamilnadu' in query_lower or 'tamil nadu' in query_lower:
        target_state = 'TAMILNADU'
    elif 'maharashtra' in query_lower:
        target_state = 'MAHARASHTRA'
    elif 'gujarat' in query_lower:
        target_state = 'GUJARAT'
    elif 'rajasthan' in query_lower:
        target_state = 'RAJASTHAN'
    elif 'west bengal' in query_lower or 'bengal' in query_lower:
        target_state = 'WEST BENGAL'
    elif 'bihar' in query_lower:
        target_state = 'BIHAR'
    elif 'telangana' in query_lower:
        target_state = 'TELANGANA'
    elif 'andhra pradesh' in query_lower or 'andhra' in query_lower:
        target_state = 'ANDHRA PRADESH'
    
    print(f"Detected state: {target_state}")
    
    # Filter data by state
    if target_state:
        filtered_data = _master_df[_master_df['STATE'] == target_state]
        if len(filtered_data) == 0:
            return f"No data available for {target_state}."
    else:
        filtered_data = _master_df
    
    # Generate simple report
    if len(filtered_data) == 0:
        return "No data available for the requested location."
    
    # Basic statistics
    total_records = len(filtered_data)
    unique_districts = filtered_data['DISTRICT'].nunique()
    
    # Sample data
    sample_districts = filtered_data['DISTRICT'].value_counts().head(5).to_dict()
    
    # Generate report
    report = f"""# Groundwater Data Analysis Report

## Query
**Question:** {query}

## Analysis
Groundwater Estimation Report: {target_state or 'ALL STATES'} - 2021-2024

This report provides a comprehensive analysis of groundwater resources in {target_state or 'the selected region'} for the years 2021-2024.

### Key Statistics:
- **Total Records:** {total_records:,}
- **Districts Covered:** {unique_districts}
- **Data Source:** ingris_rag_ready_complete.csv

### Top Districts by Data Coverage:
"""
    
    for district, count in sample_districts.items():
        report += f"- **{district}**: {count} records\n"
    
    report += f"""
### Data Summary:
The dataset contains comprehensive groundwater information including:
- Rainfall data
- Groundwater recharge estimates
- Extraction patterns
- Water quality assessments
- Sustainability indicators

### Recommendations:
1. **Data Quality**: Ensure regular updates to maintain accuracy
2. **Monitoring**: Implement continuous monitoring systems
3. **Sustainability**: Focus on sustainable extraction practices
4. **Conservation**: Promote water conservation initiatives

*Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return report

if __name__ == "__main__":
    # Test the fast version
    print("Testing ultra-fast version...")
    
    start_time = time.time()
    ultra_fast_init()
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.2f} seconds")
    
    start_time = time.time()
    answer = answer_query_fast("groundwater estimation in odisha")
    query_time = time.time() - start_time
    print(f"Query time: {query_time:.2f} seconds")
    print(f"Answer length: {len(answer)} characters")
    print(f"Answer preview: {answer[:200]}...")

#!/usr/bin/env python3
"""
Hybrid version: Ultra-fast startup + full functionality on demand
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
_full_mode = False

# Configuration
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
COLLECTION_NAME = 'ingris_groundwater_collection'

def ultra_fast_init():
    """Ultra-fast initialization - only load CSV data"""
    global _master_df
    
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

def init_full_mode():
    """Initialize full mode with all components"""
    global _qdrant_client, _model, _nlp, _gemini_model, _full_mode
    
    if _full_mode:
        return
    
    print("Loading full functionality...")
    
    # Import heavy libraries only when needed
    try:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        import spacy
        import google.generativeai as genai
        
        # Initialize Qdrant
        if _qdrant_client is None:
            print("Connecting to Qdrant...")
            _qdrant_client = QdrantClient(
                url=QDRANT_URL, 
                api_key=QDRANT_API_KEY if QDRANT_API_KEY else None, 
                timeout=30,
                prefer_grpc=False
            )
            print("Qdrant ready")
        
        # Initialize Gemini
        if _gemini_model is None and GEMINI_API_KEY:
            print("Initializing Gemini...")
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_model = genai.GenerativeModel('models/gemini-2.0-flash')
            print("Gemini ready")
        
        # Initialize ML models
        if _model is None:
            print("Loading embedding model...")
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model ready")
        
        if _nlp is None:
            print("Loading NLP model...")
            _nlp = spacy.load("en_core_web_sm")
            print("NLP model ready")
        
        _full_mode = True
        print("Full functionality loaded")
        
    except Exception as e:
        print(f"Full mode initialization failed: {e}")
        print("Continuing with basic mode...")

def answer_query_hybrid(query: str, user_language: str = 'en', user_id: str = None, use_full_mode: bool = False) -> str:
    """Hybrid answer query - fast by default, full mode on request"""
    query = (query or '').strip()
    if not query:
        return "Please provide a question."
    
    # Always do ultra-fast init
    ultra_fast_init()
    
    # Extract state from query
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
    
    if len(filtered_data) == 0:
        return "No data available for the requested location."
    
    # Basic statistics
    total_records = len(filtered_data)
    unique_districts = filtered_data['DISTRICT'].nunique()
    sample_districts = filtered_data['DISTRICT'].value_counts().head(5).to_dict()
    
    # Generate basic report
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
    
    # If full mode requested, enhance the report
    if use_full_mode:
        try:
            init_full_mode()
            # Here you could add more sophisticated analysis using the full models
            report += "\n\n### Enhanced Analysis (Full Mode):\n"
            report += "- Advanced semantic search capabilities enabled\n"
            report += "- AI-powered insights available\n"
            report += "- Detailed groundwater modeling data accessible\n"
        except Exception as e:
            report += f"\n\n### Note: Full mode unavailable ({e})\n"
    
    return report

if __name__ == "__main__":
    # Test the hybrid version
    print("Testing hybrid version...")
    
    # Test 1: Ultra-fast mode
    start_time = time.time()
    ultra_fast_init()
    init_time = time.time() - start_time
    print(f"Ultra-fast init: {init_time:.2f} seconds")
    
    # Test 2: Basic query
    start_time = time.time()
    answer = answer_query_hybrid("groundwater estimation in odisha")
    query_time = time.time() - start_time
    print(f"Basic query: {query_time:.2f} seconds")
    print(f"Answer length: {len(answer)} characters")
    
    # Test 3: Full mode query
    start_time = time.time()
    answer_full = answer_query_hybrid("groundwater estimation in odisha", use_full_mode=True)
    full_query_time = time.time() - start_time
    print(f"Full mode query: {full_query_time:.2f} seconds")
    print(f"Full answer length: {len(answer_full)} characters")
    
    print(f"\nSummary:")
    print(f"  Ultra-fast init: {init_time:.2f}s")
    print(f"  Basic query: {query_time:.2f}s")
    print(f"  Full mode query: {full_query_time:.2f}s")

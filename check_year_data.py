#!/usr/bin/env python3
"""
Check year data in Qdrant to understand the year filtering issue
"""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

def check_year_data_in_qdrant():
    """Check what year data is available in Qdrant"""
    print("Checking Year Data in Qdrant")
    print("=" * 40)
    
    try:
        # Initialize Qdrant client
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY') if os.getenv('QDRANT_API_KEY') else None,
            timeout=30
        )
        print("Qdrant client connected")
        
        # Use correct 768-dimensional model
        model = SentenceTransformer('all-mpnet-base-v2')
        print(f"Using model: all-mpnet-base-v2 (768 dimensions)")
        
        # Search for Haryana data without year filter
        print("\n1. Searching Haryana without year filter...")
        query_text = "groundwater haryana"
        query_vector = model.encode(query_text).tolist()
        
        results = client.query_points(
            collection_name="ingris_groundwater_collection",
            query=query_vector,
            limit=10,
            with_payload=True
        )
        
        print(f"Results found: {len(results.points)}")
        
        if results.points:
            print("Sample records:")
            years_found = set()
            for i, point in enumerate(results.points[:5]):
                payload = point.payload
                state = payload.get('STATE', 'Unknown')
                district = payload.get('DISTRICT', 'Unknown')
                
                # Check different possible year fields
                year_fields = ['year', 'Year', 'YEAR', 'Assessment_Year', 'assessment_year', 'assessment_year']
                year_value = None
                for field in year_fields:
                    if field in payload:
                        year_value = payload[field]
                        years_found.add(str(year_value))
                        break
                
                print(f"  {i+1}. {state} - {district} - Year: {year_value}")
                print(f"     All fields: {list(payload.keys())}")
            
            print(f"\nUnique years found: {sorted(years_found)}")
        
        # Test year filtering
        print(f"\n2. Testing year filtering for 2024...")
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Try different year field names
        year_fields_to_test = ['Assessment_Year', 'year', 'Year', 'YEAR', 'assessment_year']
        
        for year_field in year_fields_to_test:
            try:
                print(f"  Testing field: {year_field}")
                
                year_filter = Filter(
                    must=[
                        FieldCondition(
                            key="STATE",
                            match=MatchValue(value="HARYANA")
                        ),
                        FieldCondition(
                            key=year_field,
                            match=MatchValue(value=2024)
                        )
                    ]
                )
                
                year_results = client.query_points(
                    collection_name="ingris_groundwater_collection",
                    query=query_vector,
                    query_filter=year_filter,
                    limit=5,
                    with_payload=True
                )
                
                print(f"    Results with {year_field}=2024: {len(year_results.points)}")
                
                if year_results.points:
                    for point in year_results.points[:2]:
                        payload = point.payload
                        state = payload.get('STATE', 'Unknown')
                        district = payload.get('DISTRICT', 'Unknown')
                        year_val = payload.get(year_field, 'Unknown')
                        print(f"      Sample: {state} - {district} - {year_field}: {year_val}")
                
            except Exception as e:
                print(f"    Error with {year_field}: {e}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_year_data_in_qdrant()

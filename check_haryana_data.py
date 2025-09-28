#!/usr/bin/env python3
"""
Check Haryana data specifically in Qdrant
"""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

def check_haryana_data():
    """Check Haryana data in Qdrant"""
    print("Checking Haryana Data in Qdrant")
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
        
        # Search for Haryana data without any filters
        print("\n1. Searching for Haryana data without filters...")
        query_text = "haryana groundwater"
        query_vector = model.encode(query_text).tolist()
        
        results = client.query_points(
            collection_name="ingris_groundwater_collection",
            query=query_vector,
            limit=20,
            with_payload=True
        )
        
        print(f"Results found: {len(results.points)}")
        
        haryana_years = set()
        haryana_districts = set()
        
        if results.points:
            print("Haryana records found:")
            for i, point in enumerate(results.points):
                payload = point.payload
                state = payload.get('STATE', 'Unknown')
                district = payload.get('DISTRICT', 'Unknown')
                year = payload.get('Assessment_Year', 'Unknown')
                
                if 'HARYANA' in state.upper():
                    haryana_years.add(str(year))
                    haryana_districts.add(district)
                    print(f"  {i+1}. {state} - {district} - Year: {year}")
        
        print(f"\nHaryana years available: {sorted(haryana_years)}")
        print(f"Haryana districts: {sorted(haryana_districts)}")
        
        # Test Haryana with state filter only
        print(f"\n2. Testing Haryana with state filter only...")
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        state_filter = Filter(
            must=[
                FieldCondition(
                    key="STATE",
                    match=MatchValue(value="HARYANA")
                )
            ]
        )
        
        state_results = client.query_points(
            collection_name="ingris_groundwater_collection",
            query=query_vector,
            query_filter=state_filter,
            limit=10,
            with_payload=True
        )
        
        print(f"Results with STATE=HARYANA: {len(state_results.points)}")
        
        if state_results.points:
            print("Haryana records with state filter:")
            for i, point in enumerate(state_results.points[:5]):
                payload = point.payload
                state = payload.get('STATE', 'Unknown')
                district = payload.get('DISTRICT', 'Unknown')
                year = payload.get('Assessment_Year', 'Unknown')
                print(f"  {i+1}. {state} - {district} - Year: {year}")
        
        # Test Haryana with year filter
        print(f"\n3. Testing Haryana with year filter for 2024...")
        
        year_filter = Filter(
            must=[
                FieldCondition(
                    key="STATE",
                    match=MatchValue(value="HARYANA")
                ),
                FieldCondition(
                    key="Assessment_Year",
                    match=MatchValue(value=2024)
                )
            ]
        )
        
        year_results = client.query_points(
            collection_name="ingris_groundwater_collection",
            query=query_vector,
            query_filter=year_filter,
            limit=10,
            with_payload=True
        )
        
        print(f"Results with STATE=HARYANA AND Assessment_Year=2024: {len(year_results.points)}")
        
        if year_results.points:
            print("Haryana 2024 records:")
            for i, point in enumerate(year_results.points):
                payload = point.payload
                state = payload.get('STATE', 'Unknown')
                district = payload.get('DISTRICT', 'Unknown')
                year = payload.get('Assessment_Year', 'Unknown')
                print(f"  {i+1}. {state} - {district} - Year: {year}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_haryana_data()

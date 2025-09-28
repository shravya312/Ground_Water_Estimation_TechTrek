#!/usr/bin/env python3
"""
Direct verification of Odisha data in Qdrant
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

load_dotenv()

def verify_odisha_direct():
    """Direct verification of Odisha data"""
    print("Direct Odisha Verification")
    print("=" * 30)
    
    try:
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            timeout=30
        )
        
        # Test 1: Direct query for ODISHA
        print("1. Direct query for ODISHA...")
        results = client.query_points(
            collection_name='ingris_groundwater_collection',
            query=[0.1] * 768,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="STATE",
                        match=MatchValue(value="ODISHA")
                    )
                ]
            ),
            limit=10,
            with_payload=True
        )
        
        print(f"   Found {len(results.points)} results")
        if results.points:
            for i, result in enumerate(results.points[:5]):
                district = result.payload.get('DISTRICT', 'N/A')
                year = result.payload.get('Assessment_Year', 'N/A')
                print(f"   {i+1}. District: {district}, Year: {year}")
        else:
            print("   No results found")
        
        # Test 2: Check total collection size
        print("\n2. Collection info...")
        collection_info = client.get_collection('ingris_groundwater_collection')
        print(f"   Total points: {collection_info.points_count:,}")
        print(f"   Vector size: {collection_info.config.params.vectors.size}")
        
        # Test 3: Sample without filter
        print("\n3. Sample without filter...")
        sample = client.query_points(
            collection_name='ingris_groundwater_collection',
            query=[0.1] * 768,
            limit=20,
            with_payload=True
        )
        
        states = set()
        for result in sample.points:
            state = result.payload.get('STATE', 'N/A')
            if state and state != 'N/A':
                states.add(state)
        
        print(f"   States in sample: {sorted(states)}")
        print(f"   ODISHA in sample: {'ODISHA' in states}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_odisha_direct()

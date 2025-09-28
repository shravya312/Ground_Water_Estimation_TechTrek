#!/usr/bin/env python3
"""
Ultra fast states check using scroll
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def ultra_fast_check():
    """Ultra fast check using scroll"""
    print("Ultra Fast States Check")
    print("=" * 25)
    
    try:
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            timeout=20
        )
        
        # Get collection info
        collection_info = client.get_collection('ingris_groundwater_collection')
        print(f"Total points: {collection_info.points_count:,}")
        
        # Get first 1000 records using scroll
        print("Getting first 1000 records...")
        results = client.scroll(
            collection_name='ingris_groundwater_collection',
            limit=1000,
            with_payload=True
        )
        
        if results[0]:
            states = set()
            for result in results[0]:
                state = result.payload.get('STATE', 'N/A')
                if state and state != 'N/A':
                    states.add(state)
            
            print(f"Found {len(states)} unique states in first 1000 records:")
            print("-" * 40)
            
            for i, state in enumerate(sorted(states), 1):
                print(f"{i:2d}. {state}")
            
            # Check Odisha
            odisha_found = any('ODISHA' in s.upper() or 'ORISSA' in s.upper() for s in states)
            print(f"\nOdisha found: {'Yes' if odisha_found else 'No'}")
            
        else:
            print("No data found")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    ultra_fast_check()

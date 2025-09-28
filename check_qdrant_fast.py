#!/usr/bin/env python3
"""
Fast Qdrant check - get basic info and sample states
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from collections import Counter

load_dotenv()

def check_qdrant_fast():
    """Fast check of Qdrant data"""
    print("Fast Qdrant Data Check")
    print("=" * 30)
    
    try:
        # Connect to Qdrant with shorter timeout
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            timeout=30
        )
        print("Connected to Qdrant")
        
        # Get collection info
        collection_info = client.get_collection('ingris_groundwater_collection')
        total_points = collection_info.points_count
        vector_size = collection_info.config.params.vectors.size
        
        print(f"Total points: {total_points:,}")
        print(f"Vector size: {vector_size}")
        
        # Get just a small sample to check states
        print("\nGetting sample of states...")
        results = client.scroll(
            collection_name='ingris_groundwater_collection',
            limit=100,  # Small sample
            with_payload=True
        )
        
        if results[0]:
            states = []
            for result in results[0]:
                state = result.payload.get('STATE', 'N/A')
                if state and state != 'N/A':
                    states.append(state)
            
            state_counts = Counter(states)
            print(f"\nStates found in sample of {len(states)} records:")
            print("-" * 40)
            
            for state, count in state_counts.most_common():
                print(f"{state}: {count} records")
            
            # Check for Odisha specifically
            print(f"\nChecking for Odisha:")
            odisha_found = False
            for state in states:
                if 'ODISHA' in state.upper() or 'ORISSA' in state.upper():
                    print(f"Found: {state}")
                    odisha_found = True
            
            if not odisha_found:
                print("No Odisha data found in sample")
            
            print(f"\nUnique states in sample: {len(state_counts)}")
            
        else:
            print("No data found in collection")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_qdrant_fast()

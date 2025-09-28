#!/usr/bin/env python3
"""
Simple script to check all states in Qdrant collection
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from collections import Counter

load_dotenv()

def check_states_simple():
    """Check all states in Qdrant collection"""
    print("Checking States in Qdrant Collection")
    print("=" * 50)
    
    try:
        # Connect to Qdrant
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        print("Connected to Qdrant")
        
        # Get collection info
        collection_info = client.get_collection('ingris_groundwater_collection')
        print(f"Total points in collection: {collection_info.points_count}")
        print(f"Vector size: {collection_info.config.params.vectors.size}")
        
        # Get ALL data in batches to find all states
        print("\nScanning all data for states...")
        all_states = []
        batch_size = 1000
        offset = None
        total_scanned = 0
        
        while True:
            # Scroll through all data
            results = client.scroll(
                collection_name='ingris_groundwater_collection',
                limit=batch_size,
                offset=offset,
                with_payload=True
            )
            
            if not results[0]:  # No more data
                break
                
            # Extract states from this batch
            for result in results[0]:
                state = result.payload.get('STATE', 'N/A')
                if state and state != 'N/A':
                    all_states.append(state)
            
            total_scanned += len(results[0])
            offset = results[1]  # Next offset
            
            print(f"Scanned {total_scanned} records...", end='\r')
            
            if len(results[0]) < batch_size:  # Last batch
                break
        
        print(f"\n\nTotal records scanned: {total_scanned}")
        
        # Analyze states
        state_counts = Counter(all_states)
        print(f"\nState Analysis (Total: {len(all_states)} records):")
        print("=" * 40)
        
        for state, count in state_counts.most_common():
            print(f"{state}: {count:,} records")
        
        # Check specifically for Odisha
        print(f"\nChecking for Odisha specifically:")
        odisha_variations = ['ODISHA', 'ORISSA', 'odisha', 'orissa']
        found_odisha = False
        for variation in odisha_variations:
            if variation in state_counts:
                print(f"Found {variation}: {state_counts[variation]} records")
                found_odisha = True
        
        if not found_odisha:
            print("No Odisha data found in any variation")
        
        # Show unique states count
        unique_states = len(state_counts)
        print(f"\nTotal unique states: {unique_states}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_states_simple()

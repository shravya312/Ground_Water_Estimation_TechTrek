#!/usr/bin/env python3
"""
Comprehensive check of all states in Qdrant collection
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from collections import Counter
import time

load_dotenv()

def check_all_states_comprehensive():
    """Check all states in Qdrant using multiple approaches"""
    print("Comprehensive Qdrant States Check")
    print("=" * 40)
    
    try:
        # Connect to Qdrant
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            timeout=60
        )
        print("Connected to Qdrant")
        
        # Get collection info
        collection_info = client.get_collection('ingris_groundwater_collection')
        total_points = collection_info.points_count
        print(f"Total points: {total_points:,}")
        
        all_states = set()
        
        # Method 1: Use scroll with smaller batches
        print("\nMethod 1: Scrolling through data in small batches...")
        batch_size = 50
        offset = None
        total_scanned = 0
        
        try:
            while total_scanned < 20000:  # Limit to 20k records to avoid timeout
                results = client.scroll(
                    collection_name='ingris_groundwater_collection',
                    limit=batch_size,
                    offset=offset,
                    with_payload=True
                )
                
                if not results[0]:
                    break
                    
                # Extract states
                for result in results[0]:
                    state = result.payload.get('STATE', 'N/A')
                    if state and state != 'N/A':
                        all_states.add(state)
                
                total_scanned += len(results[0])
                offset = results[1]
                print(f"Scanned {total_scanned:,} records, found {len(all_states)} unique states...", end='\r')
                
                if len(results[0]) < batch_size:
                    break
                    
                time.sleep(0.1)  # Small delay to avoid overwhelming
                
        except Exception as e:
            print(f"\nScroll method failed: {e}")
        
        print(f"\nMethod 1 complete: {len(all_states)} unique states from {total_scanned:,} records")
        
        # Method 2: Use search with different dummy vectors
        print("\nMethod 2: Using search with different vectors...")
        search_states = set()
        
        # Try different dummy vectors to get different samples
        dummy_vectors = [
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768,
            [0.4] * 768,
            [0.5] * 768,
            [0.6] * 768,
            [0.7] * 768,
            [0.8] * 768,
            [0.9] * 768,
            [1.0] * 768
        ]
        
        for i, dummy_vector in enumerate(dummy_vectors):
            try:
                results = client.query_points(
                    collection_name='ingris_groundwater_collection',
                    query=dummy_vector,
                    limit=100,
                    with_payload=True
                )
                
                if results.points:
                    for result in results.points:
                        state = result.payload.get('STATE', 'N/A')
                        if state and state != 'N/A':
                            search_states.add(state)
                    
                    print(f"Search {i+1}/10: Found {len(search_states)} unique states...", end='\r')
                    
            except Exception as e:
                print(f"\nSearch {i+1} failed: {e}")
                continue
        
        print(f"\nMethod 2 complete: {len(search_states)} unique states")
        
        # Combine results
        all_states.update(search_states)
        
        print(f"\n" + "="*50)
        print(f"FINAL RESULTS")
        print(f"="*50)
        print(f"Total unique states found: {len(all_states)}")
        print(f"Records scanned: {total_scanned:,}")
        
        if all_states:
            # Sort states alphabetically
            sorted_states = sorted(all_states)
            print(f"\nAll states found:")
            print("-" * 30)
            for i, state in enumerate(sorted_states, 1):
                print(f"{i:2d}. {state}")
            
            # Check for Odisha
            odisha_variations = ['ODISHA', 'ORISSA', 'odisha', 'orissa']
            odisha_found = False
            for state in all_states:
                if any(var in state.upper() for var in odisha_variations):
                    print(f"\n✅ Odisha found: {state}")
                    odisha_found = True
                    break
            
            if not odisha_found:
                print(f"\n❌ No Odisha data found")
        
        return all_states
        
    except Exception as e:
        print(f"Error: {e}")
        return set()

if __name__ == "__main__":
    states = check_all_states_comprehensive()

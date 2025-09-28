#!/usr/bin/env python3
"""
Final Qdrant states check using search approach
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from collections import Counter
import random

load_dotenv()

def check_qdrant_states_final():
    """Check all states in Qdrant using search approach"""
    print("Qdrant States Check - Final")
    print("=" * 35)
    
    try:
        # Connect to Qdrant
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            timeout=30
        )
        print("Connected to Qdrant")
        
        # Get collection info
        collection_info = client.get_collection('ingris_groundwater_collection')
        total_points = collection_info.points_count
        print(f"Total points: {total_points:,}")
        
        # Use search with dummy vector to get random samples
        print("\nGetting random samples to identify states...")
        
        # Create a dummy vector (768 dimensions for cosine similarity)
        dummy_vector = [0.1] * 768
        
        all_states = []
        samples_needed = 5000  # Sample 5000 records
        batch_size = 100
        
        for i in range(0, samples_needed, batch_size):
            try:
                # Search with dummy vector to get random results
                results = client.search(
                    collection_name='ingris_groundwater_collection',
                    query_vector=dummy_vector,
                    limit=batch_size,
                    with_payload=True,
                    score_threshold=0.0  # Accept any score
                )
                
                if not results:
                    break
                    
                # Extract states
                for result in results:
                    state = result.payload.get('STATE', 'N/A')
                    if state and state != 'N/A':
                        all_states.append(state)
                
                print(f"Sampled {len(all_states)} records...", end='\r')
                
            except Exception as e:
                print(f"\nError in batch {i}: {e}")
                break
        
        print(f"\n\nTotal records sampled: {len(all_states)}")
        
        if all_states:
            # Analyze states
            state_counts = Counter(all_states)
            print(f"\nStates found in sample:")
            print("-" * 40)
            
            for state, count in state_counts.most_common():
                print(f"{state}: {count} records")
            
            # Check for Odisha specifically
            print(f"\nChecking for Odisha:")
            odisha_found = False
            for state in state_counts.keys():
                if 'ODISHA' in state.upper() or 'ORISSA' in state.upper():
                    print(f"Found: {state} ({state_counts[state]} records)")
                    odisha_found = True
            
            if not odisha_found:
                print("No Odisha data found")
            
            print(f"\nUnique states found: {len(state_counts)}")
            
            # Estimate total per state
            if len(all_states) > 0:
                print(f"\nEstimated total records per state:")
                print("-" * 40)
                for state, count in state_counts.most_common(10):  # Top 10
                    estimated_total = int((count / len(all_states)) * total_points)
                    print(f"{state}: ~{estimated_total:,} records")
        else:
            print("No states found in sample")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_qdrant_states_final()

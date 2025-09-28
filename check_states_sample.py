#!/usr/bin/env python3
"""
Efficient script to check states in Qdrant collection using sampling
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from collections import Counter
import random

load_dotenv()

def check_states_sample():
    """Check states in Qdrant collection using sampling"""
    print("Checking States in Qdrant Collection (Sampling Method)")
    print("=" * 60)
    
    try:
        # Connect to Qdrant
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        print("Connected to Qdrant")
        
        # Get collection info
        collection_info = client.get_collection('ingris_groundwater_collection')
        total_points = collection_info.points_count
        vector_size = collection_info.config.params.vectors.size
        
        print(f"Total points in collection: {total_points:,}")
        print(f"Vector size: {vector_size}")
        
        # Sample data to get states (sample 10,000 records)
        sample_size = min(10000, total_points)
        print(f"\nSampling {sample_size:,} records to identify states...")
        
        all_states = []
        batch_size = 1000
        samples_taken = 0
        
        # Take samples from different parts of the collection
        for i in range(0, sample_size, batch_size):
            current_batch_size = min(batch_size, sample_size - samples_taken)
            
            # Get a random sample
            results = client.scroll(
                collection_name='ingris_groundwater_collection',
                limit=current_batch_size,
                with_payload=True
            )
            
            if not results[0]:
                break
                
            # Extract states from this batch
            for result in results[0]:
                state = result.payload.get('STATE', 'N/A')
                if state and state != 'N/A':
                    all_states.append(state)
            
            samples_taken += len(results[0])
            print(f"Sampled {samples_taken:,} records...", end='\r')
            
            if len(results[0]) < current_batch_size:
                break
        
        print(f"\n\nTotal records sampled: {samples_taken:,}")
        
        # Analyze states
        state_counts = Counter(all_states)
        print(f"\nState Analysis (from {len(all_states)} sampled records):")
        print("=" * 50)
        
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
        print(f"\nTotal unique states found: {unique_states}")
        
        # Estimate total records per state
        if all_states:
            print(f"\nEstimated total records per state (based on sample):")
            print("=" * 50)
            for state, count in state_counts.most_common():
                estimated_total = int((count / len(all_states)) * total_points)
                print(f"{state}: ~{estimated_total:,} records")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_states_sample()

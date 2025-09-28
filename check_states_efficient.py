#!/usr/bin/env python3
"""
Efficiently check all states in Qdrant using sampling
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from collections import Counter
import random

load_dotenv()

def check_states_efficient():
    """Check all states in Qdrant using efficient sampling"""
    print("Efficiently Checking States in Qdrant Collection")
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
        print(f"Total points in collection: {total_points}")
        
        # Sample data from different parts of the collection
        print("\nSampling data from different parts of collection...")
        
        all_states = []
        sample_size = 1000  # Sample 1000 records from different parts
        
        # Sample from beginning
        print("Sampling from beginning...")
        results = client.scroll(
            collection_name='ingris_groundwater_collection',
            limit=sample_size,
            with_payload=True
        )
        
        for result in results[0]:
            state = result.payload.get('STATE', 'N/A')
            if state and state != 'N/A':
                all_states.append(state)
        
        # Sample from middle (using random offset)
        print("Sampling from middle...")
        middle_offset = total_points // 2
        results = client.scroll(
            collection_name='ingris_groundwater_collection',
            limit=sample_size,
            offset=middle_offset,
            with_payload=True
        )
        
        for result in results[0]:
            state = result.payload.get('STATE', 'N/A')
            if state and state != 'N/A':
                all_states.append(state)
        
        # Sample from end
        print("Sampling from end...")
        end_offset = total_points - sample_size
        if end_offset > 0:
            results = client.scroll(
                collection_name='ingris_groundwater_collection',
                limit=sample_size,
                offset=end_offset,
                with_payload=True
            )
            
            for result in results[0]:
                state = result.payload.get('STATE', 'N/A')
                if state and state != 'N/A':
                    all_states.append(state)
        
        # Count states
        state_counts = Counter(all_states)
        unique_states = len(state_counts)
        
        print(f"\nFrom {len(all_states)} sampled records, found {unique_states} unique states:")
        print("=" * 50)
        
        # Sort by count (descending)
        for state, count in state_counts.most_common():
            print(f"{state:25} : {count:6} records")
        
        # Check Karnataka specifically
        karnataka_count = state_counts.get('KARNATAKA', 0)
        print(f"\nKarnataka records in sample: {karnataka_count}")
        
        # Estimate total Karnataka records
        if karnataka_count > 0:
            estimated_karnataka = (karnataka_count / len(all_states)) * total_points
            print(f"Estimated total Karnataka records: {estimated_karnataka:.0f}")
        
        # Check if we have a good sample
        if unique_states >= 30:  # Should have most states
            print(f"\n✅ Good sample! Found {unique_states} states")
        else:
            print(f"\n⚠️ Limited sample - only {unique_states} states found")
            print("This might indicate incomplete data upload")
        
        # Check for common states
        common_states = ['KARNATAKA', 'MAHARASHTRA', 'TELANGANA', 'ANDHRA PRADESH', 'GUJARAT', 'TAMILNADU']
        print(f"\nChecking common states:")
        for state in common_states:
            count = state_counts.get(state, 0)
            if count > 0:
                print(f"✅ {state:20} : {count:6} records")
            else:
                print(f"❌ {state:20} : NOT FOUND")
        
        print(f"\nSummary:")
        print(f"• Total points in collection: {total_points}")
        print(f"• Sampled records: {len(all_states)}")
        print(f"• Unique states found: {unique_states}")
        print(f"• Karnataka in sample: {karnataka_count}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_states_efficient()

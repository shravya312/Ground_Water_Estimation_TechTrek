#!/usr/bin/env python3
"""
Check all states in Qdrant collection - comprehensive scan
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from collections import Counter

load_dotenv()

def check_all_qdrant_states():
    """Check all states in Qdrant collection"""
    print("Checking ALL States in Qdrant Collection")
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
        
        # Count states
        state_counts = Counter(all_states)
        unique_states = len(state_counts)
        
        print(f"\nFound {unique_states} unique states:")
        print("=" * 40)
        
        # Sort by count (descending)
        for state, count in state_counts.most_common():
            print(f"{state:25} : {count:6} records")
        
        # Check Karnataka specifically
        karnataka_count = state_counts.get('KARNATAKA', 0)
        print(f"\nKarnataka records: {karnataka_count}")
        
        # Check if all expected states are present
        expected_states = [
            'ANDHRA PRADESH', 'ARUNACHAL PRADESH', 'ASSAM', 'BIHAR', 'CHHATTISGARH',
            'GOA', 'GUJARAT', 'HARYANA', 'HIMACHAL PRADESH', 'JHARKHAND', 'KARNATAKA',
            'KERALA', 'MADHYA PRADESH', 'MAHARASHTRA', 'MANIPUR', 'MEGHALAYA',
            'MIZORAM', 'NAGALAND', 'ODISHA', 'PUNJAB', 'RAJASTHAN', 'SIKKIM',
            'TAMILNADU', 'TELANGANA', 'TRIPURA', 'UTTAR PRADESH', 'UTTARAKHAND',
            'WEST BENGAL', 'DELHI', 'JAMMU AND KASHMIR', 'LADAKH', 'PUDUCHERRY',
            'CHANDIGARH', 'DADRA AND NAGAR HAVELI', 'DAMAN AND DIU', 'LAKSHADWEEP',
            'ANDAMAN AND NICOBAR ISLANDS'
        ]
        
        print(f"\nChecking against expected states ({len(expected_states)}):")
        print("=" * 50)
        
        missing_states = []
        for expected_state in expected_states:
            if expected_state in state_counts:
                count = state_counts[expected_state]
                print(f"✅ {expected_state:30} : {count:6} records")
            else:
                missing_states.append(expected_state)
                print(f"❌ {expected_state:30} : MISSING")
        
        if missing_states:
            print(f"\nMissing states: {len(missing_states)}")
            print("Missing:", missing_states)
        else:
            print(f"\n✅ All expected states present!")
        
        print(f"\nSummary:")
        print(f"• Total records: {total_scanned}")
        print(f"• Unique states: {unique_states}")
        print(f"• Karnataka records: {karnataka_count}")
        print(f"• Missing states: {len(missing_states)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_all_qdrant_states()

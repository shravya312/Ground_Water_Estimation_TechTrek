#!/usr/bin/env python3
"""
Check how many states have been uploaded to ChromaDB so far
"""

import chromadb
from collections import Counter

def check_uploaded_states():
    print("Checking Uploaded States in ChromaDB")
    print("=" * 50)
    
    try:
        # Connect to ChromaDB
        print("1. Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("ingris_groundwater_collection")
        print("[OK] Connected to ChromaDB")
        
        # Get current record count
        total_count = collection.count()
        print(f"Total records uploaded: {total_count:,}")
        
        # Get sample of records to analyze states
        print("\n2. Analyzing uploaded states...")
        
        # Get all records in batches to analyze states
        states = []
        batch_size = 1000
        offset = 0
        
        while offset < total_count:
            # Get batch of records
            results = collection.get(
                limit=batch_size,
                offset=offset,
                include=['metadatas']
            )
            
            if not results['metadatas']:
                break
                
            # Extract states from metadata
            for metadata in results['metadatas']:
                state = metadata.get('STATE', metadata.get('state', 'N/A'))
                if state and state != 'N/A':
                    states.append(state)
            
            offset += batch_size
            print(f"Processed {min(offset, total_count)}/{total_count} records...")
        
        # Analyze states
        print(f"\n3. State Analysis:")
        print(f"Records with state data: {len(states)}")
        
        if states:
            state_counter = Counter(states)
            unique_states = len(state_counter)
            
            print(f"Unique states uploaded: {unique_states}")
            print(f"\nState distribution (top 15):")
            
            for i, (state, count) in enumerate(state_counter.most_common(15), 1):
                percentage = (count / len(states)) * 100
                print(f"  {i:2d}. {state}: {count:,} records ({percentage:.1f}%)")
            
            # Check for Karnataka specifically
            karnataka_count = state_counter.get('KARNATAKA', 0)
            print(f"\n4. Karnataka Status:")
            print(f"Karnataka records uploaded: {karnataka_count}")
            
            if karnataka_count > 0:
                print("[OK] Karnataka data is being uploaded!")
            else:
                print("[INFO] Karnataka data not yet uploaded")
            
            # Check for other major states
            print(f"\n5. Major States Status:")
            major_states = ['ANDHRA PRADESH', 'TELANGANA', 'TAMILNADU', 'UTTAR PRADESH', 'BIHAR']
            for state in major_states:
                count = state_counter.get(state, 0)
                if count > 0:
                    print(f"  {state}: {count:,} records")
                else:
                    print(f"  {state}: Not yet uploaded")
        
        else:
            print("[WARNING] No state data found in uploaded records")
        
        print(f"\n[SUCCESS] State analysis completed!")
        print(f"Upload progress: {total_count:,} records with {unique_states} states")
        
    except Exception as e:
        print(f"[ERROR] Failed to check states: {e}")

if __name__ == "__main__":
    check_uploaded_states()

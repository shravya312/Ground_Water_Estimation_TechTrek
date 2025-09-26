#!/usr/bin/env python3
"""
Check Qdrant data in ingris_groundwater_collection
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

def check_qdrant_data():
    """Check the actual data in Qdrant collection"""
    print("🔍 Checking Qdrant Data in ingris_groundwater_collection")
    print("=" * 60)
    
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = "ingris_groundwater_collection"
    
    try:
        # Connect to Qdrant
        print("🔄 Connecting to Qdrant...")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        
        # Get collection info
        print("🔄 Getting collection information...")
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"✅ Collection: {COLLECTION_NAME}")
        print(f"✅ Total Points: {collection_info.points_count:,}")
        print(f"✅ Vector Size: {collection_info.config.params.vectors.size}")
        print(f"✅ Distance Metric: {collection_info.config.params.vectors.distance}")
        
        # Sample some data to check structure
        print("\n🔄 Sampling data to check structure...")
        sample_points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        print(f"✅ Retrieved {len(sample_points)} sample points")
        
        if sample_points:
            print("\n📊 Sample Point Structure:")
            for i, point in enumerate(sample_points, 1):
                print(f"\n--- Point {i} ---")
                print(f"ID: {point.id}")
                print(f"Payload keys: {list(point.payload.keys())}")
                
                # Show key fields
                for key in ['STATE', 'DISTRICT', 'Assessment_Year', 'rainfall_mm', 'ground_water_recharge_ham']:
                    if key in point.payload:
                        print(f"{key}: {point.payload[key]}")
        
        # Check states in the data
        print("\n🔄 Analyzing states in the data...")
        all_states = []
        batch_size = 1000
        offset = None
        
        while True:
            scroll_result, next_page_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not scroll_result:
                break
                
            for point in scroll_result:
                if 'STATE' in point.payload and point.payload['STATE']:
                    all_states.append(point.payload['STATE'])
            
            offset = next_page_offset
            if offset is None:
                break
            
            print(f"   Processed {len(all_states)} records so far...")
        
        # Analyze states
        state_counts = Counter(all_states)
        print(f"\n📊 State Analysis (Total: {len(all_states)} records):")
        print("=" * 40)
        
        for state, count in state_counts.most_common():
            print(f"{state}: {count:,} records")
        
        # Check specifically for Odisha
        print(f"\n🔍 Checking for Odisha specifically:")
        odisha_variations = ['ODISHA', 'ORISSA', 'odisha', 'orissa']
        for variation in odisha_variations:
            count = state_counts.get(variation, 0)
            print(f"   {variation}: {count} records")
        
        # Check for similar states
        print(f"\n🔍 States with similar names to Odisha:")
        similar_states = [state for state in state_counts.keys() 
                         if any(word in state.upper() for word in ['ODISHA', 'ORISSA', 'ODI', 'ORI'])]
        for state in similar_states:
            print(f"   {state}: {state_counts[state]} records")
        
        # Sample Odisha data if it exists
        odisha_states = [state for state in state_counts.keys() 
                        if 'ODISHA' in state.upper() or 'ORISSA' in state.upper()]
        
        if odisha_states:
            print(f"\n📊 Sample Odisha Data:")
            odisha_filter = Filter(
                must=[
                    FieldCondition(
                        key="STATE",
                        match=MatchValue(value=odisha_states[0])
                    )
                ]
            )
            
            odisha_sample, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=odisha_filter,
                limit=3,
                with_payload=True,
                with_vectors=False
            )
            
            for i, point in enumerate(odisha_sample, 1):
                print(f"\n--- Odisha Sample {i} ---")
                payload = point.payload
                print(f"State: {payload.get('STATE')}")
                print(f"District: {payload.get('DISTRICT')}")
                print(f"Year: {payload.get('Assessment_Year')}")
                print(f"Rainfall: {payload.get('rainfall_mm')} mm")
                print(f"Recharge: {payload.get('ground_water_recharge_ham')} ham")
        else:
            print("\n❌ No Odisha data found in the collection")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    check_qdrant_data()

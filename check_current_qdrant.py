#!/usr/bin/env python3
"""
Check current Qdrant collection status
"""

import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "groundwater_excel_collection"

def check_current_qdrant():
    """Check current Qdrant collection status."""
    print("🔍 Checking Current Qdrant Collection")
    print("=" * 50)
    
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        timeout=60
    )
    
    try:
        # Get collection info
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"📊 Collection: {COLLECTION_NAME}")
        print(f"   Points: {collection_info.points_count:,}")
        print(f"   Vector size: {collection_info.config.params.vectors.size}")
        print(f"   Distance metric: {collection_info.config.params.vectors.distance}")
        
        # Get sample data
        print(f"\n📋 Sample Data:")
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=3,
            with_payload=True,
            with_vectors=False
        )
        
        for i, point in enumerate(points):
            payload = point.payload
            print(f"\nRecord {i+1}:")
            print(f"   STATE: {payload.get('STATE', 'N/A')}")
            print(f"   DISTRICT: {payload.get('DISTRICT', 'N/A')}")
            print(f"   Assessment_Year: {payload.get('Assessment_Year', 'N/A')}")
            
            # Check for null/empty values
            null_fields = []
            for key, value in payload.items():
                if value is None or value == 0.0 or value == '0.0' or value == '':
                    null_fields.append(key)
            
            print(f"   Null/Empty fields: {len(null_fields)} out of {len(payload)}")
            if null_fields:
                print(f"   Sample null fields: {null_fields[:5]}...")
        
        # Check state distribution
        print(f"\n📊 State Distribution:")
        all_points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=collection_info.points_count,
            with_payload=True,
            with_vectors=False
        )
        
        states = {}
        for point in all_points:
            state = point.payload.get('STATE', 'Unknown')
            states[state] = states.get(state, 0) + 1
        
        print(f"   Total states: {len(states)}")
        for state, count in sorted(states.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {state}: {count:,} records")
        
        return collection_info.points_count, len(states)
        
    except Exception as e:
        print(f"❌ Error checking collection: {e}")
        return 0, 0

if __name__ == "__main__":
    points, states = check_current_qdrant()
    print(f"\n📈 Summary:")
    print(f"   Total records: {points:,}")
    print(f"   Unique states: {states}")
    
    if points < 100000:
        print(f"\n⚠️ Collection has limited data. Need to upload complete INGRIS dataset (162,632 records)")
    else:
        print(f"\n✅ Collection has comprehensive data")

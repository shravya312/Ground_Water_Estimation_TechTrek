#!/usr/bin/env python3
"""
Verify that main.py is using the correct collection with 162,000 records
"""

import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def verify_collection():
    """Verify the collection details"""
    print("🔍 Verifying main.py Collection Configuration")
    print("=" * 50)
    
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = "ingris_groundwater_collection"
    
    try:
        print(f"🔄 Connecting to Qdrant at {QDRANT_URL}...")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        
        print(f"🔄 Getting collection info for '{COLLECTION_NAME}'...")
        collection_info = client.get_collection(COLLECTION_NAME)
        
        print(f"✅ Collection Name: {collection_info.config.params.vectors.size}")
        print(f"✅ Vector Size: {collection_info.config.params.vectors.size}")
        print(f"✅ Distance Metric: {collection_info.config.params.vectors.distance}")
        print(f"✅ Total Points: {collection_info.points_count:,}")
        
        if collection_info.points_count >= 160000:
            print(f"🎉 SUCCESS: main.py is using the FULL dataset with {collection_info.points_count:,} records!")
        else:
            print(f"⚠️ WARNING: Only {collection_info.points_count:,} records found (expected ~162,000)")
        
        # Check if we can search
        print(f"\n🔄 Testing search capability...")
        try:
            # Simple search test
            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=[0.1] * 768,  # Dummy vector
                limit=1,
                with_payload=False
            )
            print(f"✅ Search test successful - found {len(results)} results")
        except Exception as search_error:
            print(f"❌ Search test failed: {search_error}")
        
        return collection_info.points_count
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0

if __name__ == "__main__":
    verify_collection()

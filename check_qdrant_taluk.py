#!/usr/bin/env python3
"""
Check if taluk field is present in Qdrant data
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pandas as pd

def check_qdrant_taluk_data():
    """Check what Taluk data is actually in Qdrant"""
    print("🔍 Checking Qdrant Taluk Data")
    print("=" * 50)
    
    # Qdrant configuration
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME = "ingris_groundwater_collection"
    
    try:
        # Connect to Qdrant
        print(f"🔄 Connecting to Qdrant at {QDRANT_URL}...")
        client = QdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None, 
            timeout=30,
            prefer_grpc=False
        )
        
        # Test connection
        collections = client.get_collections()
        print(f"✅ Connected to Qdrant")
        print(f"📊 Available collections: {[c.name for c in collections.collections]}")
        
        # Get collection info
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"📊 Collection '{COLLECTION_NAME}' info:")
        print(f"  - Points count: {collection_info.points_count}")
        print(f"  - Vector size: {collection_info.config.params.vectors.size}")
        
        # Get sample data
        print(f"\n🔍 Getting sample data from Qdrant...")
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5,
            with_payload=True
        )
        
        print(f"📊 Sample of 5 records:")
        print("-" * 40)
        
        taluk_count = 0
        none_count = 0
        missing_count = 0
        
        for i, (point, payload) in enumerate(zip(results[0], results[1]), 1):
            print(f"\nRecord {i} (ID: {point.id}):")
            print(f"  State: {payload.get('STATE', 'NOT_FOUND')}")
            print(f"  District: {payload.get('DISTRICT', 'NOT_FOUND')}")
            
            # Check for taluk field
            if 'taluk' in payload:
                taluk_value = payload['taluk']
                print(f"  Taluk: {taluk_value} (type: {type(taluk_value)})")
                
                if pd.notna(taluk_value) and str(taluk_value).strip() != '' and str(taluk_value).strip().lower() != 'nan':
                    taluk_count += 1
                else:
                    none_count += 1
            else:
                print(f"  Taluk: FIELD NOT FOUND")
                missing_count += 1
        
        print(f"\n📈 Summary:")
        print(f"  Records with valid Taluk data: {taluk_count}")
        print(f"  Records with None/empty Taluk: {none_count}")
        print(f"  Records missing Taluk field: {missing_count}")
        
        # Check specifically for Karnataka data
        print(f"\n🔍 Checking Karnataka data specifically:")
        print("-" * 40)
        
        # Search for Karnataka data
        try:
            karnataka_results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=[0.1] * 768,  # Dummy vector
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="STATE",
                            match=models.MatchValue(value="KARNATAKA")
                        )
                    ]
                ),
                limit=5,
                with_payload=True
            )
            
            if karnataka_results:
                print("Karnataka query results:")
                for i, hit in enumerate(karnataka_results, 1):
                    payload = hit.payload
                    taluk_value = payload.get('taluk', 'NOT_FOUND')
                    state = payload.get('STATE', 'NOT_FOUND')
                    district = payload.get('DISTRICT', 'NOT_FOUND')
                    
                    print(f"  Result {i}: {state}, {district}, Taluk: {taluk_value}")
            else:
                print("No Karnataka results found")
                
        except Exception as e:
            print(f"Error querying Karnataka data: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_qdrant_taluk_data()

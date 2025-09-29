#!/usr/bin/env python3
"""
Check Qdrant collection data structure
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

def check_qdrant_data():
    """Check what data is available in Qdrant collection"""
    try:
        # Load environment variables
        load_dotenv()
        
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        print(f"Connecting to Qdrant: {qdrant_url}")
        
        # Connect to Qdrant
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key if qdrant_api_key else None,
            timeout=30,
            prefer_grpc=False
        )
        
        # Get collection info
        collection_info = client.get_collection("ingris_groundwater_collection")
        print(f"Collection points: {collection_info.points_count}")
        
        # Get sample data
        print("\nFetching sample data...")
        scroll_result, _ = client.scroll(
            collection_name="ingris_groundwater_collection",
            limit=5,
            with_payload=True
        )
        
        print(f"Sample records: {len(scroll_result)}")
        
        if scroll_result:
            print("\nSample record structure:")
            sample = scroll_result[0]
            print(f"ID: {sample.id}")
            print(f"Payload keys: {list(sample.payload.keys())}")
            
            # Show sample payload
            for key, value in sample.payload.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"{key}: {value[:100]}...")
                else:
                    print(f"{key}: {value}")
        
        # Check for state data
        print("\nChecking for state data...")
        search_result = client.search(
            collection_name="ingris_groundwater_collection",
            query_vector=[0.0] * 768,  # Dummy vector
            query_filter={
                "must": [
                    {"key": "STATE", "match": {"value": "KARNATAKA"}}
                ]
            },
            limit=3,
            with_payload=True
        )
        
        print(f"Karnataka records found: {len(search_result)}")
        if search_result:
            print("Sample Karnataka record:")
            sample = search_result[0]
            for key, value in sample.payload.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"{key}: {value[:100]}...")
                else:
                    print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_qdrant_data()
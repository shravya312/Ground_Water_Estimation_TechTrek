#!/usr/bin/env python3
"""
Basic Qdrant check - just get collection info
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def check_qdrant_basic():
    """Basic check of Qdrant collection info only"""
    print("Basic Qdrant Collection Check")
    print("=" * 35)
    
    try:
        # Connect to Qdrant
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            timeout=10
        )
        print("Connected to Qdrant")
        
        # Get collection info only
        collection_info = client.get_collection('ingris_groundwater_collection')
        
        print(f"Collection name: ingris_groundwater_collection")
        print(f"Total points: {collection_info.points_count:,}")
        print(f"Vector size: {collection_info.config.params.vectors.size}")
        print(f"Distance metric: {collection_info.config.params.vectors.distance}")
        
        # Try to get just 1 record to see if data is accessible
        print("\nTrying to get 1 sample record...")
        try:
            results = client.scroll(
                collection_name='ingris_groundwater_collection',
                limit=1,
                with_payload=True
            )
            
            if results[0]:
                sample = results[0][0]
                print("Sample record found:")
                print(f"  ID: {sample.id}")
                print(f"  Payload keys: {list(sample.payload.keys())}")
                
                # Check for STATE field
                if 'STATE' in sample.payload:
                    print(f"  STATE: {sample.payload['STATE']}")
                else:
                    print("  STATE field not found")
            else:
                print("No records found")
                
        except Exception as e:
            print(f"Error getting sample: {e}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nPossible issues:")
        print("- Qdrant service is down")
        print("- Network connectivity issues")
        print("- API key or URL incorrect")

if __name__ == "__main__":
    check_qdrant_basic()

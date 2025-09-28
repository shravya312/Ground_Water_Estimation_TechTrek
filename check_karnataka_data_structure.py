#!/usr/bin/env python3
"""
Check the actual structure of Karnataka data in Qdrant
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

load_dotenv()

def check_karnataka_data_structure():
    """Check the actual structure of Karnataka data"""
    print("Checking Karnataka Data Structure in Qdrant")
    print("=" * 60)
    
    try:
        # Connect to Qdrant
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        
        # Get Karnataka data
        karnataka_filter = Filter(
            must=[FieldCondition(key='STATE', match=MatchValue(value='KARNATAKA'))]
        )
        
        results = client.scroll(
            collection_name='ingris_groundwater_collection',
            scroll_filter=karnataka_filter,
            limit=5,
            with_payload=True
        )
        
        print(f"Found {len(results[0])} Karnataka records")
        
        if results[0]:
            print("\nSample Karnataka record structure:")
            print("=" * 40)
            
            sample = results[0][0]
            payload = sample.payload
            
            print("Payload fields:")
            for key, value in payload.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
            
            print("\nOriginal data structure:")
            if 'original_data' in payload:
                original = payload['original_data']
                print("Original data fields:")
                for key, value in original.items():
                    if isinstance(value, str) and len(value) > 50:
                        print(f"  {key}: {value[:50]}...")
                    else:
                        print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_karnataka_data_structure()

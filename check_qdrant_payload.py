#!/usr/bin/env python3
"""
Check Qdrant payload structure to understand why states show as Unknown
"""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

def check_qdrant_payload_structure():
    """Check the actual payload structure in Qdrant"""
    print("Checking Qdrant payload structure...")
    
    try:
        # Initialize Qdrant client
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY') if os.getenv('QDRANT_API_KEY') else None,
            timeout=30
        )
        print("Qdrant client connected")
        
        # Use correct 768-dimensional model
        model = SentenceTransformer('all-mpnet-base-v2')
        print(f"Using model: all-mpnet-base-v2 (768 dimensions)")
        
        # Test with Chhattisgarh query
        query_text = "groundwater chhattisgarh"
        query_vector = model.encode(query_text).tolist()
        
        print(f"Testing query: '{query_text}'")
        
        # Search in Qdrant
        results = client.search(
            collection_name="ingris_groundwater_collection",
            query_vector=query_vector,
            limit=5
        )
        
        print(f"Results found: {len(results)}")
        
        if results:
            print("\nDetailed payload analysis:")
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"  Score: {result.score}")
                print(f"  ID: {result.id}")
                print(f"  Payload keys: {list(result.payload.keys())}")
                print(f"  Full payload: {result.payload}")
                
                # Check different possible state field names
                state_fields = ['state', 'STATE', 'State', 'state_name', 'STATE_NAME']
                for field in state_fields:
                    if field in result.payload:
                        print(f"  {field}: {result.payload[field]}")
        
        # Also try a broader search to see more states
        print(f"\nTrying broader search...")
        broad_query = "groundwater"
        broad_vector = model.encode(broad_query).tolist()
        
        broad_results = client.search(
            collection_name="ingris_groundwater_collection",
            query_vector=broad_vector,
            limit=10
        )
        
        print(f"Broad search results: {len(broad_results)}")
        
        states_found = set()
        for i, result in enumerate(broad_results):
            payload = result.payload
            print(f"\nBroad Result {i+1}:")
            print(f"  Payload keys: {list(payload.keys())}")
            
            # Try all possible state fields
            for field in ['state', 'STATE', 'State', 'state_name', 'STATE_NAME']:
                if field in payload:
                    state_value = payload[field]
                    states_found.add(state_value)
                    print(f"  {field}: {state_value}")
        
        print(f"\nUnique states found: {len(states_found)}")
        for state in sorted(states_found):
            if 'CHHATTISGARH' in str(state).upper() or 'CHATTISGARH' in str(state).upper():
                print(f"  *** {state} ***")
            else:
                print(f"  {state}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_qdrant_payload_structure()
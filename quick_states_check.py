#!/usr/bin/env python3
"""
Quick states check - fast method
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from collections import Counter

load_dotenv()

def quick_states_check():
    """Quick check of all states"""
    print("Quick Qdrant States Check")
    print("=" * 30)
    
    try:
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            timeout=30
        )
        
        # Get collection info
        collection_info = client.get_collection('ingris_groundwater_collection')
        print(f"Total points: {collection_info.points_count:,}")
        
        # Use multiple search vectors to get diverse samples
        all_states = set()
        
        # Try different vectors to get different samples
        vectors = [
            [0.1] * 768, [0.2] * 768, [0.3] * 768, [0.4] * 768, [0.5] * 768,
            [0.6] * 768, [0.7] * 768, [0.8] * 768, [0.9] * 768, [1.0] * 768,
            [0.15] * 768, [0.25] * 768, [0.35] * 768, [0.45] * 768, [0.55] * 768,
            [0.65] * 768, [0.75] * 768, [0.85] * 768, [0.95] * 768, [0.05] * 768
        ]
        
        for i, vector in enumerate(vectors):
            try:
                results = client.query_points(
                    collection_name='ingris_groundwater_collection',
                    query=vector,
                    limit=200,
                    with_payload=True
                )
                
                for result in results.points:
                    state = result.payload.get('STATE', 'N/A')
                    if state and state != 'N/A':
                        all_states.add(state)
                
                print(f"Search {i+1}/{len(vectors)}: {len(all_states)} states", end='\r')
                
            except:
                continue
        
        print(f"\n\nFound {len(all_states)} unique states:")
        print("-" * 30)
        
        for i, state in enumerate(sorted(all_states), 1):
            print(f"{i:2d}. {state}")
        
        # Check Odisha
        odisha_found = any('ODISHA' in s.upper() or 'ORISSA' in s.upper() for s in all_states)
        print(f"\nOdisha found: {'Yes' if odisha_found else 'No'}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    quick_states_check()

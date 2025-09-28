#!/usr/bin/env python3
"""
Check Karnataka data in Qdrant Cloud collection
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

load_dotenv()

def main():
    # Connect to Qdrant Cloud
    client = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY')
    )

    print('Connected to Qdrant Cloud')

    # Check what states are in the collection
    try:
        all_results = client.scroll(
            collection_name='ingris_groundwater_collection',
            limit=100,
            with_payload=True
        )
        states = set()
        for result in all_results[0]:
            state = result.payload.get('STATE')
            if state:
                states.add(state)
        print(f'States in Qdrant collection: {sorted(states)}')
        
        # Check specifically for Karnataka
        karnataka_results = client.scroll(
            collection_name='ingris_groundwater_collection',
            scroll_filter=Filter(
                must=[FieldCondition(key='STATE', match=MatchValue(value='KARNATAKA'))]
            ),
            limit=5,
            with_payload=True
        )
        print(f'Found {len(karnataka_results[0])} Karnataka records in Qdrant')
        
        if len(karnataka_results[0]) > 0:
            print('Sample Karnataka record from Qdrant:')
            sample = karnataka_results[0][0]
            print(f'  State: {sample.payload.get("STATE")}')
            print(f'  District: {sample.payload.get("DISTRICT")}')
            print(f'  Year: {sample.payload.get("Assessment_Year")}')
        
        # Test a search query for Karnataka
        print('\nTesting search for Karnataka groundwater data...')
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        query_vector = model.encode(['groundwater estimation in Karnataka'])[0].tolist()
        
        search_results = client.search(
            collection_name='ingris_groundwater_collection',
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key='STATE', match=MatchValue(value='KARNATAKA'))]
            ),
            limit=5,
            with_payload=True
        )
        
        print(f'Search returned {len(search_results)} results for Karnataka')
        for i, result in enumerate(search_results):
            print(f'  Result {i+1}: {result.payload.get("STATE")} - {result.payload.get("DISTRICT")} (score: {result.score:.3f})')
        
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    main()

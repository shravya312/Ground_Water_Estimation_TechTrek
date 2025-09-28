#!/usr/bin/env python3
"""
Read complete Qdrant data and check for Karnataka
"""

import os
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from collections import Counter

load_dotenv()

def read_complete_qdrant_data():
    print("Reading Complete Qdrant Data")
    print("=" * 50)
    
    # Connect to Qdrant Cloud
    try:
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        print("Connected to Qdrant Cloud")
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        return
    
    # Get collection info
    print("\n1. Collection Information")
    print("-" * 30)
    try:
        collection_info = client.get_collection('ingris_groundwater_collection')
        print(f"Collection name: {collection_info.config.params.collection_name}")
        print(f"Vector size: {collection_info.config.params.vectors.size}")
        print(f"Distance metric: {collection_info.config.params.vectors.distance}")
        print(f"Points count: {collection_info.points_count}")
    except Exception as e:
        print(f"Error getting collection info: {e}")
    
    # Read all data in batches
    print("\n2. Reading All Data")
    print("-" * 30)
    all_data = []
    offset = None
    batch_size = 1000
    total_processed = 0
    
    try:
        while True:
            # Scroll through all data
            result = client.scroll(
                collection_name='ingris_groundwater_collection',
                limit=batch_size,
                offset=offset,
                with_payload=True
            )
            
            points = result[0]
            if not points:
                break
                
            all_data.extend(points)
            total_processed += len(points)
            offset = result[1]  # Next offset
            
            print(f"Processed {total_processed} records...")
            
            if len(points) < batch_size:
                break
                
    except Exception as e:
        print(f"Error reading data: {e}")
        return
    
    print(f"\nTotal records read: {len(all_data)}")
    
    # Analyze state data
    print("\n3. State Analysis")
    print("-" * 30)
    states = []
    state_field_variations = []
    
    for point in all_data:
        payload = point.payload
        
        # Check different possible state field names
        state_value = None
        state_field = None
        
        for field in ['STATE', 'state', 'State']:
            if field in payload:
                state_value = payload[field]
                state_field = field
                break
        
        if state_value:
            states.append(state_value)
            if state_field not in state_field_variations:
                state_field_variations.append(state_field)
    
    print(f"State field variations found: {state_field_variations}")
    print(f"Records with state data: {len(states)}")
    
    # Count states
    state_counter = Counter(states)
    print(f"\nUnique states: {len(state_counter)}")
    print("\nState distribution (top 20):")
    for state, count in state_counter.most_common(20):
        print(f"  {state}: {count} records")
    
    # Check for Karnataka specifically
    print("\n4. Karnataka Analysis")
    print("-" * 30)
    karnataka_variants = []
    for state in state_counter.keys():
        if 'KARNATAKA' in str(state).upper():
            karnataka_variants.append(state)
    
    print(f"Karnataka variants found: {karnataka_variants}")
    
    total_karnataka = 0
    for variant in karnataka_variants:
        count = state_counter[variant]
        total_karnataka += count
        print(f"  {variant}: {count} records")
    
    print(f"\nTotal Karnataka records: {total_karnataka}")
    
    if total_karnataka > 0:
        print("\n5. Sample Karnataka Records")
        print("-" * 30)
        karnataka_samples = []
        for point in all_data:
            payload = point.payload
            state_value = None
            for field in ['STATE', 'state', 'State']:
                if field in payload and 'KARNATAKA' in str(payload[field]).upper():
                    state_value = payload[field]
                    break
            
            if state_value:
                karnataka_samples.append({
                    'id': point.id,
                    'state': state_value,
                    'district': payload.get('DISTRICT', payload.get('district', 'N/A')),
                    'year': payload.get('Assessment_Year', payload.get('year', 'N/A')),
                    'taluk': payload.get('taluk', 'N/A')
                })
                
                if len(karnataka_samples) >= 5:
                    break
        
        for i, sample in enumerate(karnataka_samples, 1):
            print(f"Sample {i}:")
            print(f"  ID: {sample['id']}")
            print(f"  State: {sample['state']}")
            print(f"  District: {sample['district']}")
            print(f"  Year: {sample['year']}")
            print(f"  Taluk: {sample['taluk']}")
            print()
    else:
        print("❌ No Karnataka data found in Qdrant collection!")
    
    # Check field structure
    print("\n6. Field Structure Analysis")
    print("-" * 30)
    if all_data:
        sample_payload = all_data[0].payload
        print("Sample payload fields:")
        for field in sorted(sample_payload.keys()):
            value = sample_payload[field]
            if isinstance(value, str) and len(value) > 50:
                print(f"  {field}: {value[:50]}...")
            else:
                print(f"  {field}: {value}")
    
    # Search test for Karnataka
    print("\n7. Search Test for Karnataka")
    print("-" * 30)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        query = "groundwater estimation in Karnataka"
        query_vector = model.encode([query])[0].tolist()
        
        # Test search without filter
        print("Testing search without filter...")
        results = client.search(
            collection_name='ingris_groundwater_collection',
            query_vector=query_vector,
            limit=10,
            with_payload=True
        )
        
        print(f"Search results: {len(results)}")
        for i, result in enumerate(results):
            state = result.payload.get('STATE', result.payload.get('state', 'N/A'))
            district = result.payload.get('DISTRICT', result.payload.get('district', 'N/A'))
            print(f"  {i+1}. {state} - {district} (score: {result.score:.3f})")
        
        # Test search with Karnataka filter
        print("\nTesting search with Karnataka filter...")
        try:
            karnataka_results = client.search(
                collection_name='ingris_groundwater_collection',
                query_vector=query_vector,
                query_filter=Filter(
                    must=[FieldCondition(key='STATE', match=MatchValue(value='KARNATAKA'))]
                ),
                limit=10,
                with_payload=True
            )
            print(f"Karnataka filter results: {len(karnataka_results)}")
            for i, result in enumerate(karnataka_results):
                state = result.payload.get('STATE', result.payload.get('state', 'N/A'))
                district = result.payload.get('DISTRICT', result.payload.get('district', 'N/A'))
                print(f"  {i+1}. {state} - {district} (score: {result.score:.3f})")
        except Exception as e:
            print(f"Error with Karnataka filter: {e}")
            
    except Exception as e:
        print(f"Error in search test: {e}")

if __name__ == "__main__":
    read_complete_qdrant_data()

#!/usr/bin/env python3
"""
Check for Karnataka data in Qdrant efficiently
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from collections import Counter

load_dotenv()

def check_karnataka_qdrant():
    print("Checking Karnataka Data in Qdrant")
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
        print(f"Vector size: {collection_info.config.params.vectors.size}")
        print(f"Distance metric: {collection_info.config.params.vectors.distance}")
        print(f"Points count: {collection_info.points_count}")
    except Exception as e:
        print(f"Error getting collection info: {e}")
    
    # Read sample data to understand structure
    print("\n2. Sample Data Structure")
    print("-" * 30)
    try:
        sample_results = client.scroll(
            collection_name='ingris_groundwater_collection',
            limit=10,
            with_payload=True
        )
        
        if sample_results[0]:
            print(f"Retrieved {len(sample_results[0])} sample records")
            sample_payload = sample_results[0][0].payload
            print("Sample payload fields:")
            for field in sorted(sample_payload.keys()):
                value = sample_payload[field]
                if isinstance(value, str) and len(value) > 50:
                    print(f"  {field}: {value[:50]}...")
                else:
                    print(f"  {field}: {value}")
        else:
            print("No sample data found")
            
    except Exception as e:
        print(f"Error getting sample data: {e}")
    
    # Check for Karnataka using different field names and variations
    print("\n3. Karnataka Search")
    print("-" * 30)
    
    karnataka_variations = [
        'KARNATAKA',
        'Karnataka', 
        'karnataka',
        'KARNATKA',
        'Karnatka'
    ]
    
    field_variations = ['STATE', 'state', 'State']
    
    total_karnataka_found = 0
    
    for field in field_variations:
        print(f"\nChecking field: {field}")
        for variation in karnataka_variations:
            try:
                results = client.scroll(
                    collection_name='ingris_groundwater_collection',
                    scroll_filter=Filter(
                        must=[FieldCondition(key=field, match=MatchValue(value=variation))]
                    ),
                    limit=100,
                    with_payload=True
                )
                
                count = len(results[0])
                if count > 0:
                    print(f"  {variation}: {count} records found")
                    total_karnataka_found += count
                    
                    # Show sample records
                    if count > 0:
                        print(f"    Sample record:")
                        sample = results[0][0]
                        state = sample.payload.get('STATE', sample.payload.get('state', 'N/A'))
                        district = sample.payload.get('DISTRICT', sample.payload.get('district', 'N/A'))
                        year = sample.payload.get('Assessment_Year', sample.payload.get('year', 'N/A'))
                        print(f"      State: {state}")
                        print(f"      District: {district}")
                        print(f"      Year: {year}")
                else:
                    print(f"  {variation}: 0 records")
                    
            except Exception as e:
                print(f"  {variation}: Error - {e}")
    
    print(f"\nTotal Karnataka records found: {total_karnataka_found}")
    
    # Test vector search for Karnataka
    print("\n4. Vector Search Test")
    print("-" * 30)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        query = "groundwater estimation in Karnataka"
        query_vector = model.encode([query])[0].tolist()
        
        # Search without filter
        print("Searching without filter...")
        results = client.search(
            collection_name='ingris_groundwater_collection',
            query_vector=query_vector,
            limit=20,
            with_payload=True
        )
        
        print(f"Found {len(results)} results")
        
        karnataka_in_results = 0
        for i, result in enumerate(results):
            state = result.payload.get('STATE', result.payload.get('state', 'N/A'))
            district = result.payload.get('DISTRICT', result.payload.get('district', 'N/A'))
            print(f"  {i+1}. {state} - {district} (score: {result.score:.3f})")
            
            if 'KARNATAKA' in str(state).upper():
                karnataka_in_results += 1
        
        print(f"\nKarnataka records in search results: {karnataka_in_results}")
        
    except Exception as e:
        print(f"Error in vector search: {e}")
    
    # Check all unique states in a sample
    print("\n5. State Distribution Sample")
    print("-" * 30)
    try:
        # Get a larger sample to check state distribution
        sample_results = client.scroll(
            collection_name='ingris_groundwater_collection',
            limit=1000,
            with_payload=True
        )
        
        states = []
        for point in sample_results[0]:
            state = point.payload.get('STATE', point.payload.get('state', None))
            if state:
                states.append(state)
        
        state_counter = Counter(states)
        print(f"States in sample of {len(states)} records:")
        for state, count in state_counter.most_common(15):
            print(f"  {state}: {count}")
            
        # Check if Karnataka is in the sample
        karnataka_in_sample = sum(count for state, count in state_counter.items() 
                                if 'KARNATAKA' in str(state).upper())
        print(f"\nKarnataka records in sample: {karnataka_in_sample}")
        
    except Exception as e:
        print(f"Error checking state distribution: {e}")

if __name__ == "__main__":
    check_karnataka_qdrant()

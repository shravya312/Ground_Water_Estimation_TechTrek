#!/usr/bin/env python3
"""
Comprehensive analysis of Qdrant data structure
"""

import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from collections import Counter

load_dotenv()

def analyze_qdrant_structure():
    print("Qdrant Data Structure Analysis")
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
    
    # Sample data analysis
    print("\n2. Sample Data Analysis")
    print("-" * 30)
    try:
        # Get a sample of records
        sample_results = client.scroll(
            collection_name='ingris_groundwater_collection',
            limit=10,
            with_payload=True
        )
        
        print(f"Retrieved {len(sample_results[0])} sample records")
        
        if sample_results[0]:
            print("\nSample record structure:")
            sample_record = sample_results[0][0]
            print(f"ID: {sample_record.id}")
            print(f"Vector dimension: {len(sample_record.vector)}")
            print(f"Payload keys: {list(sample_record.payload.keys())}")
            
            print("\nSample payload:")
            for key, value in sample_record.payload.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
                    
    except Exception as e:
        print(f"Error in sample analysis: {e}")
    
    # Field analysis
    print("\n3. Field Analysis")
    print("-" * 30)
    try:
        # Get more records for field analysis
        all_results = client.scroll(
            collection_name='ingris_groundwater_collection',
            limit=1000,
            with_payload=True
        )
        
        print(f"Analyzing {len(all_results[0])} records...")
        
        # Collect all field names
        all_fields = set()
        for result in all_results[0]:
            all_fields.update(result.payload.keys())
        
        print(f"Total unique fields: {len(all_fields)}")
        print("All fields:")
        for field in sorted(all_fields):
            print(f"  - {field}")
        
        # Analyze STATE field specifically
        print("\n4. STATE Field Analysis")
        print("-" * 30)
        state_values = []
        for result in all_results[0]:
            state = result.payload.get('STATE')
            if state:
                state_values.append(state)
        
        state_counter = Counter(state_values)
        print(f"Total records with STATE field: {len(state_values)}")
        print(f"Unique STATE values: {len(state_counter)}")
        print("\nSTATE value distribution:")
        for state, count in state_counter.most_common():
            print(f"  {state}: {count} records")
        
        # Check for Karnataka specifically
        karnataka_variants = [state for state in state_counter.keys() if 'KARNATAKA' in state.upper()]
        print(f"\nKarnataka variants found: {karnataka_variants}")
        for variant in karnataka_variants:
            print(f"  {variant}: {state_counter[variant]} records")
            
    except Exception as e:
        print(f"Error in field analysis: {e}")
    
    # Test different search approaches
    print("\n5. Search Test Analysis")
    print("-" * 30)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        query = "groundwater estimation in Karnataka"
        query_vector = model.encode([query])[0].tolist()
        
        # Test 1: Search without filter
        print("Test 1: Search without filter")
        results_no_filter = client.search(
            collection_name='ingris_groundwater_collection',
            query_vector=query_vector,
            limit=10,
            with_payload=True
        )
        
        print(f"Results without filter: {len(results_no_filter)}")
        for i, result in enumerate(results_no_filter):
            state = result.payload.get('STATE', 'N/A')
            district = result.payload.get('DISTRICT', 'N/A')
            print(f"  {i+1}. {state} - {district} (score: {result.score:.3f})")
        
        # Test 2: Search with Karnataka filter
        print("\nTest 2: Search with Karnataka filter")
        try:
            results_with_filter = client.search(
                collection_name='ingris_groundwater_collection',
                query_vector=query_vector,
                query_filter=Filter(
                    must=[FieldCondition(key='STATE', match=MatchValue(value='KARNATAKA'))]
                ),
                limit=10,
                with_payload=True
            )
            
            print(f"Results with Karnataka filter: {len(results_with_filter)}")
            for i, result in enumerate(results_with_filter):
                state = result.payload.get('STATE', 'N/A')
                district = result.payload.get('DISTRICT', 'N/A')
                print(f"  {i+1}. {state} - {district} (score: {result.score:.3f})")
                
        except Exception as e:
            print(f"Error with Karnataka filter: {e}")
        
        # Test 3: Try different Karnataka variations
        print("\nTest 3: Try different Karnataka variations")
        karnataka_variations = ['KARNATAKA', 'Karnataka', 'karnataka', 'KARNATKA']
        
        for variation in karnataka_variations:
            try:
                results = client.search(
                    collection_name='ingris_groundwater_collection',
                    query_vector=query_vector,
                    query_filter=Filter(
                        must=[FieldCondition(key='STATE', match=MatchValue(value=variation))]
                    ),
                    limit=5,
                    with_payload=True
                )
                print(f"  Variation '{variation}': {len(results)} results")
                if results:
                    print(f"    Sample: {results[0].payload.get('STATE')} - {results[0].payload.get('DISTRICT')}")
            except Exception as e:
                print(f"  Variation '{variation}': Error - {e}")
                
    except Exception as e:
        print(f"Error in search test: {e}")
    
    # Data quality analysis
    print("\n6. Data Quality Analysis")
    print("-" * 30)
    try:
        # Check for missing or null values in key fields
        key_fields = ['STATE', 'DISTRICT', 'Assessment_Year', 'taluk', 'block']
        
        for field in key_fields:
            non_null_count = 0
            null_count = 0
            empty_count = 0
            
            for result in all_results[0]:
                value = result.payload.get(field)
                if value is None:
                    null_count += 1
                elif value == '' or str(value).strip() == '':
                    empty_count += 1
                else:
                    non_null_count += 1
            
            print(f"{field}:")
            print(f"  Non-null: {non_null_count}")
            print(f"  Null: {null_count}")
            print(f"  Empty: {empty_count}")
            print(f"  Total: {non_null_count + null_count + empty_count}")
            
    except Exception as e:
        print(f"Error in data quality analysis: {e}")

if __name__ == "__main__":
    analyze_qdrant_structure()

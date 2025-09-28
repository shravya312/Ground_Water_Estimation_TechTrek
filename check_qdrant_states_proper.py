#!/usr/bin/env python3
"""
Properly check all states in Qdrant with correct vector dimensions
"""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

def check_qdrant_with_correct_dimensions():
    """Check Qdrant with correct vector dimensions"""
    print("Checking Qdrant with correct vector dimensions...")
    
    try:
        # Initialize Qdrant client
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY') if os.getenv('QDRANT_API_KEY') else None,
            timeout=30
        )
        print("Qdrant client connected")
        
        # Try different embedding models to match the 768 dimension
        models_to_try = [
            'all-mpnet-base-v2',  # 768 dimensions
            'all-MiniLM-L12-v2',  # 384 dimensions
            'all-MiniLM-L6-v2'    # 384 dimensions
        ]
        
        for model_name in models_to_try:
            try:
                print(f"\nTrying model: {model_name}")
                model = SentenceTransformer(model_name)
                vector_dim = model.get_sentence_embedding_dimension()
                print(f"  Vector dimension: {vector_dim}")
                
                # Test with a simple query
                query_text = "groundwater chhattisgarh"
                query_vector = model.encode(query_text).tolist()
                
                print(f"  Testing query: '{query_text}'")
                
                # Search in Qdrant
                results = client.search(
                    collection_name="ingris_groundwater_collection",
                    query_vector=query_vector,
                    limit=10
                )
                
                print(f"  Results found: {len(results)}")
                
                if results:
                    states_found = set()
                    for i, result in enumerate(results[:5]):
                        payload = result.payload
                        state = payload.get('state', 'Unknown')
                        district = payload.get('district', 'Unknown')
                        states_found.add(state)
                        print(f"    Result {i+1}: State={state}, District={district}")
                    
                    # Check if Chhattisgarh is found
                    chhattisgarh_found = any('CHHATTISGARH' in state.upper() or 'CHATTISGARH' in state.upper() for state in states_found)
                    if chhattisgarh_found:
                        print(f"  SUCCESS: Chhattisgarh found with {model_name}!")
                        return True
                    else:
                        print(f"  States found: {list(states_found)}")
                else:
                    print("  No results found")
                    
            except Exception as e:
                print(f"  Error with {model_name}: {e}")
        
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def check_qdrant_collection_info():
    """Check Qdrant collection information"""
    print("\nChecking Qdrant collection info...")
    
    try:
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY') if os.getenv('QDRANT_API_KEY') else None,
            timeout=30
        )
        
        # Get collection info
        collection_info = client.get_collection("ingris_groundwater_collection")
        print(f"Collection info: {collection_info}")
        
        # Get collection stats
        stats = client.get_collection_stats("ingris_groundwater_collection")
        print(f"Collection stats: {stats}")
        
    except Exception as e:
        print(f"Error getting collection info: {e}")

def check_all_states_in_qdrant():
    """Check all states by sampling more records"""
    print("\nChecking all states in Qdrant by sampling...")
    
    try:
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY') if os.getenv('QDRANT_API_KEY') else None,
            timeout=30
        )
        
        # Use scroll to get more records
        all_states = set()
        offset = None
        batch_size = 1000
        
        for batch in range(5):  # Get 5 batches of 1000 records each
            try:
                print(f"  Getting batch {batch + 1}...")
                scroll_result = client.scroll(
                    collection_name="ingris_groundwater_collection",
                    limit=batch_size,
                    offset=offset
                )
                
                records = scroll_result[0]
                offset = scroll_result[1]  # Next offset
                
                print(f"    Retrieved {len(records)} records")
                
                for record in records:
                    state = record.payload.get('state', '')
                    if state:
                        all_states.add(state)
                
                if not records:  # No more records
                    break
                    
            except Exception as e:
                print(f"    Error in batch {batch + 1}: {e}")
                break
        
        print(f"\nTotal unique states found: {len(all_states)}")
        print("All states:")
        for state in sorted(all_states):
            if 'CHHATTISGARH' in state.upper() or 'CHATTISGARH' in state.upper():
                print(f"  *** {state} ***")
            else:
                print(f"  {state}")
        
        chhattisgarh_found = any('CHHATTISGARH' in state.upper() or 'CHATTISGARH' in state.upper() for state in all_states)
        return chhattisgarh_found
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main function"""
    print("Comprehensive Qdrant state check")
    print("=" * 50)
    
    # Check collection info
    check_qdrant_collection_info()
    
    # Check with correct dimensions
    chhattisgarh_found_dimensions = check_qdrant_with_correct_dimensions()
    
    # Check all states by sampling
    chhattisgarh_found_sampling = check_all_states_in_qdrant()
    
    print(f"\nSummary:")
    print(f"  Chhattisgarh found (correct dimensions): {chhattisgarh_found_dimensions}")
    print(f"  Chhattisgarh found (sampling): {chhattisgarh_found_sampling}")

if __name__ == "__main__":
    main()

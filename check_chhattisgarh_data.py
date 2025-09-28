#!/usr/bin/env python3
"""
Check if Chhattisgarh data exists in both CSV and Qdrant
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

def check_csv_data():
    """Check if Chhattisgarh exists in CSV file"""
    print("Checking CSV data...")
    
    try:
        # Load CSV data
        df = pd.read_csv("ingris_rag_ready_complete.csv", low_memory=False)
        print(f"CSV loaded: {len(df)} records")
        
        # Check for Chhattisgarh variations
        chhattisgarh_variations = [
            'CHHATTISGARH', 'chhattisgarh', 'CHATTISGARH', 'chattisgarh',
            'Chhattisgarh', 'Chattisgarh'
        ]
        
        found_states = set()
        for col in ['state', 'STATE']:
            if col in df.columns:
                unique_states = df[col].dropna().unique()
                found_states.update(unique_states)
        
        print(f"Found {len(found_states)} unique states in CSV")
        
        # Check for Chhattisgarh
        chhattisgarh_found = False
        for state in found_states:
            if any(var in str(state).upper() for var in ['CHHATTISGARH', 'CHATTISGARH']):
                print(f"✅ Found Chhattisgarh in CSV: '{state}'")
                chhattisgarh_found = True
                
                # Show sample records
                chhattisgarh_data = df[df[col].str.contains('CHHATTISGARH|CHATTISGARH', case=False, na=False)]
                print(f"   Records found: {len(chhattisgarh_data)}")
                if len(chhattisgarh_data) > 0:
                    print(f"   Sample districts: {chhattisgarh_data['district'].dropna().unique()[:5].tolist()}")
                break
        
        if not chhattisgarh_found:
            print("❌ Chhattisgarh not found in CSV")
            print("Available states sample:")
            for state in sorted(list(found_states))[:20]:
                print(f"  - {state}")
        
        return chhattisgarh_found
        
    except Exception as e:
        print(f"Error checking CSV: {e}")
        return False

def check_qdrant_data():
    """Check if Chhattisgarh exists in Qdrant"""
    print("\nChecking Qdrant data...")
    
    try:
        # Initialize Qdrant client
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY') if os.getenv('QDRANT_API_KEY') else None,
            timeout=30
        )
        print("Qdrant client connected")
        
        # Initialize embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded")
        
        # Test queries for Chhattisgarh
        test_queries = [
            "CHHATTISGARH",
            "chhattisgarh groundwater",
            "groundwater estimation chhattisgarh"
        ]
        
        chhattisgarh_found = False
        for query in test_queries:
            try:
                print(f"\nTesting query: '{query}'")
                
                # Create query vector
                query_vector = model.encode(query).tolist()
                
                # Search in Qdrant
                results = client.search(
                    collection_name="ingris_groundwater_collection",
                    query_vector=query_vector,
                    limit=10
                )
                
                print(f"  Results found: {len(results)}")
                
                if results:
                    states_found = set()
                    for result in results[:5]:
                        payload = result.payload
                        state = payload.get('state', 'Unknown')
                        district = payload.get('district', 'Unknown')
                        states_found.add(state)
                        print(f"    State: {state}, District: {district}")
                    
                    # Check if any result contains Chhattisgarh
                    if any('CHHATTISGARH' in state.upper() or 'CHATTISGARH' in state.upper() for state in states_found):
                        print("  ✅ Chhattisgarh found in Qdrant!")
                        chhattisgarh_found = True
                    else:
                        print(f"  States found: {list(states_found)}")
                
            except Exception as e:
                print(f"  Error with query '{query}': {e}")
        
        if not chhattisgarh_found:
            print("❌ Chhattisgarh not found in Qdrant")
            
            # Get a sample of states from Qdrant
            print("\nSampling states from Qdrant...")
            try:
                sample_vector = model.encode("groundwater").tolist()
                sample_results = client.search(
                    collection_name="ingris_groundwater_collection",
                    query_vector=sample_vector,
                    limit=50
                )
                
                states = set()
                for result in sample_results:
                    state = result.payload.get('state', '')
                    if state:
                        states.add(state)
                
                print(f"Sample states from Qdrant ({len(states)}):")
                for state in sorted(list(states))[:20]:
                    print(f"  - {state}")
                    
            except Exception as e:
                print(f"Error sampling states: {e}")
        
        return chhattisgarh_found
        
    except Exception as e:
        print(f"Error checking Qdrant: {e}")
        return False

def main():
    """Main function"""
    print("Checking Chhattisgarh data availability")
    print("=" * 50)
    
    csv_found = check_csv_data()
    qdrant_found = check_qdrant_data()
    
    print(f"\nSummary:")
    print(f"  CSV data: {'✅ Found' if csv_found else '❌ Not found'}")
    print(f"  Qdrant data: {'✅ Found' if qdrant_found else '❌ Not found'}")
    
    if csv_found and qdrant_found:
        print("🎉 Chhattisgarh data is available in both sources!")
    elif csv_found:
        print("⚠️  Chhattisgarh data only in CSV, not in Qdrant")
    elif qdrant_found:
        print("⚠️  Chhattisgarh data only in Qdrant, not in CSV")
    else:
        print("❌ Chhattisgarh data not found in either source")

if __name__ == "__main__":
    main()
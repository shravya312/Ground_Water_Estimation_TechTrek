#!/usr/bin/env python3
"""
Check ChromaDB Taluk data specifically
"""

import chromadb
from chromadb.config import Settings
import pandas as pd

def check_chromadb_taluk_data():
    """Check what Taluk data is actually in ChromaDB"""
    print("🔍 Checking ChromaDB Taluk Data")
    print("=" * 50)
    
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(
            path='./chroma_db',
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection = client.get_collection('ingris_groundwater_collection')
        
        # Get sample data
        results = collection.get(limit=10, include=['metadatas'])
        
        print(f"📊 Total records in ChromaDB: {collection.count()}")
        print(f"🔍 Sample of 10 records:")
        print("-" * 40)
        
        taluk_count = 0
        none_count = 0
        
        for i, (id, metadata) in enumerate(zip(results['ids'], results['metadatas']), 1):
            taluk_value = metadata.get('taluk', 'NOT_FOUND')
            state = metadata.get('state', 'NOT_FOUND')
            district = metadata.get('district', 'NOT_FOUND')
            
            print(f"Record {i}:")
            print(f"  ID: {id}")
            print(f"  State: {state}")
            print(f"  District: {district}")
            print(f"  Taluk: {taluk_value}")
            print(f"  Taluk type: {type(taluk_value)}")
            print()
            
            if taluk_value and str(taluk_value).strip() != '' and str(taluk_value).strip().lower() != 'none':
                taluk_count += 1
            else:
                none_count += 1
        
        print(f"📈 Summary:")
        print(f"  Records with Taluk data: {taluk_count}")
        print(f"  Records with None/empty Taluk: {none_count}")
        
        # Check specifically for Karnataka data
        print("\n🔍 Checking Karnataka data specifically:")
        print("-" * 40)
        
        # Try to query for Karnataka data
        try:
            karnataka_results = collection.query(
                query_texts=["Karnataka groundwater"],
                n_results=5,
                include=['metadatas']
            )
            
            if karnataka_results['metadatas'] and karnataka_results['metadatas'][0]:
                print("Karnataka query results:")
                for i, metadata in enumerate(karnataka_results['metadatas'][0], 1):
                    taluk_value = metadata.get('taluk', 'NOT_FOUND')
                    state = metadata.get('state', 'NOT_FOUND')
                    district = metadata.get('district', 'NOT_FOUND')
                    
                    print(f"  Result {i}: {state}, {district}, Taluk: {taluk_value}")
            else:
                print("No Karnataka results found")
                
        except Exception as e:
            print(f"Error querying Karnataka data: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_chromadb_taluk_data()

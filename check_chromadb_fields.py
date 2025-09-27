#!/usr/bin/env python3
"""
Check what fields are actually stored in ChromaDB metadata
"""

import chromadb
from chromadb.config import Settings
import json

def check_chromadb_fields():
    """Check what fields are actually stored in ChromaDB"""
    print("🔍 Checking ChromaDB Metadata Fields")
    print("=" * 50)
    
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(
            path='./chroma_db',
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection = client.get_collection('ingris_groundwater_collection')
        
        # Get sample data
        results = collection.get(limit=3, include=['metadatas'])
        
        print(f"📊 Total records in ChromaDB: {collection.count()}")
        print(f"🔍 Sample metadata fields:")
        print("-" * 40)
        
        for i, (id, metadata) in enumerate(zip(results['ids'], results['metadatas']), 1):
            print(f"\nRecord {i} (ID: {id}):")
            print("Available fields:")
            for key, value in metadata.items():
                print(f"  {key}: {value} (type: {type(value)})")
        
        # Check if taluk field exists in any record
        print(f"\n🔍 Checking for 'taluk' field specifically:")
        print("-" * 40)
        
        all_results = collection.get(limit=100, include=['metadatas'])
        taluk_found = False
        
        for metadata in all_results['metadatas']:
            if 'taluk' in metadata:
                taluk_found = True
                print(f"Found taluk field: {metadata['taluk']}")
                break
        
        if not taluk_found:
            print("❌ 'taluk' field not found in any of the first 100 records")
            print("Available fields in first record:")
            if all_results['metadatas']:
                for key in all_results['metadatas'][0].keys():
                    print(f"  - {key}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_chromadb_fields()

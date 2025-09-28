#!/usr/bin/env python3
"""
Simple ChromaDB cleanup - Remove all data
"""

import chromadb
import os
import shutil

def clear_chromadb_simple():
    print("Clearing ChromaDB Data")
    print("=" * 50)
    
    try:
        # Connect to ChromaDB
        print("1. Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path="./chroma_db")
        print("[OK] Connected to ChromaDB")
        
        # List collections
        print("\n2. Checking existing collections...")
        collections = client.list_collections()
        print(f"Found {len(collections)} collections:")
        
        total_records = 0
        for collection in collections:
            count = collection.count()
            total_records += count
            print(f"  - {collection.name}: {count} records")
        
        if total_records == 0:
            print("[INFO] No data to clear")
            return True
        
        # Delete all collections
        print(f"\n3. Deleting {len(collections)} collections...")
        for collection in collections:
            try:
                client.delete_collection(collection.name)
                print(f"[OK] Deleted collection: {collection.name}")
            except Exception as e:
                print(f"[ERROR] Failed to delete {collection.name}: {e}")
        
        # Remove ChromaDB directory completely
        print("\n4. Removing ChromaDB directory...")
        chroma_path = "./chroma_db"
        if os.path.exists(chroma_path):
            try:
                shutil.rmtree(chroma_path)
                print("[OK] ChromaDB directory removed completely")
            except Exception as e:
                print(f"[ERROR] Failed to remove directory: {e}")
        else:
            print("[INFO] ChromaDB directory does not exist")
        
        print("\n[SUCCESS] ChromaDB data cleared successfully!")
        print("You can now upload fresh data using the ChromaDB Smart Upload Tracker.")
        
    except Exception as e:
        print(f"[ERROR] Failed to clear ChromaDB: {e}")
        return False
    
    return True

if __name__ == "__main__":
    clear_chromadb_simple()

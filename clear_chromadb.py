#!/usr/bin/env python3
"""
Clear ChromaDB collection - Remove previously uploaded data
"""

import chromadb
import os
import shutil

def clear_chromadb():
    print("Clearing ChromaDB Collection")
    print("=" * 50)
    
    try:
        # Connect to ChromaDB
        print("1. Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path="./chroma_db")
        print("[OK] Connected to ChromaDB")
        
        # List all collections
        print("\n2. Listing existing collections...")
        collections = client.list_collections()
        print(f"Found {len(collections)} collections:")
        for collection in collections:
            print(f"  - {collection.name}")
        
        # Clear the groundwater collection
        print("\n3. Clearing groundwater collection...")
        try:
            # Get collection info before deletion
            collection = client.get_collection("ingris_groundwater_collection")
            count_before = collection.count()
            print(f"Collection had {count_before} records")
            
            # Delete the collection
            client.delete_collection("ingris_groundwater_collection")
            print("[OK] Collection deleted successfully")
            
        except Exception as e:
            print(f"[WARNING] Collection may not exist: {e}")
        
        # Clear test collection if it exists
        print("\n4. Clearing test collection...")
        try:
            test_collection = client.get_collection("test_groundwater_collection")
            count_before = test_collection.count()
            print(f"Test collection had {count_before} records")
            
            client.delete_collection("test_groundwater_collection")
            print("[OK] Test collection deleted successfully")
            
        except Exception as e:
            print(f"[INFO] Test collection may not exist: {e}")
        
        # Optionally remove the entire ChromaDB directory
        print("\n5. Checking ChromaDB directory...")
        chroma_path = "./chroma_db"
        if os.path.exists(chroma_path):
            print(f"ChromaDB directory exists: {chroma_path}")
            
            # List contents
            contents = os.listdir(chroma_path)
            print(f"Directory contents: {contents}")
            
            # Ask if user wants to remove entire directory
            print("\n6. Options:")
            print("  a) Keep directory (collections cleared)")
            print("  b) Remove entire ChromaDB directory")
            
            choice = input("\nEnter choice (a/b): ").lower().strip()
            
            if choice == 'b':
                print("Removing entire ChromaDB directory...")
                shutil.rmtree(chroma_path)
                print("[OK] ChromaDB directory removed completely")
            else:
                print("[OK] ChromaDB directory kept (collections cleared)")
        else:
            print("[INFO] ChromaDB directory does not exist")
        
        print("\n[SUCCESS] ChromaDB cleanup completed!")
        print("You can now upload fresh data using the ChromaDB Smart Upload Tracker.")
        
    except Exception as e:
        print(f"[ERROR] Failed to clear ChromaDB: {e}")
        return False
    
    return True

def show_chromadb_status():
    """Show current ChromaDB status"""
    print("\nCurrent ChromaDB Status:")
    print("-" * 30)
    
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = client.list_collections()
        
        if not collections:
            print("No collections found")
        else:
            for collection in collections:
                count = collection.count()
                print(f"Collection: {collection.name} - {count} records")
                
    except Exception as e:
        print(f"Error checking status: {e}")

if __name__ == "__main__":
    print("ChromaDB Data Cleanup Tool")
    print("=" * 50)
    
    # Show current status
    show_chromadb_status()
    
    # Confirm before clearing
    print("\nThis will remove all previously uploaded data from ChromaDB.")
    confirm = input("Are you sure you want to continue? (y/N): ").lower().strip()
    
    if confirm in ['y', 'yes']:
        clear_chromadb()
    else:
        print("Operation cancelled.")

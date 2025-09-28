#!/usr/bin/env python3
"""
Monitor ChromaDB upload progress and verify data is being added
"""

import chromadb
import os
import time
from datetime import datetime

def monitor_chromadb_upload():
    print("Monitoring ChromaDB Upload Progress")
    print("=" * 50)
    
    try:
        # Connect to ChromaDB
        print("1. Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path="./chroma_db")
        print("[OK] Connected to ChromaDB")
        
        # Check if collection exists
        print("\n2. Checking collections...")
        collections = client.list_collections()
        print(f"Found {len(collections)} collections:")
        
        for collection in collections:
            print(f"  - {collection.name}")
        
        # Check groundwater collection specifically
        print("\n3. Checking groundwater collection...")
        try:
            collection = client.get_collection("ingris_groundwater_collection")
            count = collection.count()
            print(f"[OK] Collection 'ingris_groundwater_collection' exists")
            print(f"Current record count: {count:,}")
            
            if count > 0:
                print("\n4. Sample data from collection:")
                # Get a few sample records
                sample_results = collection.peek(limit=3)
                
                if sample_results['ids']:
                    print(f"Sample record IDs: {sample_results['ids'][:3]}")
                    
                    # Check metadata structure
                    if sample_results['metadatas']:
                        sample_metadata = sample_results['metadatas'][0]
                        print(f"Metadata fields: {list(sample_metadata.keys())}")
                        
                        # Check for key fields
                        key_fields = ['STATE', 'state', 'DISTRICT', 'district', 'serial_number']
                        for field in key_fields:
                            if field in sample_metadata:
                                print(f"  {field}: {sample_metadata[field]}")
                    
                    # Check if Karnataka data exists
                    print("\n5. Checking for Karnataka data...")
                    try:
                        # Search for Karnataka records
                        karnataka_results = collection.query(
                            query_texts=["Karnataka groundwater"],
                            n_results=5
                        )
                        
                        if karnataka_results['ids'] and karnataka_results['ids'][0]:
                            print(f"[OK] Found {len(karnataka_results['ids'][0])} Karnataka-related records")
                            
                            # Show sample Karnataka record
                            if karnataka_results['metadatas'] and karnataka_results['metadatas'][0]:
                                karnataka_metadata = karnataka_results['metadatas'][0][0]
                                print(f"Sample Karnataka record:")
                                print(f"  State: {karnataka_metadata.get('STATE', karnataka_metadata.get('state', 'N/A'))}")
                                print(f"  District: {karnataka_metadata.get('DISTRICT', karnataka_metadata.get('district', 'N/A'))}")
                                print(f"  Serial: {karnataka_metadata.get('serial_number', 'N/A')}")
                        else:
                            print("[INFO] No Karnataka records found yet")
                            
                    except Exception as e:
                        print(f"[WARNING] Could not search for Karnataka data: {e}")
                
                # Check upload progress
                print(f"\n6. Upload Progress Analysis:")
                print(f"   Total records uploaded: {count:,}")
                
                if count > 0:
                    # Estimate progress based on expected total
                    expected_total = 162632  # From CSV analysis
                    progress_percent = (count / expected_total) * 100
                    print(f"   Estimated progress: {progress_percent:.2f}%")
                    
                    if count < expected_total:
                        remaining = expected_total - count
                        print(f"   Remaining records: {remaining:,}")
                        print(f"   Status: Upload in progress...")
                    else:
                        print(f"   Status: Upload completed!")
                else:
                    print(f"   Status: No data uploaded yet")
                    
            else:
                print("[INFO] Collection is empty - upload may not have started yet")
                
        except Exception as e:
            print(f"[ERROR] Collection 'ingris_groundwater_collection' not found: {e}")
            print("Upload may not have started yet or collection name is different")
        
        # Check ChromaDB directory
        print("\n7. Checking ChromaDB directory...")
        chroma_path = "./chroma_db"
        if os.path.exists(chroma_path):
            print(f"[OK] ChromaDB directory exists: {chroma_path}")
            
            # Check directory size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(chroma_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            print(f"Directory size: {total_size / 1024 / 1024:.2f} MB")
            
            # List contents
            contents = os.listdir(chroma_path)
            print(f"Directory contents: {contents}")
        else:
            print("[WARNING] ChromaDB directory does not exist")
        
        print(f"\n[SUCCESS] Monitoring completed at {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"[ERROR] Failed to monitor ChromaDB: {e}")

def continuous_monitor():
    """Monitor upload progress continuously"""
    print("Starting continuous monitoring (press Ctrl+C to stop)...")
    print("=" * 60)
    
    try:
        while True:
            print(f"\n--- Monitoring at {datetime.now().strftime('%H:%M:%S')} ---")
            monitor_chromadb_upload()
            print("\nWaiting 30 seconds before next check...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n[INFO] Monitoring stopped by user")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        continuous_monitor()
    else:
        monitor_chromadb_upload()

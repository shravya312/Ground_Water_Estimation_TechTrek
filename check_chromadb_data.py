#!/usr/bin/env python3
"""
Check what data is actually stored in ChromaDB vs CSV
"""

import pandas as pd
import chromadb
from chromadb.config import Settings

def check_chromadb_data():
    """Check what data is stored in ChromaDB"""
    print("🔍 Checking ChromaDB Data")
    print("=" * 40)
    
    # Connect to ChromaDB
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("ingris_groundwater_collection")
        print("✅ ChromaDB connection successful")
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        return
    
    # Get some sample data from ChromaDB
    try:
        results = collection.get(limit=10)
        print(f"📊 ChromaDB has {collection.count()} total records")
        print(f"📋 Retrieved {len(results['ids'])} sample records")
        
        # Check what data is in ChromaDB
        print("\n📝 Sample ChromaDB records:")
        for i, (id, metadata) in enumerate(zip(results['ids'][:5], results['metadatas'][:5])):
            print(f"  {i+1}. ID: {id}")
            if metadata:
                print(f"     State: {metadata.get('state', 'N/A')}")
                print(f"     District: {metadata.get('district', 'N/A')}")
                print(f"     Taluk: {metadata.get('taluk', 'N/A')}")
                print(f"     Block: {metadata.get('block', 'N/A')}")
                print(f"     Mandal: {metadata.get('mandal', 'N/A')}")
                print(f"     Village: {metadata.get('village', 'N/A')}")
                print()
        
        # Search for Karnataka records specifically
        print("🔍 Searching for Karnataka records in ChromaDB:")
        karnataka_results = collection.query(
            query_texts=["Karnataka groundwater"],
            n_results=10
        )
        
        print(f"Found {len(karnataka_results['ids'][0])} Karnataka-related records")
        for i, (id, metadata) in enumerate(zip(karnataka_results['ids'][0][:5], karnataka_results['metadatas'][0][:5])):
            print(f"  {i+1}. ID: {id}")
            if metadata:
                print(f"     State: {metadata.get('state', 'N/A')}")
                print(f"     District: {metadata.get('district', 'N/A')}")
                print(f"     Taluk: {metadata.get('taluk', 'N/A')}")
                print(f"     Block: {metadata.get('block', 'N/A')}")
                print(f"     Mandal: {metadata.get('mandal', 'N/A')}")
                print(f"     Village: {metadata.get('village', 'N/A')}")
                print()
        
    except Exception as e:
        print(f"❌ Error querying ChromaDB: {e}")

def check_csv_data():
    """Check what data is in the CSV"""
    print("\n🔍 Checking CSV Data")
    print("=" * 40)
    
    try:
        df = pd.read_csv('ingris_rag_ready_complete.csv', low_memory=False)
        karnataka = df[df['state'].str.contains('KARNATAKA', case=False, na=False)]
        
        print(f"📊 CSV has {len(df)} total records")
        print(f"📋 Karnataka records: {len(karnataka)}")
        print(f"📋 Karnataka with Taluk: {karnataka['taluk'].notna().sum()}")
        print(f"📋 Karnataka with Block: {karnataka['block'].notna().sum()}")
        print(f"📋 Karnataka with Mandal: {karnataka['mandal'].notna().sum()}")
        print(f"📋 Karnataka with Village: {karnataka['village'].notna().sum()}")
        
        print("\n📝 Sample Karnataka CSV records:")
        sample = karnataka[['state', 'district', 'taluk', 'block', 'mandal', 'village']].head(5)
        for idx, row in sample.iterrows():
            print(f"  {row['district']}: Taluk={row['taluk']}, Block={row['block']}, Mandal={row['mandal']}, Village={row['village']}")
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")

def main():
    """Main function"""
    print("🧪 ChromaDB vs CSV Data Comparison")
    print("=" * 50)
    
    check_csv_data()
    check_chromadb_data()
    
    print("\n💡 Analysis:")
    print("This will help identify if the issue is:")
    print("1. Data not uploaded to ChromaDB correctly")
    print("2. Search not finding the right records")
    print("3. Data format mismatch between CSV and ChromaDB")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fix ChromaDB upload with correct model and complete data
"""

import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import os

def fix_chromadb_upload():
    """Fix ChromaDB upload with correct model and complete data"""
    print("🔧 Fixing ChromaDB Upload")
    print("=" * 40)
    
    # Initialize the correct model (768 dimensions)
    print("🔄 Initializing correct model (all-mpnet-base-v2)...")
    model = SentenceTransformer('all-mpnet-base-v2')
    print(f"✅ Model loaded: {model.get_sentence_embedding_dimension()} dimensions")
    
    # Connect to ChromaDB
    print("🔄 Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete existing collection if it exists
    try:
        client.delete_collection("ingris_groundwater_collection")
        print("🗑️ Deleted existing collection")
    except:
        print("ℹ️ No existing collection to delete")
    
    # Create new collection with correct dimensions
    print("🔄 Creating new collection with correct dimensions...")
    collection = client.create_collection(
        name="ingris_groundwater_collection",
        metadata={"hnsw:space": "cosine"}
    )
    print("✅ New collection created")
    
    # Load CSV data
    print("🔄 Loading CSV data...")
    df = pd.read_csv('ingris_rag_ready_complete.csv', low_memory=False)
    print(f"📊 Loaded {len(df)} records from CSV")
    
    # Process data in batches
    batch_size = 1000
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    print(f"🔄 Processing {total_batches} batches of {batch_size} records each...")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        print(f"🔄 Processing batch {batch_idx + 1}/{total_batches} (records {start_idx} to {end_idx-1})...")
        
        # Prepare batch data
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in batch_df.iterrows():
            # Create document text
            doc_parts = []
            if pd.notna(row.get('state')):
                doc_parts.append(f"State: {row['state']}")
            if pd.notna(row.get('district')):
                doc_parts.append(f"District: {row['district']}")
            if pd.notna(row.get('taluk')):
                doc_parts.append(f"Taluk: {row['taluk']}")
            if pd.notna(row.get('block')):
                doc_parts.append(f"Block: {row['block']}")
            if pd.notna(row.get('mandal')):
                doc_parts.append(f"Mandal: {row['mandal']}")
            if pd.notna(row.get('village')):
                doc_parts.append(f"Village: {row['village']}")
            if pd.notna(row.get('watershed_category')):
                doc_parts.append(f"Watershed Category: {row['watershed_category']}")
            if pd.notna(row.get('instorage_unconfined_ground_water_resourcesham')):
                doc_parts.append(f"Storage: {row['instorage_unconfined_ground_water_resourcesham']}")
            
            document = " | ".join(doc_parts)
            documents.append(document)
            
            # Create metadata (convert all to lowercase for consistency)
            metadata = {}
            for col in row.index:
                if pd.notna(row[col]):
                    metadata[col.lower()] = row[col]
                else:
                    metadata[col.lower()] = None
            metadatas.append(metadata)
            
            # Create unique ID
            ids.append(f"record_{idx}")
        
        # Generate embeddings
        print(f"🔄 Generating embeddings for {len(documents)} documents...")
        embeddings = model.encode(documents, show_progress_bar=False)
        
        # Add to collection
        print(f"🔄 Adding batch to ChromaDB...")
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist()
        )
        
        print(f"✅ Batch {batch_idx + 1} completed")
    
    print(f"🎉 Upload completed! Total records: {len(df)}")
    
    # Verify the upload
    print("🔍 Verifying upload...")
    count = collection.count()
    print(f"📊 ChromaDB now has {count} records")
    
    # Test search
    print("🔍 Testing search...")
    test_results = collection.query(
        query_texts=["Karnataka groundwater"],
        n_results=5
    )
    
    print(f"✅ Search test successful: {len(test_results['ids'][0])} results")
    
    # Show sample results
    print("📝 Sample results:")
    for i, (id, metadata) in enumerate(zip(test_results['ids'][0][:3], test_results['metadatas'][0][:3])):
        print(f"  {i+1}. {metadata.get('state', 'N/A')}, {metadata.get('district', 'N/A')}")
        print(f"     Taluk: {metadata.get('taluk', 'N/A')}")
        print(f"     Block: {metadata.get('block', 'N/A')}")
        print(f"     Mandal: {metadata.get('mandal', 'N/A')}")
        print(f"     Village: {metadata.get('village', 'N/A')}")
        print()

def main():
    """Main function"""
    print("🔧 ChromaDB Fix Script")
    print("=" * 50)
    print("This will:")
    print("1. Delete the existing incomplete collection")
    print("2. Create a new collection with correct dimensions (768)")
    print("3. Upload all 162,632 records with proper metadata")
    print("4. Test the upload")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        fix_chromadb_upload()
    else:
        print("❌ Cancelled")

if __name__ == "__main__":
    main()

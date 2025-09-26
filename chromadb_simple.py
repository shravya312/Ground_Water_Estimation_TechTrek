#!/usr/bin/env python3
"""
Simple ChromaDB implementation for groundwater data
Can handle 2 lakh+ records efficiently
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class GroundwaterChromaDB:
    def __init__(self, collection_name="groundwater_data"):
        """Initialize ChromaDB for groundwater data"""
        print("🔄 Initializing ChromaDB...")
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory="./chroma_db",  # Local storage
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"✅ Found existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Groundwater data collection"}
            )
            print(f"✅ Created new collection: {collection_name}")
        
        # Initialize embedding model
        print("🔄 Loading embedding model...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        print("✅ ChromaDB ready!")
    
    def add_data(self, data_list, batch_size=1000):
        """Add data to ChromaDB in batches"""
        print(f"🔄 Adding {len(data_list)} records to ChromaDB...")
        
        # Process in batches to handle large datasets
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            
            # Prepare data for ChromaDB
            ids = [f"record_{i + j}" for j in range(len(batch))]
            documents = [str(item.get('text', '')) for item in batch]
            metadatas = [item for item in batch]
            
            # Generate embeddings
            embeddings = self.model.encode(documents).tolist()
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            print(f"✅ Added batch {i//batch_size + 1}/{(len(data_list)-1)//batch_size + 1}")
        
        print(f"🎉 Successfully added {len(data_list)} records!")
    
    def search(self, query, n_results=10, where_filter=None):
        """Search the database"""
        # Generate query embedding
        query_embedding = self.model.encode([query])[0].tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        
        return results
    
    def get_collection_info(self):
        """Get collection statistics"""
        count = self.collection.count()
        return {
            "total_records": count,
            "collection_name": self.collection.name
        }

def test_chromadb_capacity():
    """Test ChromaDB with sample data"""
    print("🔍 Testing ChromaDB Capacity")
    print("=" * 40)
    
    # Initialize ChromaDB
    db = GroundwaterChromaDB("test_groundwater")
    
    # Create sample data (simulating 2000 records)
    print("🔄 Creating sample data...")
    sample_data = []
    for i in range(2000):
        sample_data.append({
            "text": f"Groundwater data for district {i} with rainfall {1000 + i}mm",
            "district": f"District_{i}",
            "state": "KARNATAKA" if i % 2 == 0 else "MAHARASHTRA",
            "year": 2020 + (i % 4),
            "rainfall_mm": 1000 + i,
            "ground_water_recharge_ham": 100 + i,
            "stage_of_extraction": 50 + (i % 50)
        })
    
    # Add data
    db.add_data(sample_data, batch_size=500)
    
    # Test search
    print("\n🔍 Testing search...")
    results = db.search("groundwater karnataka", n_results=5)
    
    print(f"✅ Found {len(results['documents'][0])} results")
    for i, doc in enumerate(results['documents'][0][:3]):
        print(f"  {i+1}. {doc[:100]}...")
        print(f"     Metadata: {results['metadatas'][0][i]}")
    
    # Get collection info
    info = db.get_collection_info()
    print(f"\n📊 Collection Info:")
    print(f"   Total records: {info['total_records']}")
    print(f"   Collection: {info['collection_name']}")
    
    return True

if __name__ == "__main__":
    test_chromadb_capacity()

#!/usr/bin/env python3
"""
Script to create the Qdrant collection with correct vector size for improved RAG system.
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "groundwater_excel_collection"
VECTOR_SIZE = 768  # For all-mpnet-base-v2 model

def create_collection():
    """Create the Qdrant collection with correct vector size."""
    print("🚀 Creating Qdrant collection for improved RAG system...")
    print("=" * 60)
    
    # Initialize Qdrant client
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        timeout=60
    )
    
    try:
        # Check if collection exists
        try:
            existing_info = client.get_collection(COLLECTION_NAME)
            print(f"📋 Collection '{COLLECTION_NAME}' already exists")
            print(f"   Points: {existing_info.points_count}")
            print(f"   Vector size: {existing_info.config.params.vectors.size}")
            
            if existing_info.config.params.vectors.size != VECTOR_SIZE:
                print(f"❌ Vector size mismatch! Expected {VECTOR_SIZE}, got {existing_info.config.params.vectors.size}")
                print("🔄 Deleting existing collection...")
                client.delete_collection(COLLECTION_NAME)
                print("✅ Collection deleted")
            else:
                print("✅ Collection has correct vector size")
                return True
                
        except Exception as e:
            if "doesn't exist" in str(e) or "Not found" in str(e):
                print(f"📋 Collection '{COLLECTION_NAME}' doesn't exist, creating new one...")
            else:
                print(f"❌ Error checking collection: {e}")
                return False
        
        # Create collection with correct vector size
        print(f"🔄 Creating collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}...")
        
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        
        print("✅ Collection created successfully!")
        
        # Verify collection
        info = client.get_collection(COLLECTION_NAME)
        print(f"📊 Collection info:")
        print(f"   Name: {COLLECTION_NAME}")
        print(f"   Vector size: {info.config.params.vectors.size}")
        print(f"   Distance metric: {info.config.params.vectors.distance}")
        print(f"   Points: {info.points_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return False

if __name__ == "__main__":
    success = create_collection()
    if success:
        print("\n🎉 Collection ready for data upload!")
    else:
        print("\n💥 Failed to create collection!")

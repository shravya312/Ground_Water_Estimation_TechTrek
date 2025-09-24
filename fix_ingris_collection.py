#!/usr/bin/env python3
"""
Fix INGRIS Collection - Delete and recreate with correct vector size
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "ingris_groundwater_collection"
VECTOR_SIZE = 768

def fix_ingris_collection():
    """Delete existing collection and recreate with correct vector size."""
    print("🔧 Fixing INGRIS Collection Vector Size")
    print("=" * 50)

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        timeout=60
    )

    try:
        # Check if collection exists
        try:
            existing_info = client.get_collection(COLLECTION_NAME)
            print(f"📋 Collection '{COLLECTION_NAME}' found")
            print(f"   Current vector size: {existing_info.config.params.vectors.size}")
            print(f"   Points: {existing_info.points_count}")
            
            if existing_info.config.params.vectors.size != VECTOR_SIZE:
                print(f"❌ Vector size mismatch! Current: {existing_info.config.params.vectors.size}, Expected: {VECTOR_SIZE}")
                print("🔄 Deleting existing collection...")
                client.delete_collection(COLLECTION_NAME)
                print("✅ Collection deleted successfully")
            else:
                print("✅ Collection already has correct vector size")
                return True

        except Exception as e:
            if "doesn't exist" in str(e) or "Not found" in str(e):
                print(f"📋 Collection '{COLLECTION_NAME}' doesn't exist, creating new one...")
            else:
                print(f"❌ Error checking collection: {e}")
                return False

        # Create collection with correct vector size
        print(f"🆕 Creating collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}...")

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print("✅ Collection created successfully!")

        # Verify creation
        info = client.get_collection(COLLECTION_NAME)
        print("\n📊 Collection info:")
        print(f"   Name: {info.name}")
        print(f"   Vector size: {info.config.params.vectors.size}")
        print(f"   Distance metric: {info.config.params.vectors.distance}")
        print(f"   Points: {info.points_count}")
        print("\n🎉 Collection ready for structured data upload!")
        return True

    except Exception as e:
        print(f"💥 Failed to fix collection! Error: {e}")
        return False

if __name__ == "__main__":
    fix_ingris_collection()

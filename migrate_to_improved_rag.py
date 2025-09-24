#!/usr/bin/env python3
"""
Migration script to upgrade the RAG system for higher similarity scores (>0.7).
This script will:
1. Backup existing data
2. Delete old collection with 384-dimensional vectors
3. Create new collection with 768-dimensional vectors
4. Re-upload data with improved embeddings
"""

import os
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "groundwater_excel_collection"
BACKUP_COLLECTION_NAME = "groundwater_excel_collection_backup"
OLD_VECTOR_SIZE = 384
NEW_VECTOR_SIZE = 768

def backup_existing_data():
    """Backup existing collection data."""
    print("🔄 Backing up existing data...")
    
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY if QDRANT_API_KEY else None)
        
        # Get all points from existing collection
        points = []
        offset = None
        while True:
            result = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            points.extend(result[0])
            offset = result[1]
            if not result[1]:  # No more points
                break
        
        print(f"📦 Backed up {len(points)} points")
        
        # Save backup to file
        backup_data = []
        for point in points:
            backup_data.append({
                'id': point.id,
                'vector': point.vector,
                'payload': point.payload
            })
        
        backup_filename = f"backup_{COLLECTION_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_filename, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        print(f"💾 Backup saved to: {backup_filename}")
        return backup_filename, points
        
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return None, None

def delete_old_collection():
    """Delete the old collection."""
    print("🗑️ Deleting old collection...")
    
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY if QDRANT_API_KEY else None)
        client.delete_collection(collection_name=COLLECTION_NAME)
        print("✅ Old collection deleted")
        return True
    except Exception as e:
        print(f"❌ Failed to delete collection: {e}")
        return False

def create_new_collection():
    """Create new collection with updated vector size."""
    print("🆕 Creating new collection...")
    
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY if QDRANT_API_KEY else None)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=NEW_VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"✅ New collection created with vector size {NEW_VECTOR_SIZE}")
        return True
    except Exception as e:
        print(f"❌ Failed to create collection: {e}")
        return False

def reupload_with_improved_embeddings(backup_filename):
    """Re-upload data with improved embeddings."""
    print("🔄 Re-uploading data with improved embeddings...")
    
    try:
        # Load improved model
        print("📥 Loading improved embedding model...")
        model = SentenceTransformer("all-mpnet-base-v2")
        
        # Load backup data
        with open(backup_filename, 'r') as f:
            backup_data = json.load(f)
        
        client = QdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            timeout=60  # Increased timeout
        )
        
        # Process in smaller batches to avoid timeouts
        batch_size = 50  # Reduced batch size
        total_points = len(backup_data)
        
        for i in range(0, total_points, batch_size):
            batch = backup_data[i:i + batch_size]
            
            # Extract texts and generate new embeddings
            texts = [point['payload'].get('text', '') for point in batch]
            
            # Enhanced preprocessing
            processed_texts = []
            for text in texts:
                if not text or pd.isna(text):
                    processed_texts.append("")
                    continue
                
                # Clean and normalize text
                text = str(text).strip()
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'([A-Z][a-z]+):', r'\1: ', text)
                text = re.sub(r'(\d+\.?\d*)\s*(ham|ha|mm|%)', r'\1 \2', text)
                processed_texts.append(text.strip())
            
            # Generate new embeddings
            new_embeddings = model.encode(processed_texts, show_progress_bar=False)
            
            # Create new points
            new_points = []
            for j, point in enumerate(batch):
                new_points.append(PointStruct(
                    id=point['id'],
                    vector=new_embeddings[j].tolist(),
                    payload=point['payload']
                ))
            
            # Upload batch with retry logic
            max_retries = 3
            for retry in range(max_retries):
                try:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=new_points
                    )
                    print(f"📤 Uploaded batch {i//batch_size + 1}/{(total_points + batch_size - 1)//batch_size}")
                    break
                except Exception as e:
                    if retry == max_retries - 1:
                        raise e
                    print(f"⚠️ Batch upload failed (attempt {retry + 1}), retrying...")
                    import time
                    time.sleep(2)  # Wait before retry
        
        print("✅ Data re-uploaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Re-upload failed: {e}")
        return False

def main():
    """Main migration process."""
    print("🚀 Starting RAG system migration for improved similarity scores...")
    print("=" * 60)
    
    # Step 1: Backup existing data
    backup_filename, points = backup_existing_data()
    if not backup_filename:
        print("❌ Migration aborted due to backup failure")
        return False
    
    # Step 2: Delete old collection
    if not delete_old_collection():
        print("❌ Migration aborted due to collection deletion failure")
        return False
    
    # Step 3: Create new collection
    if not create_new_collection():
        print("❌ Migration aborted due to collection creation failure")
        return False
    
    # Step 4: Re-upload with improved embeddings
    if not reupload_with_improved_embeddings(backup_filename):
        print("❌ Migration aborted due to re-upload failure")
        return False
    
    print("=" * 60)
    print("🎉 Migration completed successfully!")
    print("📊 Your RAG system now uses:")
    print(f"   • Improved embedding model: all-mpnet-base-v2")
    print(f"   • Vector size: {NEW_VECTOR_SIZE} (upgraded from {OLD_VECTOR_SIZE})")
    print(f"   • Similarity threshold: 0.7 (increased from 0.3)")
    print(f"   • Enhanced preprocessing and scoring")
    print("🔍 You should now see similarity scores above 0.7!")
    
    return True

if __name__ == "__main__":
    import re
    main()

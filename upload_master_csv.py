#!/usr/bin/env python3
"""
Upload the master CSV data to Qdrant with all states
"""

import os
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json
import uuid
import re
from datetime import datetime

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "groundwater_excel_collection"
VECTOR_SIZE = 768

def preprocess_text_for_embedding(text):
    """Enhanced text preprocessing for better semantic understanding."""
    if not text or pd.isna(text):
        return ""
    
    # Convert to string and clean
    text = str(text).strip()
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Preserve important structure markers
    text = re.sub(r'([A-Z][a-z]+):', r'\1: ', text)  # Add space after colons
    text = re.sub(r'(\d+\.?\d*)\s*(ham|ha|mm|%)', r'\1 \2', text)  # Preserve units
    
    # Enhanced preprocessing for better semantic matching
    text = re.sub(r'\bground\s*water\b', 'groundwater', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwater\s*table\b', 'watertable', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwater\s*level\b', 'waterlevel', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwater\s*recharge\b', 'waterrecharge', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwater\s*extraction\b', 'waterextraction', text, flags=re.IGNORECASE)
    
    # Normalize state names
    state_mappings = {
        'karnataka': 'karnataka state',
        'maharashtra': 'maharashtra state', 
        'tamil nadu': 'tamil nadu state',
        'gujarat': 'gujarat state',
        'rajasthan': 'rajasthan state',
        'kerala': 'kerala state',
        'andhra pradesh': 'andhra pradesh state'
    }
    
    for state, expanded in state_mappings.items():
        text = re.sub(rf'\b{re.escape(state)}\b', expanded, text, flags=re.IGNORECASE)
    
    return text.strip()

def create_detailed_combined_text(row):
    """Generates a detailed combined text string for a DataFrame row."""
    parts = []
    for col, value in row.items():
        if pd.notna(value) and value != '' and col not in ['S.No']:
            parts.append(f"{col}: {value}")
    return " | ".join(parts)

def upload_master_csv_to_qdrant():
    """Upload master CSV data to Qdrant."""
    print("🚀 Uploading Master CSV Data to Qdrant")
    print("=" * 60)
    
    # Initialize components
    print("🔄 Initializing components...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        timeout=60
    )
    
    model = SentenceTransformer("all-mpnet-base-v2")
    print("✅ Components initialized")
    
    # Read master CSV
    print("📄 Reading master CSV...")
    df = pd.read_csv('master_groundwater_data.csv', low_memory=False)
    print(f"📊 Found {len(df)} records in master CSV")
    
    # Create combined text for each row
    print("🔄 Creating combined text...")
    df['combined_text'] = df.apply(create_detailed_combined_text, axis=1)
    
    # Remove duplicates based on combined text
    print("🔄 Removing duplicates...")
    initial_count = len(df)
    df = df.drop_duplicates(subset=['combined_text'])
    final_count = len(df)
    print(f"📊 Deduplication: {initial_count} → {final_count} records ({initial_count - final_count} duplicates removed)")
    
    # Prepare data for upload
    print("🔄 Preparing data for upload...")
    texts = df['combined_text'].tolist()
    
    # Generate embeddings in batches
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        processed_texts = [preprocess_text_for_embedding(text) for text in batch_texts]
        batch_embeddings = model.encode(processed_texts, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)
        print(f"   📤 Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
    
    # Prepare points for upload
    points = []
    for i, (_, row) in enumerate(df.iterrows()):
        point_id = str(uuid.uuid4())
        payload = row.to_dict()
        payload['text'] = row['combined_text']
        
        points.append(PointStruct(
            id=point_id,
            vector=all_embeddings[i].tolist(),
            payload=payload
        ))
    
    # Upload to Qdrant
    print("🔄 Uploading to Qdrant...")
    upload_batch_size = 50
    
    for i in range(0, len(points), upload_batch_size):
        batch_points = points[i:i + upload_batch_size]
        
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch_points
            )
            print(f"   📤 Batch {i//upload_batch_size + 1}/{(len(points) + upload_batch_size - 1)//upload_batch_size}: {len(batch_points)} records uploaded")
        except Exception as e:
            print(f"   ❌ Error uploading batch {i//upload_batch_size + 1}: {e}")
            continue
    
    # Verify upload
    print("🔄 Verifying upload...")
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"✅ Total points in collection: {collection_info.points_count}")
    
    print("\n🎉 Master CSV upload completed!")
    print(f"📈 Uploaded {final_count} records from master CSV")
    print("🔍 Your RAG system now has comprehensive groundwater data for all states!")

if __name__ == "__main__":
    upload_master_csv_to_qdrant()

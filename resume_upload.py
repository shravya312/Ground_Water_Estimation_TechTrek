#!/usr/bin/env python3
"""
Resume upload script to complete the dataset upload process.
This script will upload the remaining records from the Excel files.
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
import glob

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
    
    return text.strip()

def create_detailed_combined_text(row):
    """Generates a detailed combined text string for a DataFrame row."""
    parts = []
    for col, value in row.items():
        if pd.notna(value) and value != '' and col not in ['S.No']:
            parts.append(f"{col}: {value}")
    return " | ".join(parts)

def get_existing_ids(client):
    """Get all existing IDs from the collection to avoid duplicates."""
    print("🔍 Checking existing records...")
    existing_ids = set()
    
    try:
        # Get all points in batches
        offset = None
        while True:
            result = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )
            
            for point in result[0]:
                existing_ids.add(point.id)
            
            offset = result[1]
            if not result[1]:  # No more points
                break
        
        print(f"📊 Found {len(existing_ids)} existing records")
        return existing_ids
        
    except Exception as e:
        print(f"⚠️ Error getting existing IDs: {e}")
        return set()

def upload_remaining_data():
    """Upload remaining data from Excel files."""
    print("🚀 Resuming Dataset Upload")
    print("=" * 40)
    
    # Initialize components
    try:
        client = QdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            timeout=60
        )
        model = SentenceTransformer("all-mpnet-base-v2")
        print("✅ Components initialized")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    # Get existing IDs
    existing_ids = get_existing_ids(client)
    
    # Process Excel files
    datasets_dir = "datasets123/datasets"
    excel_files = glob.glob(os.path.join(datasets_dir, "*.xlsx"))
    
    all_new_records = []
    
    for excel_file in excel_files:
        print(f"📄 Processing: {os.path.basename(excel_file)}")
        
        try:
            df = pd.read_excel(excel_file)
            df['combined_text'] = df.apply(create_detailed_combined_text, axis=1)
            df = df[df['combined_text'].str.strip() != '']
            
            # Filter out existing records
            new_records = []
            for _, row in df.iterrows():
                content_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, row['combined_text']))
                if content_hash not in existing_ids:
                    new_records.append(row)
            
            all_new_records.extend(new_records)
            print(f"   📊 {len(new_records)} new records from this file")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    if not all_new_records:
        print("✅ No new records to upload!")
        return True
    
    print(f"📊 Total new records to upload: {len(all_new_records)}")
    
    # Convert to DataFrame
    new_df = pd.DataFrame(all_new_records)
    
    # Upload in small batches
    batch_size = 25  # Smaller batches for stability
    uploaded_count = 0
    
    for i in range(0, len(new_df), batch_size):
        batch_df = new_df.iloc[i:i + batch_size]
        
        try:
            # Prepare texts
            texts = batch_df['combined_text'].tolist()
            processed_texts = [preprocess_text_for_embedding(text) for text in texts]
            
            # Generate embeddings
            embeddings = model.encode(processed_texts, show_progress_bar=False)
            
            # Create points
            points = []
            for j, (_, row) in enumerate(batch_df.iterrows()):
                content_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, row['combined_text']))
                
                point = PointStruct(
                    id=content_hash,
                    vector=embeddings[j].tolist(),
                    payload={
                        "text": row['combined_text'],
                        "original_data": row.to_dict()
                    }
                )
                points.append(point)
            
            # Upload batch
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            
            uploaded_count += len(points)
            batch_num = i // batch_size + 1
            total_batches = (len(new_df) + batch_size - 1) // batch_size
            print(f"   📤 Batch {batch_num}/{total_batches}: {len(points)} records uploaded")
            
        except Exception as e:
            print(f"   ❌ Batch upload failed: {e}")
            continue
    
    # Final verification
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        total_points = collection_info.points_count
        print(f"📊 Total points in collection: {total_points}")
    except Exception as e:
        print(f"⚠️ Could not verify total points: {e}")
    
    print("=" * 40)
    print(f"🎉 Upload completed! Added {uploaded_count} new records")
    
    return True

if __name__ == "__main__":
    upload_remaining_data()

#!/usr/bin/env python3
"""
Upload master CSV data with proper structured payload format
"""

import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json
import uuid
import re
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "groundwater_excel_collection"
VECTOR_SIZE = 768
MASTER_CSV_PATH = "master_groundwater_data.csv"

# Global model instance
_model = None

def initialize_sentence_transformer():
    global _model
    if _model is None:
        print("📥 Loading Sentence Transformer model...")
        _model = SentenceTransformer("all-mpnet-base-v2")
        print("✅ Model loaded.")
    return _model

def preprocess_text_for_embedding(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([A-Z][a-z]+):', r'\1: ', text)
    text = re.sub(r'(\d+\.?\d*)\s*(ham|ha|mm|%)', r'\1 \2', text)
    return text.strip()

def create_detailed_combined_text(row):
    parts = []
    for col, value in row.items():
        if pd.notna(value) and value != '' and col not in ['S.No']:
            parts.append(f"{col}: {value}")
    return " | ".join(parts)

def clear_and_create_collection(client):
    """Deletes and recreates the Qdrant collection."""
    print(f"🔄 Checking for existing collection '{COLLECTION_NAME}'...")
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"🗑️ Collection '{COLLECTION_NAME}' found. Deleting...")
        client.delete_collection(COLLECTION_NAME)
        print("✅ Collection deleted.")
    except Exception as e:
        if "doesn't exist" in str(e) or "Not found" in str(e):
            print(f"📋 Collection '{COLLECTION_NAME}' does not exist. Proceeding to create.")
        else:
            print(f"❌ Error checking collection: {e}")
            raise

    print(f"🆕 Creating new collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print("✅ New collection created successfully.")

def upload_structured_data():
    print("🚀 Uploading Master CSV Data with Structured Payload")
    print("=" * 60)

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        timeout=120
    )
    model = initialize_sentence_transformer()

    try:
        clear_and_create_collection(client)

        print(f"📄 Reading master CSV from {MASTER_CSV_PATH}...")
        df = pd.read_csv(MASTER_CSV_PATH, low_memory=False)
        print(f"📊 Found {len(df)} records in master CSV")

        print("🔄 Creating combined text...")
        df['combined_text'] = df.apply(create_detailed_combined_text, axis=1)

        print("🔄 Removing duplicates based on combined_text...")
        initial_records = len(df)
        df.drop_duplicates(subset=['combined_text'], inplace=True)
        unique_records = len(df)
        print(f"📊 Deduplication: {initial_records} → {unique_records} records ({initial_records - unique_records} duplicates removed)")

        print("🔄 Preparing structured data for upload...")
        points_to_upsert = []
        batch_size = 50
        total_batches = (unique_records + batch_size - 1) // batch_size
        uploaded_count = 0

        for i in range(0, unique_records, batch_size):
            batch_df = df.iloc[i:i + batch_size]
            processed_texts = [preprocess_text_for_embedding(text) for text in batch_df['combined_text'].tolist()]
            batch_embeddings = model.encode(processed_texts, show_progress_bar=False)

            for idx, row in batch_df.iterrows():
                # Create structured payload with individual fields
                payload = {
                    # Individual structured fields
                    'STATE': str(row.get('STATE', 'N/A')),
                    'DISTRICT': str(row.get('DISTRICT', 'N/A')),
                    'Assessment_Year': str(row.get('Assessment_Year', 'N/A')),
                    'S.No': str(row.get('S.No', 'N/A')),
                    
                    # Text fields for search
                    'text': row['combined_text'],
                    'combined_text': row['combined_text'],
                    
                    # Store all original data
                    'original_data': row.to_dict()
                }
                
                # Add all other columns as individual fields
                for col in row.index:
                    if col not in ['S.No', 'STATE', 'DISTRICT', 'Assessment_Year', 'combined_text']:
                        payload[col] = str(row[col]) if pd.notna(row[col]) else 'N/A'

                points_to_upsert.append(
                    PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, row['combined_text'])),
                        vector=batch_embeddings[len(points_to_upsert) % batch_size].tolist(),
                        payload=payload
                    )
                )
            
            # Upsert in batches
            if len(points_to_upsert) >= batch_size or (i + batch_size) >= unique_records:
                retries = 0
                while retries < 3:
                    try:
                        client.upsert(
                            collection_name=COLLECTION_NAME,
                            wait=True,
                            points=points_to_upsert
                        )
                        uploaded_count += len(points_to_upsert)
                        print(f"   📤 Batch {(i // batch_size) + 1}/{total_batches}: {len(points_to_upsert)} records uploaded. Total: {uploaded_count}/{unique_records}")
                        points_to_upsert = []
                        break
                    except Exception as e:
                        print(f"❌ Error uploading batch {(i // batch_size) + 1}: {e}. Retrying...")
                        retries += 1
                        time.sleep(5)
                if retries == 3:
                    print(f"⚠️ Failed to upload batch {(i // batch_size) + 1} after multiple retries. Skipping remaining records.")
                    break

        print(f"✅ Successfully uploaded {uploaded_count} records")
        
        # Verify the upload
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"📊 Total points in collection: {collection_info.points_count}")
        
        # Test a sample record
        print("\n🔍 Testing sample record structure...")
        sample_points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        if sample_points:
            sample_payload = sample_points[0].payload
            print(f"   Sample STATE: {sample_payload.get('STATE', 'N/A')}")
            print(f"   Sample DISTRICT: {sample_payload.get('DISTRICT', 'N/A')}")
            print(f"   Sample Assessment_Year: {sample_payload.get('Assessment_Year', 'N/A')}")
            print(f"   Payload keys: {list(sample_payload.keys())}")
        
        print("============================================================\n🎉 Upload completed successfully!")
        print(f"📈 Uploaded {uploaded_count} records from {MASTER_CSV_PATH}")
        print("🔍 Your RAG system now has properly structured groundwater data!")

    except Exception as e:
        print(f"💥 An error occurred during the upload process: {e}")

if __name__ == "__main__":
    upload_structured_data()

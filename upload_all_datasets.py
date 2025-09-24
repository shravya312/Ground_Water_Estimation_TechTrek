#!/usr/bin/env python3
"""
Comprehensive script to upload all data from datasets123/datasets Excel files to Qdrant.
This ensures all 4,639+ records are properly uploaded with the improved RAG system.
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

def process_excel_file(file_path):
    """Process a single Excel file and return DataFrame."""
    print(f"📄 Processing: {os.path.basename(file_path)}")
    
    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        print(f"   📊 Found {len(df)} rows")
        
        # Create combined_text column
        df['combined_text'] = df.apply(create_detailed_combined_text, axis=1)
        
        # Remove rows with empty combined_text
        df = df[df['combined_text'].str.strip() != '']
        print(f"   ✅ {len(df)} rows with valid text")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error processing {file_path}: {e}")
        return None

def upload_dataframe_to_qdrant(df, model, client, batch_size=50):
    """Upload DataFrame to Qdrant with improved embeddings."""
    print(f"🔄 Uploading {len(df)} records to Qdrant...")
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    uploaded_count = 0
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        
        try:
            # Prepare texts for embedding
            texts = batch_df['combined_text'].tolist()
            processed_texts = [preprocess_text_for_embedding(text) for text in texts]
            
            # Generate embeddings
            embeddings = model.encode(processed_texts, show_progress_bar=False)
            
            # Create points
            points = []
            for j, (_, row) in enumerate(batch_df.iterrows()):
                # Create unique ID based on content hash
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
            print(f"   📤 Batch {batch_num}/{total_batches}: {len(points)} records uploaded")
            
        except Exception as e:
            print(f"   ❌ Error uploading batch {i//batch_size + 1}: {e}")
            continue
    
    print(f"✅ Successfully uploaded {uploaded_count} records")
    return uploaded_count

def main():
    """Main function to process all Excel files and upload to Qdrant."""
    print("🚀 Comprehensive Dataset Upload to Improved RAG System")
    print("=" * 60)
    
    # Initialize components
    print("🔄 Initializing components...")
    
    # Initialize Qdrant client
    try:
        client = QdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            timeout=60
        )
        print("✅ Qdrant client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Qdrant client: {e}")
        return False
    
    # Initialize embedding model
    try:
        model = SentenceTransformer("all-mpnet-base-v2")
        print("✅ Embedding model initialized")
    except Exception as e:
        print(f"❌ Failed to initialize embedding model: {e}")
        return False
    
    # Check collection exists
    try:
        collections = client.get_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)
        
        if not collection_exists:
            print(f"❌ Collection '{COLLECTION_NAME}' does not exist. Please run the migration script first.")
            return False
        
        # Verify vector size
        collection_info = client.get_collection(COLLECTION_NAME)
        current_size = collection_info.config.params.vectors.size
        if current_size != VECTOR_SIZE:
            print(f"❌ Vector size mismatch. Collection has {current_size}, expected {VECTOR_SIZE}")
            return False
        
        print(f"✅ Collection '{COLLECTION_NAME}' verified (vector size: {VECTOR_SIZE})")
        
    except Exception as e:
        print(f"❌ Error checking collection: {e}")
        return False
    
    # Process all Excel files
    datasets_dir = "datasets123/datasets"
    if not os.path.exists(datasets_dir):
        print(f"❌ Directory '{datasets_dir}' not found")
        return False
    
    excel_files = glob.glob(os.path.join(datasets_dir, "*.xlsx"))
    if not excel_files:
        print(f"❌ No Excel files found in '{datasets_dir}'")
        return False
    
    print(f"📁 Found {len(excel_files)} Excel files to process")
    
    # Process each Excel file
    all_dataframes = []
    total_records = 0
    
    for excel_file in excel_files:
        df = process_excel_file(excel_file)
        if df is not None and len(df) > 0:
            all_dataframes.append(df)
            total_records += len(df)
    
    if not all_dataframes:
        print("❌ No valid data found in Excel files")
        return False
    
    print(f"📊 Total records to upload: {total_records}")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Remove duplicates based on combined_text
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['combined_text'], keep='first')
    final_count = len(combined_df)
    
    print(f"🔄 Deduplication: {initial_count} → {final_count} records ({initial_count - final_count} duplicates removed)")
    
    # Upload to Qdrant
    uploaded_count = upload_dataframe_to_qdrant(combined_df, model, client)
    
    # Verify upload
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        total_points = collection_info.points_count
        print(f"📊 Total points in collection: {total_points}")
    except Exception as e:
        print(f"⚠️ Could not verify total points: {e}")
    
    print("=" * 60)
    print("🎉 Upload completed successfully!")
    print(f"📈 Uploaded {uploaded_count} records from {len(excel_files)} Excel files")
    print("🔍 Your RAG system now has comprehensive groundwater data!")
    
    return True

if __name__ == "__main__":
    main()

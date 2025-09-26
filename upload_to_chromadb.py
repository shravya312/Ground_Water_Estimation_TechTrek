#!/usr/bin/env python3
"""
Upload INGRIS data to ChromaDB
Based on smart_upload_tracker_structured.py but using ChromaDB instead of Qdrant
"""

import pandas as pd
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import time
import json
from datetime import datetime
import re
import uuid

load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "ingris_groundwater_collection"
PROGRESS_FILE = "chromadb_upload_progress.json"
CSV_FILE = "ingris_rag_ready_complete.csv"
BATCH_SIZE = 1000

def initialize_components():
    """Initialize all required components."""
    try:
        print("🔄 Initializing components...")
        
        # ChromaDB client
        chroma_client = chromadb.Client(Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        
        # Sentence transformer
        model = SentenceTransformer("all-mpnet-base-v2", device='cpu')
        
        # Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        print("✅ Components initialized successfully!")
        return {
            'chroma_client': chroma_client,
            'model': model,
            'gemini_model': gemini_model
        }
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return None

def preprocess_text_for_embedding(text):
    """Preprocess text for better embedding generation."""
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([A-Z][a-z]+):', r'\1: ', text)
    text = re.sub(r'(\d+\.?\d*)\s*(ham|ha|mm|%)', r'\1 \2', text)
    return text.strip()

def create_detailed_combined_text(row):
    """Create detailed combined text from row data."""
    parts = []
    for col, value in row.items():
        if pd.notna(value) and value != '' and col not in ['serial_number']:
            parts.append(f"{col}: {value}")
    return " | ".join(parts)

def setup_collection(components):
    """Set up ChromaDB collection."""
    try:
        print("🔄 Setting up ChromaDB collection...")
        
        # Get or create collection
        try:
            collection = components['chroma_client'].get_collection(COLLECTION_NAME)
            print(f"✅ Found existing collection: {COLLECTION_NAME}")
        except:
            collection = components['chroma_client'].create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "INGRIS Groundwater Data Collection"}
            )
            print(f"✅ Created new collection: {COLLECTION_NAME}")
        
        return collection
    except Exception as e:
        print(f"❌ Failed to setup collection: {e}")
        return None

def load_progress():
    """Load upload progress from file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"last_uploaded_batch": 0, "total_uploaded": 0, "total_batches": 0}
    return {"last_uploaded_batch": 0, "total_uploaded": 0, "total_batches": 0}

def save_progress(progress):
    """Save upload progress to file."""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save progress: {e}")

def upload_batch_chromadb(components, collection, batch_df, batch_num, progress):
    """Upload a batch to ChromaDB."""
    try:
        batch_size = len(batch_df)
        print(f"🔄 Processing batch {batch_num} with {batch_size} records...")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for j, (_, row) in enumerate(batch_df.iterrows()):
            # Create combined text for embedding
            combined_text = create_detailed_combined_text(row)
            
            if combined_text and len(combined_text.strip()) > 0:
                # Preprocess text for embedding
                processed_text = preprocess_text_for_embedding(combined_text)
                
                # Generate unique ID
                record_id = f"ingris_{batch_num}_{j}_{uuid.uuid4().hex[:8]}"
                
                # Create structured metadata
                metadata = {}
                for col, value in row.items():
                    if pd.notna(value) and value != '':
                        # Convert to appropriate type
                        if isinstance(value, (int, float)):
                            metadata[col] = value
                        else:
                            metadata[col] = str(value)
                
                # Add combined text to metadata
                metadata['combined_text'] = combined_text
                metadata['processed_text'] = processed_text
                
                ids.append(record_id)
                documents.append(processed_text)
                metadatas.append(metadata)
        
        if documents:
            # Generate embeddings
            print(f"🔄 Generating embeddings for {len(documents)} documents...")
            embeddings = components['model'].encode(documents).tolist()
            
            # Add to ChromaDB
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            print(f"✅ Successfully uploaded batch {batch_num} with {len(documents)} records")
            return len(documents)
        else:
            print(f"⚠️ No valid documents in batch {batch_num}")
            return 0
            
    except Exception as e:
        print(f"❌ Error uploading batch {batch_num}: {e}")
        return 0

def main():
    """Main upload function."""
    print("🚀 Starting INGRIS Data Upload to ChromaDB")
    print("=" * 50)
    
    # Initialize components
    components = initialize_components()
    if not components:
        return
    
    # Setup collection
    collection = setup_collection(components)
    if not collection:
        return
    
    # Load progress
    progress = load_progress()
    print(f"📊 Previous progress: {progress['total_uploaded']} records uploaded")
    
    # Load CSV data
    print(f"🔄 Loading data from {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"✅ Loaded {len(df)} records from CSV")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return
    
    # Calculate batches
    total_records = len(df)
    total_batches = (total_records + BATCH_SIZE - 1) // BATCH_SIZE
    progress['total_batches'] = total_batches
    
    print(f"📊 Total records: {total_records}")
    print(f"📊 Batch size: {BATCH_SIZE}")
    print(f"📊 Total batches: {total_batches}")
    
    # Start from where we left off
    start_batch = progress['last_uploaded_batch']
    if start_batch > 0:
        print(f"🔄 Resuming from batch {start_batch + 1}")
    
    # Upload batches
    for batch_num in range(start_batch, total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min((batch_num + 1) * BATCH_SIZE, total_records)
        batch_df = df.iloc[start_idx:end_idx]
        
        print(f"\n🔄 Processing batch {batch_num + 1}/{total_batches}")
        print(f"   Records: {start_idx + 1} to {end_idx}")
        
        # Upload batch
        uploaded_count = upload_batch_chromadb(components, collection, batch_df, batch_num + 1, progress)
        
        # Update progress
        progress['last_uploaded_batch'] = batch_num + 1
        progress['total_uploaded'] += uploaded_count
        save_progress(progress)
        
        # Show progress
        percentage = ((batch_num + 1) / total_batches) * 100
        print(f"📊 Progress: {percentage:.1f}% ({progress['total_uploaded']}/{total_records} records)")
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.1)
    
    # Final verification
    final_count = collection.count()
    print(f"\n🎉 Upload completed!")
    print(f"📊 Final record count in ChromaDB: {final_count}")
    
    # Test search
    print("\n🔍 Testing search functionality...")
    try:
        results = collection.query(
            query_texts=["ground water estimation in karnataka"],
            n_results=5
        )
        
        print(f"✅ Search test successful! Found {len(results['documents'][0])} results")
        for i, doc in enumerate(results['documents'][0][:3]):
            print(f"  {i+1}. {doc[:100]}...")
            print(f"     State: {results['metadatas'][0][i].get('STATE', 'N/A')}")
            print(f"     District: {results['metadatas'][0][i].get('DISTRICT', 'N/A')}")
    except Exception as e:
        print(f"⚠️ Search test failed: {e}")
    
    # Clean up progress file
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("🧹 Cleaned up progress file")

if __name__ == "__main__":
    main()

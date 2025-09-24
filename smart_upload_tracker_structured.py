#!/usr/bin/env python3
"""
Smart Upload Tracker - Structured Payload Version
Uploads INGRIS data with proper structured payload format (individual fields like STATE, DISTRICT, etc.)
"""

import streamlit as st
import pandas as pd
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import time
import json
from datetime import datetime
import threading
import uuid
import re

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "ingris_groundwater_collection"
VECTOR_SIZE = 768  # Updated to match the improved RAG system
PROGRESS_FILE = "upload_progress_structured.json"

def initialize_components():
    """Initialize all required components."""
    try:
        # Qdrant client
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        
        # Sentence transformer - using improved model
        model = SentenceTransformer("all-mpnet-base-v2", device='cpu')
        
        # Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        return {
            'qdrant_client': qdrant_client,
            'model': model,
            'gemini_model': gemini_model
        }
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
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
    """Set up Qdrant collection with proper vector size."""
    try:
        collections = components['qdrant_client'].get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            components['qdrant_client'].create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            st.success(f"✅ Created new collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}")
            return True
        else:
            # Check if existing collection has correct vector size
            try:
                collection_info = components['qdrant_client'].get_collection(COLLECTION_NAME)
                current_vector_size = collection_info.config.params.vectors.size
                if current_vector_size != VECTOR_SIZE:
                    st.warning(f"⚠️ Vector size mismatch! Current: {current_vector_size}, Expected: {VECTOR_SIZE}")
                    st.info("🔄 Please delete the existing collection and recreate it with the correct vector size.")
                    return False
                else:
                    st.success(f"✅ Collection '{COLLECTION_NAME}' exists with correct vector size {VECTOR_SIZE}")
                    return True
            except Exception as e:
                st.error(f"Error checking collection: {e}")
                return False
    except Exception as e:
        st.error(f"Failed to setup collection: {e}")
        return False

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
        st.warning(f"Could not save progress: {e}")

def get_collection_point_count(components):
    """Get total number of points in the collection."""
    try:
        collection_info = components['qdrant_client'].get_collection(COLLECTION_NAME)
        return collection_info.points_count
    except Exception as e:
        st.warning(f"Could not get collection count: {e}")
        return 0

def upload_batch_structured(components, batch_df, batch_num, progress_container, status_container, progress):
    """Upload a batch with structured payload format."""
    try:
        batch_size = len(batch_df)
        
        points = []
        
        for j, (_, row) in enumerate(batch_df.iterrows()):
            # Create combined text for embedding
            combined_text = create_detailed_combined_text(row)
            
            if combined_text and len(combined_text.strip()) > 0:
                # Preprocess text for embedding
                processed_text = preprocess_text_for_embedding(combined_text)
                
                # Generate embedding
                embedding = components['model'].encode([processed_text])[0]
                
                # Create structured payload with individual fields
                payload = {
                    # Individual structured fields
                    'STATE': str(row.get('state', 'N/A')),
                    'DISTRICT': str(row.get('district', 'N/A')),
                    'Assessment_Year': str(row.get('year', 'N/A')),
                    'serial_number': str(row.get('serial_number', 'N/A')),
                    
                    # Text fields for search
                    'text': combined_text,
                    'combined_text': combined_text,
                    
                    # Store all original data
                    'original_data': row.to_dict()
                }
                
                # Add all other columns as individual fields
                for col in row.index:
                    if col not in ['serial_number', 'state', 'district', 'year', 'combined_text']:
                        payload[col] = str(row[col]) if pd.notna(row[col]) else 'N/A'
                
                # Create point with UUID-based ID
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, combined_text))
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
                points.append(point)
        
        # Upload batch
        if points:
            components['qdrant_client'].upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True
            )
            
            # Update progress
            progress["last_uploaded_batch"] = batch_num
            progress["total_uploaded"] = get_collection_point_count(components)
            save_progress(progress)
            
            return len(points), True, 0
        else:
            return 0, True, 0
            
    except Exception as e:
        status_container.error(f"❌ Error uploading batch {batch_num}: {e}")
        return 0, False, 0

def upload_data_with_structured_progress(components, df, batch_size=25):
    """Upload data with structured payload and smart progress tracking."""
    total_rows = len(df)
    total_batches = (total_rows + batch_size - 1) // batch_size
    
    # Load existing progress
    progress = load_progress()
    
    # Calculate start batch based on current collection size
    current_count = get_collection_point_count(components)
    if current_count > 0:
        # Estimate which batch to start from based on current collection size
        estimated_batch = (current_count // batch_size) + 1
        start_batch = max(1, estimated_batch)
        progress["total_uploaded"] = current_count
        save_progress(progress)
    else:
        start_batch = progress.get("last_uploaded_batch", 0) + 1
    
    # Create progress containers
    progress_container = st.container()
    status_container = st.container()
    stats_container = st.container()
    
    # Initialize progress
    progress_bar = progress_container.progress(0)
    status_text = status_container.empty()
    stats_text = stats_container.empty()
    
    uploaded_count = progress.get("total_uploaded", 0)
    failed_batches = 0
    skipped_total = 0
    start_time = time.time()
    
    if start_batch > 1:
        status_text.text(f"🔄 Smart Resume: Starting from batch {start_batch}/{total_batches} (Collection has {current_count} points)")
    else:
        status_text.text(f"🚀 Starting structured upload of {total_rows} rows in {total_batches} batches...")
    
    for i in range((start_batch - 1) * batch_size, total_rows, batch_size):
        batch_num = (i // batch_size) + 1
        batch_df = df.iloc[i:i + batch_size]
        
        # Update status
        status_text.text(f"📤 Processing BATCH {batch_num}/{total_batches} (rows {i+1}-{min(i+batch_size, total_rows)})...")
        
        # Upload batch with structured payload
        batch_uploaded, success, skipped = upload_batch_structured(
            components, batch_df, batch_num, progress_container, status_container, progress
        )
        
        if success:
            uploaded_count = get_collection_point_count(components)
            
            status_text.text(f"✅ BATCH {batch_num} UPLOADED! (Total in collection: {uploaded_count})")
            
            # Update progress bar
            progress_value = batch_num / total_batches
            progress_bar.progress(progress_value)
            
            # Update stats
            elapsed_time = time.time() - start_time
            speed = batch_uploaded / elapsed_time if elapsed_time > 0 else 0
            stats_text.text(f"📊 Progress: {batch_num}/{total_batches} batches ({progress_value*100:.1f}%) | Collection Total: {uploaded_count} | Speed: {speed:.1f} entries/sec")
            
        else:
            failed_batches += 1
            status_text.text(f"❌ BATCH {batch_num} FAILED!")
        
        # Small delay
        time.sleep(0.1)
    
    # Final results
    end_time = time.time()
    duration = end_time - start_time
    
    final_count = get_collection_point_count(components)
    status_text.text("🎉 STRUCTURED UPLOAD COMPLETED!")
    stats_text.text(f"""
    📊 Final Results:
    • Total batches: {total_batches}
    • Processed batches: {total_batches - failed_batches}
    • Failed batches: {failed_batches}
    • Total entries in collection: {final_count}
    • Success rate: {((total_batches - failed_batches) / total_batches * 100):.1f}%
    • Total time: {duration:.2f} seconds
    • Average speed: {batch_uploaded/duration:.2f} entries/second
    """)
    
    return uploaded_count

def reset_progress():
    """Reset upload progress."""
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    st.success("Progress reset! Next upload will start from the beginning.")

def get_collection_stats(components):
    """Get detailed collection statistics."""
    try:
        collection_info = components['qdrant_client'].get_collection(COLLECTION_NAME)
        return {
            'points_count': collection_info.points_count,
            'vectors_count': collection_info.vectors_count,
            'status': collection_info.status,
            'vector_size': collection_info.config.params.vectors.size
        }
    except Exception as e:
        return {'error': str(e)}

def clear_collection(components):
    """Clear the entire collection."""
    try:
        components['qdrant_client'].delete_collection(COLLECTION_NAME)
        st.success(f"✅ Collection '{COLLECTION_NAME}' deleted successfully!")
        reset_progress()
        return True
    except Exception as e:
        st.error(f"❌ Error clearing collection: {e}")
        return False

def main():
    """Main Streamlit app for structured upload tracking."""
    st.set_page_config(
        page_title="Smart INGRIS Structured Upload Tracker",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Smart INGRIS Structured Data Upload Tracker")
    st.markdown("**Upload with proper structured payload format (STATE, DISTRICT, Assessment_Year, etc.)**")
    
    # Load progress
    progress = load_progress()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Upload Settings")
        batch_size = st.slider("Batch Size", 10, 100, 25, help="Number of entries per batch")
        
        st.header("📊 Collection Status")
        if st.button("🔍 Check Collection Status"):
            try:
                components = initialize_components()
                if components:
                    stats = get_collection_stats(components)
                    if 'error' in stats:
                        st.error(f"Error: {stats['error']}")
                    else:
                        st.success(f"Collection has {stats['points_count']} points")
                        st.info(f"Status: {stats['status']}")
                        st.info(f"Vector Size: {stats.get('vector_size', 'Unknown')}")
                else:
                    st.error("Failed to connect to Qdrant")
            except Exception as e:
                st.error(f"Error checking collection: {e}")
        
        st.header("🔄 Progress Management")
        st.info(f"Last uploaded batch: {progress.get('last_uploaded_batch', 0)}")
        st.info(f"Total uploaded: {progress.get('total_uploaded', 0)}")
        
        if st.button("🔄 Reset Progress"):
            reset_progress()
            st.rerun()
        
        if st.button("🗑️ Clear Collection"):
            try:
                components = initialize_components()
                if components:
                    if clear_collection(components):
                        st.rerun()
            except Exception as e:
                st.error(f"Error clearing collection: {e}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Data Source")
        
        # Check if CSV exists
        csv_file = "ingris_rag_ready.csv"
        if not os.path.exists(csv_file):
            st.error(f"❌ CSV file not found: {csv_file}")
            st.info("💡 Please run 'python excel_to_csv_extractor.py' first to create the CSV file")
            return
        
        # Load CSV data
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Show sample data
            with st.expander("📋 Sample Data"):
                st.dataframe(df.head(10))
                
            # Show data structure info
            with st.expander("🔍 Data Structure Info"):
                st.write("**Key Fields Found:**")
                key_fields = ['state', 'district', 'year', 'serial_number']
                for field in key_fields:
                    if field in df.columns:
                        unique_count = df[field].nunique()
                        st.write(f"• {field}: {unique_count} unique values")
                    else:
                        st.write(f"• {field}: ❌ Not found")
                
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return
    
    with col2:
        st.header("🚀 Structured Upload Control")
        
        if st.button("🏗️ Setup Collection", use_container_width=True):
            with st.spinner("Setting up collection..."):
                components = initialize_components()
                if components and setup_collection(components):
                    st.success("✅ Collection setup complete!")
                else:
                    st.error("❌ Failed to setup collection")
        
        if st.button("📤 Start Structured Upload", use_container_width=True, type="primary"):
            # Initialize components
            components = initialize_components()
            if not components:
                st.error("❌ Failed to initialize components")
                return
            
            # Setup collection
            if not setup_collection(components):
                st.error("❌ Failed to setup collection")
                return
            
            # Start structured upload
            st.header("📊 Structured Upload Progress")
            uploaded_count = upload_data_with_structured_progress(components, df, batch_size)
            
            if uploaded_count > 0:
                st.success(f"🎉 Structured upload completed! {uploaded_count} new entries uploaded!")
                
                # Show final collection status
                try:
                    stats = get_collection_stats(components)
                    if 'points_count' in stats:
                        st.info(f"🔍 Collection now has {stats['points_count']} total points")
                        st.info(f"🔍 Vector size: {stats.get('vector_size', 'Unknown')}")
                except:
                    pass
                
                # Show next steps
                st.markdown("### 🎯 Next Steps")
                st.markdown("1. **Test the structured data**: Check if STATE, DISTRICT, Assessment_Year fields are properly stored")
                st.markdown("2. **Start the RAG Chatbot**: `streamlit run app6.py`")
                st.markdown("3. **Ask questions** about your INGRIS groundwater data")
                st.markdown("4. **Use the interactive map** to explore data by location")
            else:
                st.info("ℹ️ No new data to upload - everything is already uploaded!")
    
    # Footer
    st.markdown("---")
    st.markdown("**Smart INGRIS Structured Upload Tracker** - Upload with proper structured payload format!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Streamlit Upload Tracker - Real-time batch upload progress
Shows live progress of data upload to Qdrant
"""

import streamlit as st
import pandas as pd
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import time
import json
from datetime import datetime
import threading

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "ingris_groundwater_collection"
VECTOR_SIZE = 384

def initialize_components():
    """Initialize all required components."""
    try:
        # Qdrant client
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        
        # Sentence transformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
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

def setup_collection(components):
    """Set up Qdrant collection."""
    try:
        collections = components['qdrant_client'].get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            components['qdrant_client'].create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            return True
        return True
    except Exception as e:
        st.error(f"Failed to setup collection: {e}")
        return False

def upload_batch(components, batch_df, batch_num, progress_container, status_container):
    """Upload a single batch and update progress."""
    try:
        points = []
        for j, (_, row) in enumerate(batch_df.iterrows()):
            text = row.get('combined_text', '')
            if text and len(text.strip()) > 0:
                # Generate embedding
                embedding = components['model'].encode([text])[0]
                
                # Create payload
                payload = row.to_dict()
                
                # Create point with integer ID
                point_id = (batch_num - 1) * len(batch_df) + j + 1
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
            return len(points), True
        else:
            return 0, False
            
    except Exception as e:
        status_container.error(f"❌ Error uploading batch {batch_num}: {e}")
        return 0, False

def upload_data_with_progress(components, df, batch_size=25):
    """Upload data with real-time progress tracking."""
    total_rows = len(df)
    total_batches = (total_rows + batch_size - 1) // batch_size
    
    # Create progress containers
    progress_container = st.container()
    status_container = st.container()
    stats_container = st.container()
    
    # Initialize progress
    progress_bar = progress_container.progress(0)
    status_text = status_container.empty()
    stats_text = stats_container.empty()
    
    uploaded_count = 0
    failed_batches = 0
    start_time = time.time()
    
    status_text.text(f"🚀 Starting upload of {total_rows} rows in {total_batches} batches...")
    
    for i in range(0, total_rows, batch_size):
        batch_num = (i // batch_size) + 1
        batch_df = df.iloc[i:i + batch_size]
        
        # Update status
        status_text.text(f"📤 Uploading BATCH {batch_num}/{total_batches} (rows {i+1}-{min(i+batch_size, total_rows)})...")
        
        # Upload batch
        batch_uploaded, success = upload_batch(components, batch_df, batch_num, progress_container, status_container)
        
        if success:
            uploaded_count += batch_uploaded
            status_text.text(f"✅ BATCH {batch_num} UPLOADED! ({batch_uploaded} entries)")
            
            # Update progress bar
            progress = batch_num / total_batches
            progress_bar.progress(progress)
            
            # Update stats
            elapsed_time = time.time() - start_time
            speed = uploaded_count / elapsed_time if elapsed_time > 0 else 0
            stats_text.text(f"📊 Progress: {batch_num}/{total_batches} batches ({progress*100:.1f}%) | Uploaded: {uploaded_count} | Speed: {speed:.1f} entries/sec")
            
        else:
            failed_batches += 1
            status_text.text(f"❌ BATCH {batch_num} FAILED!")
        
        # Small delay
        time.sleep(0.1)
    
    # Final results
    end_time = time.time()
    duration = end_time - start_time
    
    status_text.text("🎉 UPLOAD COMPLETED!")
    stats_text.text(f"""
    📊 Final Results:
    • Total batches: {total_batches}
    • Successful batches: {total_batches - failed_batches}
    • Failed batches: {failed_batches}
    • Total entries uploaded: {uploaded_count}
    • Success rate: {((total_batches - failed_batches) / total_batches * 100):.1f}%
    • Total time: {duration:.2f} seconds
    • Average speed: {uploaded_count/duration:.2f} entries/second
    """)
    
    return uploaded_count

def main():
    """Main Streamlit app for upload tracking."""
    st.set_page_config(
        page_title="INGRIS Upload Tracker",
        page_icon="📤",
        layout="wide"
    )
    
    st.title("📤 INGRIS Data Upload Tracker")
    st.markdown("**Real-time progress tracking for CSV data upload to Qdrant**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Upload Settings")
        batch_size = st.slider("Batch Size", 10, 100, 25, help="Number of entries per batch")
        
        st.header("📊 Collection Status")
        if st.button("🔍 Check Collection Status"):
            try:
                components = initialize_components()
                if components:
                    collection_info = components['qdrant_client'].get_collection(COLLECTION_NAME)
                    st.success(f"Collection has {collection_info.points_count} points")
                else:
                    st.error("Failed to connect to Qdrant")
            except Exception as e:
                st.error(f"Error checking collection: {e}")
        
        if st.button("🗑️ Clear Collection"):
            try:
                components = initialize_components()
                if components:
                    components['qdrant_client'].delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=models.FilterSelector(filter=models.Filter(must=[]))
                    )
                    st.success("Collection cleared!")
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
                
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return
    
    with col2:
        st.header("🚀 Upload Control")
        
        if st.button("🏗️ Setup Collection", use_container_width=True):
            with st.spinner("Setting up collection..."):
                components = initialize_components()
                if components and setup_collection(components):
                    st.success("✅ Collection setup complete!")
                else:
                    st.error("❌ Failed to setup collection")
        
        if st.button("📤 Start Upload", use_container_width=True, type="primary"):
            # Initialize components
            components = initialize_components()
            if not components:
                st.error("❌ Failed to initialize components")
                return
            
            # Setup collection
            if not setup_collection(components):
                st.error("❌ Failed to setup collection")
                return
            
            # Start upload
            st.header("📊 Upload Progress")
            uploaded_count = upload_data_with_progress(components, df, batch_size)
            
            if uploaded_count > 0:
                st.success(f"🎉 Upload completed! {uploaded_count} entries uploaded successfully!")
                
                # Show final collection status
                try:
                    collection_info = components['qdrant_client'].get_collection(COLLECTION_NAME)
                    st.info(f"🔍 Collection now has {collection_info.points_count} total points")
                except:
                    pass
                
                # Show next steps
                st.markdown("### 🎯 Next Steps")
                st.markdown("1. **Start the RAG Chatbot**: `streamlit run app5_csv.py`")
                st.markdown("2. **Ask questions** about your INGRIS groundwater data")
                st.markdown("3. **Use filters** to search by state, district, or year")
            else:
                st.error("❌ Upload failed! Please check the logs above.")
    
    # Footer
    st.markdown("---")
    st.markdown("**INGRIS Upload Tracker** - Real-time progress monitoring for data uploads")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Smart Upload Tracker - Resume from where it left off
Tracks progress and avoids re-uploading existing embeddings
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

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "ingris_groundwater_collection"
VECTOR_SIZE = 384
PROGRESS_FILE = "upload_progress.json"

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

def calculate_progress_from_collection_size(components, total_rows):
    """Calculate progress based on existing collection size."""
    try:
        current_count = get_collection_point_count(components)
        
        # If we have data, estimate progress
        if current_count > 0:
            # Estimate which batch we should start from
            estimated_batch = (current_count // 25) + 1  # Assuming 25 per batch
            return max(1, estimated_batch), current_count
        else:
            return 1, 0
    except Exception as e:
        st.warning(f"Could not calculate progress: {e}")
        return 1, 0

def upload_batch_smart(components, batch_df, batch_num, progress_container, status_container, progress):
    """Upload a batch with smart duplicate checking."""
    try:
        batch_size = len(batch_df)
        batch_start_id = (batch_num - 1) * batch_size + 1
        
        points = []
        
        for j, (_, row) in enumerate(batch_df.iterrows()):
            text = row.get('combined_text', '')
            if text and len(text.strip()) > 0:
                # Generate embedding
                embedding = components['model'].encode([text])[0]
                
                # Create payload
                payload = row.to_dict()
                
                # Create point with integer ID
                point_id = batch_start_id + j
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
                points.append(point)
        
        # Upload batch (Qdrant will handle duplicates automatically)
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

def upload_data_with_smart_progress(components, df, batch_size=25):
    """Upload data with smart progress tracking and resume capability."""
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
        status_text.text(f"🚀 Starting upload of {total_rows} rows in {total_batches} batches...")
    
    for i in range((start_batch - 1) * batch_size, total_rows, batch_size):
        batch_num = (i // batch_size) + 1
        batch_df = df.iloc[i:i + batch_size]
        
        # Update status
        status_text.text(f"📤 Processing BATCH {batch_num}/{total_batches} (rows {i+1}-{min(i+batch_size, total_rows)})...")
        
        # Upload batch with smart checking
        batch_uploaded, success, skipped = upload_batch_smart(
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
    status_text.text("🎉 UPLOAD COMPLETED!")
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
            'status': collection_info.status
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    """Main Streamlit app for smart upload tracking."""
    st.set_page_config(
        page_title="Smart INGRIS Upload Tracker",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Smart INGRIS Data Upload Tracker")
    st.markdown("**Resume from where you left off - No duplicate uploads!**")
    
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
                    # Delete all points
                    components['qdrant_client'].delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=Filter(must=[])
                    )
                    st.success("Collection cleared!")
                    reset_progress()
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
        st.header("🚀 Smart Upload Control")
        
        if st.button("🏗️ Setup Collection", use_container_width=True):
            with st.spinner("Setting up collection..."):
                components = initialize_components()
                if components and setup_collection(components):
                    st.success("✅ Collection setup complete!")
                else:
                    st.error("❌ Failed to setup collection")
        
        if st.button("📤 Start Smart Upload", use_container_width=True, type="primary"):
            # Initialize components
            components = initialize_components()
            if not components:
                st.error("❌ Failed to initialize components")
                return
            
            # Setup collection
            if not setup_collection(components):
                st.error("❌ Failed to setup collection")
                return
            
            # Start smart upload
            st.header("📊 Smart Upload Progress")
            uploaded_count = upload_data_with_smart_progress(components, df, batch_size)
            
            if uploaded_count > 0:
                st.success(f"🎉 Smart upload completed! {uploaded_count} new entries uploaded!")
                
                # Show final collection status
                try:
                    stats = get_collection_stats(components)
                    if 'points_count' in stats:
                        st.info(f"🔍 Collection now has {stats['points_count']} total points")
                except:
                    pass
                
                # Show next steps
                st.markdown("### 🎯 Next Steps")
                st.markdown("1. **Start the RAG Chatbot**: `streamlit run app6.py`")
                st.markdown("2. **Ask questions** about your INGRIS groundwater data")
                st.markdown("3. **Use the interactive map** to explore data by location")
            else:
                st.info("ℹ️ No new data to upload - everything is already uploaded!")
    
    # Footer
    st.markdown("---")
    st.markdown("**Smart INGRIS Upload Tracker** - Resume from where you left off, no duplicates!")

if __name__ == "__main__":
    main()

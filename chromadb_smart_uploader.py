#!/usr/bin/env python3
"""
Smart ChromaDB Upload Tracker for Groundwater Data
Uploads complete data in the same format as Qdrant
"""

import streamlit as st
import pandas as pd
import chromadb
from chromadb.config import Settings
import json
import uuid
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import time
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="ChromaDB Smart Upload Tracker",
    page_icon="💧",
    layout="wide"
)

class ChromaDBSmartUploader:
    def __init__(self):
        self.client = None
        self.collection = None
        self.model = None
        self.upload_stats = {
            'total_processed': 0,
            'successful_uploads': 0,
            'failed_uploads': 0,
            'start_time': None,
            'end_time': None
        }
    
    def initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.client.get_or_create_collection(
                name="ingris_groundwater_collection",
                metadata={"description": "Groundwater estimation data collection"}
            )
            return True
        except Exception as e:
            st.error(f"Failed to initialize ChromaDB: {e}")
            return False
    
    def initialize_model(self):
        """Initialize the embedding model"""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            return True
        except Exception as e:
            st.error(f"Failed to initialize embedding model: {e}")
            return False
    
    def create_combined_text(self, row):
        """Create combined text similar to Qdrant format"""
        combined_parts = []
        
        # Add all relevant fields
        fields_to_include = [
            'serial_number', 'state', 'district', 'watershed_district',
            'rainfall_mm', 'total_geographical_area_ha', 'ground_water_recharge_ham',
            'inflows_and_outflows_ham', 'annual_ground_water_recharge_ham',
            'environmental_flows_ham', 'annual_extractable_ground_water_resource_ham',
            'ground_water_extraction_for_all_uses_ham', 'stage_of_ground_water_extraction_',
            'categorization_of_assessment_unit', 'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham',
            'net_annual_ground_water_availability_for_future_use_ham',
            'instorage_unconfined_ground_water_resourcesham',
            'total_ground_water_availability_in_unconfined_aquifier_ham',
            'total_ground_water_availability_in_the_area_ham',
            'year', 'block', 'village', 'watershed_category'
        ]
        
        for field in fields_to_include:
            if field in row and pd.notna(row[field]) and str(row[field]).strip() != '':
                combined_parts.append(f"{field}: {row[field]}")
        
        return " | ".join(combined_parts)
    
    def create_payload(self, row):
        """Create payload similar to Qdrant format"""
        payload = {}
        
        # Map CSV columns to Qdrant-style payload
        field_mapping = {
            'serial_number': 'serial_number',
            'state': 'STATE',
            'district': 'DISTRICT', 
            'Assessment_Year': 'Assessment_Year',
            'watershed_district': 'watershed_district',
            'rainfall_mm': 'rainfall_mm',
            'total_geographical_area_ha': 'total_geographical_area_ha',
            'ground_water_recharge_ham': 'ground_water_recharge_ham',
            'inflows_and_outflows_ham': 'inflows_and_outflows_ham',
            'annual_ground_water_recharge_ham': 'annual_ground_water_recharge_ham',
            'environmental_flows_ham': 'environmental_flows_ham',
            'annual_extractable_ground_water_resource_ham': 'annual_extractable_ground_water_resource_ham',
            'ground_water_extraction_for_all_uses_ham': 'ground_water_extraction_for_all_uses_ham',
            'stage_of_ground_water_extraction_': 'stage_of_ground_water_extraction_',
            'categorization_of_assessment_unit': 'categorization_of_assessment_unit',
            'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham': 'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham',
            'net_annual_ground_water_availability_for_future_use_ham': 'net_annual_ground_water_availability_for_future_use_ham',
            'instorage_unconfined_ground_water_resourcesham': 'instorage_unconfined_ground_water_resourcesham',
            'total_ground_water_availability_in_unconfined_aquifier_ham': 'total_ground_water_availability_in_unconfined_aquifier_ham',
            'total_ground_water_availability_in_the_area_ham': 'total_ground_water_availability_in_the_area_ham',
            'source_file': 'source_file',
            'block': 'block',
            'village': 'village',
            'watershed_category': 'watershed_category',
            'taluk': 'taluk',
            'tehsil': 'tehsil',
            'mandal': 'mandal',
            'valley': 'valley',
            'assessment_unit': 'assessment_unit',
            'firka': 'firka',
            'island': 'island',
            'pre_monsoon_of_gw_trend': 'pre_monsoon_of_gw_trend',
            'post_monsoon_of_gw_trend': 'post_monsoon_of_gw_trend',
            'quality_tagging': 'quality_tagging',
            'additional_potential_resources_under_specific_conditionsham': 'additional_potential_resources_under_specific_conditionsham',
            'coastal_areas': 'coastal_areas',
            'dynamic_confined_ground_water_resourcesham': 'dynamic_confined_ground_water_resourcesham',
            'instorage_confined_ground_water_resourcesham': 'instorage_confined_ground_water_resourcesham',
            'total_confined_ground_water_resources_ham': 'total_confined_ground_water_resources_ham',
            'dynamic_semi_confined_ground_water_resources_ham': 'dynamic_semi_confined_ground_water_resources_ham',
            'instorage_semi_confined_ground_water_resources_ham': 'instorage_semi_confined_ground_water_resources_ham',
            'total_semiconfined_ground_water_resources_ham': 'total_semiconfined_ground_water_resources_ham'
        }
        
        # Add mapped fields
        for csv_field, qdrant_field in field_mapping.items():
            if csv_field in row and pd.notna(row[csv_field]):
                value = row[csv_field]
                if isinstance(value, (int, float)):
                    payload[qdrant_field] = value
                else:
                    payload[qdrant_field] = str(value)
            else:
                payload[qdrant_field] = "N/A"
        
        # Add combined text
        payload['combined_text'] = self.create_combined_text(row)
        
        # Add original data as JSON string
        original_data = {}
        for col in row.index:
            if pd.notna(row[col]):
                original_data[col] = str(row[col])
            else:
                original_data[col] = None
        payload['original_data'] = json.dumps(original_data, ensure_ascii=False)
        
        return payload
    
    def upload_batch(self, df_batch, batch_num, total_batches):
        """Upload a batch of data to ChromaDB"""
        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for idx, row in df_batch.iterrows():
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                
                # Create payload
                payload = self.create_payload(row)
                metadatas.append(payload)
                
                # Use combined text as document
                documents.append(payload['combined_text'])
                
                # Generate embedding
                embedding = self.model.encode(payload['combined_text']).tolist()
                embeddings.append(embedding)
            
            # Upload to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            return len(ids)
            
        except Exception as e:
            st.error(f"Error uploading batch {batch_num}: {e}")
            return 0
    
    def upload_complete_data(self, csv_file_path, batch_size=100):
        """Upload complete CSV data to ChromaDB"""
        try:
            # Read CSV
            st.info("Reading CSV file...")
            df = pd.read_csv(csv_file_path)
            st.success(f"Loaded {len(df)} records from CSV")
            
            # Initialize components
            if not self.initialize_chromadb():
                return False
            
            if not self.initialize_model():
                return False
            
            # Clear existing collection
            st.info("Clearing existing collection...")
            try:
                self.client.delete_collection("ingris_groundwater_collection")
                self.collection = self.client.create_collection(
                    name="ingris_groundwater_collection",
                    metadata={"description": "Groundwater estimation data collection"}
                )
            except:
                pass
            
            # Upload in batches
            total_records = len(df)
            total_batches = (total_records + batch_size - 1) // batch_size
            
            st.info(f"Starting upload: {total_records} records in {total_batches} batches")
            
            self.upload_stats['start_time'] = datetime.now()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            successful_uploads = 0
            
            for i in range(0, total_records, batch_size):
                batch_num = (i // batch_size) + 1
                df_batch = df.iloc[i:i + batch_size]
                
                status_text.text(f"Processing batch {batch_num}/{total_batches} ({len(df_batch)} records)")
                
                uploaded_count = self.upload_batch(df_batch, batch_num, total_batches)
                successful_uploads += uploaded_count
                
                # Update progress
                progress = min((i + batch_size) / total_records, 1.0)
                progress_bar.progress(progress)
                
                # Update stats
                self.upload_stats['total_processed'] = i + len(df_batch)
                self.upload_stats['successful_uploads'] = successful_uploads
                self.upload_stats['failed_uploads'] = (i + len(df_batch)) - successful_uploads
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
            
            self.upload_stats['end_time'] = datetime.now()
            
            st.success(f"Upload completed! {successful_uploads} records uploaded successfully")
            return True
            
        except Exception as e:
            st.error(f"Upload failed: {e}")
            return False
    
    def get_collection_stats(self):
        """Get collection statistics"""
        try:
            if not self.collection:
                self.initialize_chromadb()
            
            count = self.collection.count()
            return count
        except Exception as e:
            st.error(f"Error getting collection stats: {e}")
            return 0
    
    def search_test(self, query, limit=5):
        """Test search functionality"""
        try:
            if not self.collection:
                self.initialize_chromadb()
            
            if not self.model:
                self.initialize_model()
            
            # Generate query embedding
            query_embedding = self.model.encode(query).tolist()
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            return results
        except Exception as e:
            st.error(f"Search test failed: {e}")
            return None

def main():
    st.title("💧 ChromaDB Smart Upload Tracker")
    st.markdown("Upload complete groundwater data in Qdrant-compatible format")
    
    # Initialize uploader
    if 'uploader' not in st.session_state:
        st.session_state.uploader = ChromaDBSmartUploader()
    
    uploader = st.session_state.uploader
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📊 Upload Controls")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload the ingris_rag_ready_complete.csv file"
        )
        
        # Batch size
        batch_size = st.slider(
            "Batch Size",
            min_value=50,
            max_value=500,
            value=100,
            help="Number of records to process in each batch"
        )
        
        # Upload button
        if st.button("🚀 Start Upload", type="primary"):
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with open("temp_upload.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Start upload
                with st.spinner("Uploading data..."):
                    success = uploader.upload_complete_data("temp_upload.csv", batch_size)
                
                if success:
                    st.success("✅ Upload completed successfully!")
                else:
                    st.error("❌ Upload failed!")
                
                # Clean up temp file
                if os.path.exists("temp_upload.csv"):
                    os.remove("temp_upload.csv")
            else:
                st.warning("Please upload a CSV file first")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📈 Collection Statistics")
        
        # Get collection stats
        if st.button("🔄 Refresh Stats"):
            count = uploader.get_collection_stats()
            st.metric("Total Records", count)
        
        # Show upload stats if available
        if uploader.upload_stats['start_time']:
            st.subheader("📊 Upload Statistics")
            
            col1_1, col1_2, col1_3 = st.columns(3)
            with col1_1:
                st.metric("Total Processed", uploader.upload_stats['total_processed'])
            with col1_2:
                st.metric("Successful", uploader.upload_stats['successful_uploads'])
            with col1_3:
                st.metric("Failed", uploader.upload_stats['failed_uploads'])
            
            if uploader.upload_stats['end_time']:
                duration = uploader.upload_stats['end_time'] - uploader.upload_stats['start_time']
                st.metric("Duration", str(duration).split('.')[0])
    
    with col2:
        st.header("🔍 Search Test")
        
        # Search test
        test_query = st.text_input(
            "Test Query",
            value="groundwater estimation in Karnataka",
            help="Enter a query to test the search functionality"
        )
        
        if st.button("🔍 Test Search"):
            if test_query:
                with st.spinner("Searching..."):
                    results = uploader.search_test(test_query, limit=5)
                
                if results and results['ids']:
                    st.success(f"Found {len(results['ids'][0])} results")
                    
                    for i, (doc_id, distance, metadata) in enumerate(zip(
                        results['ids'][0],
                        results['distances'][0],
                        results['metadatas'][0]
                    )):
                        with st.expander(f"Result {i+1} (Distance: {distance:.3f})"):
                            st.write(f"**State:** {metadata.get('STATE', 'N/A')}")
                            st.write(f"**District:** {metadata.get('DISTRICT', 'N/A')}")
                            st.write(f"**Year:** {metadata.get('Assessment_Year', 'N/A')}")
                            st.write(f"**Serial Number:** {metadata.get('serial_number', 'N/A')}")
                            st.write(f"**Document:** {metadata.get('combined_text', 'N/A')[:200]}...")
                else:
                    st.warning("No results found")
            else:
                st.warning("Please enter a test query")
    
    # Footer
    st.markdown("---")
    st.markdown("**ChromaDB Smart Upload Tracker** - Upload complete groundwater data in Qdrant-compatible format")

if __name__ == "__main__":
    main()

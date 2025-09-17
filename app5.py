#!/usr/bin/env python3
"""
INGRIS RAG Chatbot - Streamlit Application
A RAG-based chatbot for querying INGRIS (Integrated Groundwater Resource Information System) datasets.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import json
import re
import hashlib
import uuid
from rank_bm25 import BM25Okapi
import spacy
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Import our data processor
from ingris_data_processor import INGRISDataProcessor

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Constants
COLLECTION_NAME = "ingris_groundwater_collection"
VECTOR_SIZE = 384  # Based on all-MiniLM-L6-v2 model
MIN_SIMILARITY_SCORE = 0.3

# Global variables
@st.cache_resource
def initialize_components():
    """Initialize all required components."""
    components = {}
    
    # Initialize Qdrant client
    try:
        components['qdrant_client'] = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60
        )
        st.success("✅ Qdrant client initialized")
    except Exception as e:
        st.error(f"❌ Failed to initialize Qdrant client: {str(e)}")
        return None
    
    # Initialize Sentence Transformer
    try:
        components['model'] = SentenceTransformer("all-MiniLM-L6-v2", device=torch.device('cpu'))
        st.success("✅ Sentence Transformer initialized")
    except Exception as e:
        st.error(f"❌ Failed to initialize Sentence Transformer: {str(e)}")
        return None
    
    # Initialize spaCy
    try:
        components['nlp'] = spacy.load("en_core_web_sm")
        st.success("✅ spaCy NLP model initialized")
    except Exception as e:
        st.error(f"❌ Failed to initialize spaCy: {str(e)}")
        return None
    
    # Initialize Gemini
    try:
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            components['gemini_model'] = genai.GenerativeModel('gemini-1.5-flash')
            st.success("✅ Gemini API initialized")
        else:
            st.warning("⚠️ Gemini API key not found - some features may be limited")
            components['gemini_model'] = None
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini API: {str(e)}")
        components['gemini_model'] = None
    
    return components

@st.cache_data
def load_ingris_data():
    """Load and process INGRIS data."""
    processor = INGRISDataProcessor()
    
    # Try to load existing processed data first
    processed_file = "ingris_processed_data.json"
    if Path(processed_file).exists():
        st.info("📁 Loading existing processed INGRIS data...")
        data = processor.load_processed_data(processed_file)
    else:
        st.info("🔄 Processing INGRIS datasets (this may take a few minutes)...")
        data = processor.process_all_datasets()
        processor.save_processed_data(processed_file)
    
    return data

def setup_collection(components):
    """Set up Qdrant collection for INGRIS data."""
    try:
        # Check if collection exists
        collections = components['qdrant_client'].get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            # Create collection
            components['qdrant_client'].create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            st.success(f"✅ Created collection: {COLLECTION_NAME}")
        else:
            st.info(f"ℹ️ Collection {COLLECTION_NAME} already exists")
        
        return True
    except Exception as e:
        st.error(f"❌ Failed to setup collection: {str(e)}")
        return False

def upload_ingris_data_to_qdrant(components, ingris_data):
    """Upload INGRIS data to Qdrant vector database."""
    try:
        st.info("📤 Uploading INGRIS data to Qdrant...")
        
        points = []
        for idx, entry in enumerate(ingris_data):
            # Create text for embedding
            text_content = entry.get('text_content', '')
            if not text_content:
                continue
            
            # Generate embedding
            embedding = components['model'].encode([text_content])[0].tolist()
            
            # Create point
            point = {
                "id": idx,
                "vector": embedding,
                "payload": {
                    "text": text_content,
                    "state": entry.get('state', ''),
                    "district": entry.get('district', ''),
                    "year": entry.get('year', ''),
                    "assessment_unit": entry.get('assessment_unit', ''),
                    "raw_data": entry.get('raw_data', {}),
                    "metadata": entry.get('metadata', {})
                }
            }
            points.append(point)
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            components['qdrant_client'].upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
            st.progress((i + len(batch)) / len(points))
        
        st.success(f"✅ Uploaded {len(points)} INGRIS entries to Qdrant")
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to upload data to Qdrant: {str(e)}")
        return False

def search_ingris_data(components, query: str, year: str = None, state: str = None, district: str = None, limit: int = 10):
    """Search INGRIS data using hybrid search."""
    try:
        # Build filter
        filter_conditions = []
        if year:
            filter_conditions.append(FieldCondition(key="year", match=MatchValue(value=year)))
        if state:
            filter_conditions.append(FieldCondition(key="state", match=MatchValue(value=state)))
        if district:
            filter_conditions.append(FieldCondition(key="district", match=MatchValue(value=district)))
        
        qdrant_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Generate query embedding
        query_embedding = components['model'].encode([query])[0]
        
        # Search Qdrant
        search_results = components['qdrant_client'].search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True
        )
        
        return search_results
        
    except Exception as e:
        st.error(f"❌ Search failed: {str(e)}")
        return []

def generate_answer(components, query: str, search_results: List[Dict], year: str = None, state: str = None, district: str = None):
    """Generate answer using Gemini based on search results."""
    if not components['gemini_model']:
        return "❌ Gemini API not available. Please check your API key configuration."
    
    if not search_results:
        return "I couldn't find relevant information in the INGRIS datasets to answer your question."
    
    # Prepare context from search results
    context_parts = []
    for i, result in enumerate(search_results[:5], 1):
        payload = result.payload
        context_parts.append(f"Result {i}:")
        context_parts.append(f"State: {payload.get('state', 'N/A')}")
        context_parts.append(f"District: {payload.get('district', 'N/A')}")
        context_parts.append(f"Year: {payload.get('year', 'N/A')}")
        context_parts.append(f"Assessment Unit: {payload.get('assessment_unit', 'N/A')}")
        context_parts.append(f"Data: {payload.get('text', 'N/A')}")
        context_parts.append("---")
    
    context = "\n".join(context_parts)
    
    # Create prompt
    location_info = ""
    if state and district:
        location_info = f" for {district}, {state}"
    elif state:
        location_info = f" for {state}"
    
    year_info = f" for the year {year}" if year else ""
    
    prompt = f"""You are an expert groundwater analyst specializing in INGRIS (Integrated Groundwater Resource Information System) data. 
    
Provide a comprehensive answer based on the following INGRIS groundwater assessment data{location_info}{year_info}:

{context}

Question: {query}

Instructions:
- Base your answer ONLY on the provided INGRIS data
- Include specific numbers, percentages, and measurements when available
- Mention the state, district, and year of the data you're referencing
- If the data doesn't contain the answer, clearly state that
- Focus on groundwater recharge, extraction, availability, and stage of extraction
- Be precise and technical in your analysis

Answer:"""
    
    try:
        response = components['gemini_model'].generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="INGRIS RAG Chatbot",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("💧 INGRIS RAG Chatbot")
    st.markdown("**Integrated Groundwater Resource Information System - RAG-Powered Query Interface**")
    
    # Initialize components
    with st.spinner("Initializing components..."):
        components = initialize_components()
    
    if not components:
        st.error("❌ Failed to initialize components. Please check your configuration.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Data loading section
        st.subheader("📊 Data Management")
        if st.button("🔄 Reload INGRIS Data", help="Process and reload all INGRIS datasets"):
            with st.spinner("Processing INGRIS data..."):
                processor = INGRISDataProcessor()
                data = processor.process_all_datasets()
                processor.save_processed_data()
                st.success("✅ Data reloaded successfully!")
                st.rerun()
        
        # Collection setup
        if st.button("🏗️ Setup Vector Database", help="Create collection and upload data"):
            with st.spinner("Setting up vector database..."):
                if setup_collection(components):
                    ingris_data = load_ingris_data()
                    if upload_ingris_data_to_qdrant(components, ingris_data):
                        st.success("✅ Vector database setup complete!")
                    else:
                        st.error("❌ Failed to upload data")
                else:
                    st.error("❌ Failed to setup collection")
        
        # Filters
        st.subheader("🔍 Search Filters")
        year_filter = st.selectbox(
            "Year",
            ["All"] + ["2016-2017", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"],
            help="Filter by assessment year"
        )
        
        state_filter = st.text_input(
            "State",
            placeholder="e.g., Karnataka, Maharashtra",
            help="Filter by state name"
        )
        
        district_filter = st.text_input(
            "District",
            placeholder="e.g., Bangalore, Mumbai",
            help="Filter by district name"
        )
        
        # Search parameters
        st.subheader("⚙️ Search Parameters")
        max_results = st.slider("Max Results", 5, 20, 10, help="Maximum number of results to return")
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.3, 0.1, help="Minimum similarity score for results")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask About INGRIS Data")
        
        # Query input
        query = st.text_area(
            "Enter your question about groundwater data:",
            placeholder="e.g., What is the groundwater extraction stage in Karnataka?",
            height=100,
            help="Ask questions about groundwater recharge, extraction, availability, etc."
        )
        
        # Search button
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if not query.strip():
                st.warning("⚠️ Please enter a question")
            else:
                with st.spinner("Searching INGRIS data..."):
                    # Apply filters
                    year = year_filter if year_filter != "All" else None
                    state = state_filter.strip() if state_filter.strip() else None
                    district = district_filter.strip() if district_filter.strip() else None
                    
                    # Search
                    search_results = search_ingris_data(
                        components, 
                        query, 
                        year=year, 
                        state=state, 
                        district=district, 
                        limit=max_results
                    )
                    
                    if search_results:
                        # Filter by similarity threshold
                        filtered_results = [r for r in search_results if r.score >= similarity_threshold]
                        
                        if filtered_results:
                            # Generate answer
                            answer = generate_answer(
                                components, 
                                query, 
                                filtered_results, 
                                year=year, 
                                state=state, 
                                district=district
                            )
                            
                            st.subheader("📝 Answer")
                            st.markdown(answer)
                            
                            # Show search results
                            st.subheader(f"🔍 Search Results ({len(filtered_results)} found)")
                            for i, result in enumerate(filtered_results, 1):
                                with st.expander(f"Result {i} (Score: {result.score:.3f})"):
                                    payload = result.payload
                                    st.write(f"**State:** {payload.get('state', 'N/A')}")
                                    st.write(f"**District:** {payload.get('district', 'N/A')}")
                                    st.write(f"**Year:** {payload.get('year', 'N/A')}")
                                    st.write(f"**Assessment Unit:** {payload.get('assessment_unit', 'N/A')}")
                                    st.write(f"**Data:** {payload.get('text', 'N/A')}")
                        else:
                            st.warning(f"⚠️ No results found above similarity threshold ({similarity_threshold})")
                    else:
                        st.warning("⚠️ No results found for your query")
    
    with col2:
        st.subheader("📊 Data Statistics")
        
        # Load and display data stats
        try:
            ingris_data = load_ingris_data()
            if ingris_data:
                # Calculate statistics
                states = set(entry.get('state', 'Unknown') for entry in ingris_data)
                districts = set(entry.get('district', 'Unknown') for entry in ingris_data)
                years = set(entry.get('year', 'Unknown') for entry in ingris_data)
                
                st.metric("Total Entries", len(ingris_data))
                st.metric("States", len(states))
                st.metric("Districts", len(districts))
                st.metric("Years", len(years))
                
                # Show sample data
                st.subheader("📋 Sample Data")
                if ingris_data:
                    sample = ingris_data[0]
                    st.write(f"**State:** {sample.get('state', 'N/A')}")
                    st.write(f"**District:** {sample.get('district', 'N/A')}")
                    st.write(f"**Year:** {sample.get('year', 'N/A')}")
                    st.write(f"**Assessment Unit:** {sample.get('assessment_unit', 'N/A')}")
            else:
                st.warning("⚠️ No data loaded. Please reload data first.")
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
        
        # Example queries
        st.subheader("💡 Example Queries")
        example_queries = [
            "What is the groundwater extraction stage in Karnataka?",
            "Show me groundwater recharge data for Maharashtra",
            "Which districts have the highest groundwater extraction?",
            "What is the annual groundwater availability in Tamil Nadu?",
            "Compare groundwater data between 2023-2024 and 2024-2025"
        ]
        
        for i, example in enumerate(example_queries, 1):
            if st.button(f"{i}. {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.query = example
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("**INGRIS RAG Chatbot** - Powered by Qdrant Vector Search and Google Gemini AI")
    st.markdown("💡 **Tip:** Use specific state and district names for better results")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
INGRIS Groundwater Chatbot using ChromaDB
Uses the complete INGRIS dataset with proper 768-dimensional embeddings
"""

import streamlit as st
import pandas as pd
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import json

# Configuration
COLLECTION_NAME = "ingris_groundwater_collection"
CSV_FILE = "ingris_rag_ready_complete.csv"
EMBEDDING_MODEL = "all-mpnet-base-v2"  # 768 dimensions
GEMINI_MODEL = "models/gemini-2.0-flash"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chromadb_client' not in st.session_state:
    st.session_state.chromadb_client = None
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None

def initialize_components():
    """Initialize ChromaDB, embedding model, and Gemini"""
    try:
        # Initialize ChromaDB
        if st.session_state.chromadb_client is None:
            st.session_state.chromadb_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
            st.success("✅ ChromaDB client initialized")
        
        # Get collection
        if st.session_state.collection is None:
            st.session_state.collection = st.session_state.chromadb_client.get_collection(
                name=COLLECTION_NAME
            )
            st.success(f"✅ Collection '{COLLECTION_NAME}' loaded")
        
        # Initialize embedding model
        if st.session_state.embedding_model is None:
            st.session_state.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            st.success(f"✅ Embedding model '{EMBEDDING_MODEL}' loaded")
        
        # Initialize Gemini
        if st.session_state.gemini_model is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                st.session_state.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
                st.success(f"✅ Gemini model '{GEMINI_MODEL}' initialized")
            else:
                st.warning("⚠️ GEMINI_API_KEY not found. Some features will be limited.")
        
        return True
    except Exception as e:
        st.error(f"❌ Error initializing components: {str(e)}")
        return False

def search_groundwater_data(query: str, n_results: int = 20) -> List[Dict[str, Any]]:
    """Search groundwater data using ChromaDB with enhanced filtering"""
    try:
        if st.session_state.collection is None or st.session_state.embedding_model is None:
            return []
        
        # Generate query embedding
        query_embedding = st.session_state.embedding_model.encode([query])[0].tolist()
        
        # Search in ChromaDB with more results for better coverage
        results = st.session_state.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results and filter by similarity threshold
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                similarity = 1 - distance
                
                # Filter by minimum similarity threshold
                if similarity >= 0.1:  # Only include results with similarity >= 0.1
                    formatted_results.append({
                        'document': doc,
                        'metadata': metadata,
                        'distance': distance,
                        'similarity': similarity
                    })
        
        # Sort by similarity (highest first)
        formatted_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return formatted_results
    except Exception as e:
        st.error(f"❌ Error searching data: {str(e)}")
        return []

def generate_response(query: str, search_results: List[Dict[str, Any]]) -> str:
    """Generate response using Gemini with structured format"""
    try:
        if st.session_state.gemini_model and search_results:
            # Prepare context from search results
            context_parts = []
            for i, result in enumerate(search_results[:10], 1):  # Use top 10 results
                context_parts.append(f"=== RESULT {i} ===")
                context_parts.append(f"Document: {result['document']}")
                if result['metadata']:
                    context_parts.append(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
                context_parts.append(f"Similarity: {result['similarity']:.3f}")
                context_parts.append("")
            
            context = "\n".join(context_parts)
            
            # Create detailed prompt for structured response
            prompt = f"""
            You are an expert groundwater data analyst. Based on the following groundwater data from the INGRIS dataset, provide a comprehensive analysis for the query: "{query}"

            IMPORTANT: Format your response EXACTLY as shown below with proper markdown tables and structure:

            💧 Groundwater Data Analysis Report

            Query
            **Question:** {query}

            Analysis
            Groundwater Estimation Report: [State/Region] - Year [Year]

            [Brief introduction paragraph]

            District-Wise Analysis

            [For each district, provide the following 8 sections with proper markdown tables:]

            1. [District Name] District

            #### 1. 🚨 CRITICALITY ALERT & SUSTAINABILITY STATUS:

            | Parameter | Value | Unit | Significance |
            |-----------|-------|------|--------------|
            | Stage of Ground Water Extraction (%) | [value] | % | [significance] |
            | Groundwater categorization | [category] | N/A | [assessment] |

            **🚨 CRITICAL ALERT:** [if applicable]
            **Sustainability Indicators:** [analysis]

            #### 2. 📈 GROUNDWATER TREND ANALYSIS:

            | Parameter | Value |
            |-----------|-------|
            | Pre-monsoon groundwater trend | [value] |
            | Post-monsoon groundwater trend | [value] |

            **Trend Implications:** [analysis]
            **Seasonal Variation Analysis:** [analysis]

            #### 3. 🌧️ RAINFALL & RECHARGE DATA:

            | Parameter | Value | Unit | Significance |
            |-----------|-------|------|--------------|
            | Rainfall | [value] | mm | [significance] |
            | Ground Water Recharge | [value] | ham | [significance] |
            | Annual Ground Water Recharge | [value] | ham | [significance] |
            | Environmental Flows | [value] | ham | [significance] |

            **Significance:** [analysis]

            #### 4. 💧 GROUNDWATER EXTRACTION & AVAILABILITY:

            | Parameter | Value | Unit | Significance |
            |-----------|-------|------|--------------|
            | Ground Water Extraction for all uses | [value] | ham | [significance] |
            | Annual Extractable Ground Water Resource | [value] | ham | [significance] |
            | Net Annual Ground Water Availability for Future Use | [value] | ham | [significance] |
            | Allocation for Domestic Utilisation for 2025 | [value] | ham | [significance] |

            **Extraction Efficiency:** [analysis]

            #### 5. 🔬 WATER QUALITY & ENVIRONMENTAL CONCERNS:

            | Parameter | Value |
            |-----------|-------|
            | Quality Tagging | [value] |

            **Quality Concerns:** [analysis]
            **Treatment Recommendations:** [analysis]
            **Environmental Sustainability:** [analysis]

            #### 6. 🏖️ COASTAL & SPECIAL AREAS:

            | Parameter | Value |
            |-----------|-------|
            | Coastal Areas identification | [value] |
            | Additional Potential Resources under specific conditions | [value] |

            **Special Management:** [analysis]
            **Climate Resilience Considerations:** [analysis]

            #### 7. 🏗️ GROUNDWATER STORAGE & RESOURCES:

            | Parameter | Value | Unit |
            |-----------|-------|------|
            | Instorage Unconfined Ground Water Resources | [value] | ham |
            | Total Ground Water Availability in Unconfined Aquifer | [value] | ham |
            | Dynamic Confined Ground Water Resources | [value] | ham |
            | Instorage Confined Ground Water Resources | [value] | ham |
            | Total Confined Ground Water Resources | [value] | ham |
            | Dynamic Semi-confined Ground Water Resources | [value] | ham |
            | Instorage Semi-confined Ground Water Resources | [value] | ham |
            | Total Semi-confined Ground Water Resources | [value] | ham |
            | Total Ground Water Availability in the Area | [value] | ham |

            **Storage Analysis:** [analysis]

            #### 8. 🌊 WATERSHED & ADMINISTRATIVE ANALYSIS:

            | Parameter | Value |
            |-----------|-------|
            | Watershed District | [value] |
            | Watershed Category | [value] |
            | Tehsil | [value] |
            | Taluk | [value] |
            | Block | [value] |
            | Mandal | [value] |
            | Village | [value] |

            **Watershed Status:** [analysis]

            [Continue for all districts...]

            [State-Level Comprehensive Summary with same 8 sections]

            [Comparative Analysis Between Districts with table]

            [Conclusion and Recommendations]

            *Report generated by Groundwater RAG API - Multilingual Support*
            *Language: English*

            Groundwater Data Context:
            {context}
            """
            
            # Generate response
            response = st.session_state.gemini_model.generate_content(prompt)
            return response.text
        
        else:
            # Fallback response
            if search_results:
                return f"""
                **Groundwater Data Analysis for: {query}**
                
                Found {len(search_results)} relevant records:
                
                {chr(10).join([f"• {result['document'][:200]}..." for result in search_results[:3]])}
                
                *Note: AI analysis unavailable. Please check Gemini API configuration.*
                """
            else:
                return f"No relevant groundwater data found for: {query}"
    
    except Exception as e:
        st.error(f"❌ Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="INGRIS Groundwater Chatbot",
        page_icon="💧",
        layout="wide"
    )
    
    st.title("💧 INGRIS Groundwater Data Chatbot")
    st.markdown("**Powered by ChromaDB with 162,632 groundwater records**")
    
    # Initialize components
    with st.spinner("Initializing components..."):
        if not initialize_components():
            st.error("Failed to initialize components. Please check the configuration.")
            return
    
    # Display collection info
    if st.session_state.collection:
        try:
            count = st.session_state.collection.count()
            st.info(f"📊 Database contains {count:,} groundwater records")
        except:
            st.info("📊 Database loaded successfully")
    
    # Chat interface
    st.markdown("### 💬 Ask about Groundwater Data")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about groundwater data (e.g., 'groundwater estimation in Karnataka')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Search and generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching groundwater data..."):
                search_results = search_groundwater_data(prompt, n_results=10)
            
            with st.spinner("Generating analysis..."):
                response = generate_response(prompt, search_results)
            
            st.markdown(response)
            
            # Show search results summary
            if search_results:
                with st.expander("🔍 Search Results Summary"):
                    for i, result in enumerate(search_results[:3], 1):
                        st.write(f"**Result {i}** (Similarity: {result['similarity']:.3f})")
                        st.write(f"Document: {result['document'][:300]}...")
                        if result['metadata']:
                            st.json(result['metadata'])
                        st.divider()
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### 📋 Dataset Information")
        st.markdown("""
        **INGRIS Groundwater Dataset:**
        - **Records:** 162,632
        - **Coverage:** All Indian states
        - **Data:** Groundwater recharge, extraction, trends, quality
        - **Years:** 2016-2024
        - **Vector Store:** ChromaDB
        - **Embeddings:** all-mpnet-base-v2 (768D)
        """)
        
        st.markdown("### 💡 Sample Queries")
        sample_queries = [
            "groundwater estimation in Karnataka",
            "critical groundwater areas in Maharashtra", 
            "water quality issues in Tamil Nadu",
            "groundwater recharge trends in Rajasthan",
            "over-exploited districts in India"
        ]
        
        for query in sample_queries:
            if st.button(f"💧 {query}", key=f"sample_{query}"):
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        st.markdown("### 🔧 Technical Details")
        st.markdown("""
        **Search Features:**
        - Semantic search using sentence transformers
        - Similarity scoring
        - Metadata filtering
        - Context-aware responses
        
        **AI Features:**
        - Gemini 2.0 Flash integration
        - Comprehensive data analysis
        - Criticality assessment
        - Management recommendations
        """)

if __name__ == "__main__":
    main()

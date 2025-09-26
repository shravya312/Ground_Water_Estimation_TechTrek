#!/usr/bin/env python3
"""
Main API using ChromaDB instead of Qdrant
Handles 2 lakh+ records efficiently with reliable retrieval
"""

import os
import pandas as pd
import numpy as np
import re
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
import spacy
from langdetect import detect
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "ingris_groundwater_collection"
VECTOR_SIZE = 768
MIN_SIMILARITY_SCORE = 0.1

# Global variables
_chroma_client = None
_collection = None
_model = None
_gemini_model = None
_nlp = None
_master_df = None

# Language support
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'mr': 'Marathi',
    'or': 'Odia',
    'pa': 'Punjabi',
    'as': 'Assamese'
}

# FastAPI app
app = FastAPI(title="Groundwater RAG API - ChromaDB", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    user_language: str = 'en'
    user_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    language: str
    sources: List[Dict[str, Any]] = []

class LocationAnalysisRequest(BaseModel):
    latitude: float
    longitude: float
    radius_km: float = 50

def _init_components():
    """Initialize all components"""
    global _chroma_client, _collection, _model, _gemini_model, _nlp, _master_df
    
    try:
        print("🔄 Starting application initialization...")
        
        # Initialize ChromaDB
        print("🔄 Connecting to ChromaDB...")
        _chroma_client = chromadb.Client(Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        
        try:
            _collection = _chroma_client.get_collection(COLLECTION_NAME)
            print(f"✅ ChromaDB collection '{COLLECTION_NAME}' ready")
        except:
            print(f"❌ ChromaDB collection '{COLLECTION_NAME}' not found")
            return False
        
        # Initialize embedding model
        print("🔄 Loading embedding model...")
        _model = SentenceTransformer('all-mpnet-base-v2')
        print("✅ Embedding model loaded")
        
        # Initialize Gemini
        if GEMINI_API_KEY:
            print("🔄 Initializing Gemini...")
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_model = genai.GenerativeModel('models/gemini-2.0-flash')
            print("✅ Gemini model initialized")
        else:
            print("⚠️ No Gemini API key found")
        
        # Initialize spaCy
        try:
            _nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded")
        except:
            print("⚠️ spaCy model not available")
        
        # Load master data
        try:
            _master_df = pd.read_csv("ingris_rag_ready_complete.csv")
            print(f"✅ Master data loaded: {len(_master_df)} records")
        except:
            print("⚠️ Master data not available")
        
        print("✅ All components initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def search_groundwater_data(query_text, year=None, target_state=None, target_district=None, limit=20):
    """Search groundwater data using ChromaDB"""
    global _collection, _model
    
    if not _collection or not _model:
        return []
    
    try:
        # Create query vector
        query_vector = _model.encode([query_text])[0].tolist()
        
        # Create where filter for ChromaDB
        where_filter = {}
        if target_state:
            where_filter["STATE"] = target_state.upper()
        if target_district:
            where_filter["DISTRICT"] = target_district
        if year:
            where_filter["Assessment_Year"] = year
        
        # Search ChromaDB
        results = _collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=where_filter if where_filter else None
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    "score": results['distances'][0][i] if results['distances'] else 0.0,
                    "data": results['metadatas'][0][i] if results['metadatas'] else {}
                })
        
        return formatted_results
        
    except Exception as e:
        print(f"Error searching ChromaDB: {e}")
        return []

def translate_query_to_english(query):
    """Translate query to English if needed"""
    try:
        detected_lang = detect(query)
        if detected_lang in SUPPORTED_LANGUAGES and detected_lang != 'en':
            # Simple translation - in production, use proper translation service
            return query, detected_lang
        return query, 'en'
    except:
        return query, 'en'

def answer_query(query: str, user_language: str = 'en', user_id: str = None) -> str:
    """Answer groundwater queries using ChromaDB"""
    query = (query or '').strip()
    if not query:
        return "Please provide a question."
    
    try:
        if not _collection or not _model:
            return "System not initialized. Please try again later."
    except Exception as e:
        return f"Initialization error: {str(e)}"
    
    # Detect and translate query
    original_query = query
    translated_query, detected_lang = translate_query_to_english(query)
    
    # Extract year, state, district from query
    year = None
    year_match = re.search(r'\b(19|20)\d{2}\b', translated_query)
    if year_match:
        year = int(year_match.group(0))
    
    target_state = None
    target_district = None
    if _master_df is not None:
        unique_states = _master_df['STATE'].unique().tolist()
        unique_districts = _master_df['DISTRICT'].unique().tolist()
        
        # Find state
        for state in unique_states:
            if pd.notna(state):
                if re.search(r'\b' + re.escape(str(state)) + r'\b', translated_query, re.IGNORECASE):
                    target_state = state
                    break
                elif str(state).lower() in translated_query.lower():
                    target_state = state
                    break
        
        # Find district
        if target_state:
            districts_in_state = _master_df[_master_df['STATE'] == target_state]['DISTRICT'].unique().tolist()
            for district in districts_in_state:
                if pd.notna(district):
                    if re.search(r'\b' + re.escape(str(district)) + r'\b', translated_query, re.IGNORECASE):
                        target_district = district
                        break
                    elif str(district).lower() in translated_query.lower():
                        target_district = district
                        break
    
    # Search for relevant data
    candidate_results = search_groundwater_data(translated_query, year, target_state, target_district)
    
    # Fallback searches if no results
    if not candidate_results:
        candidate_results = search_groundwater_data(translated_query, year, None, None)
    
    if not candidate_results:
        candidate_results = search_groundwater_data(original_query, year, None, None)
    
    if not candidate_results:
        candidate_results = search_groundwater_data("groundwater data", year, target_state, target_district)
    
    if not candidate_results:
        return f"I couldn't find enough relevant information in the groundwater data to answer your question about '{query}'."
    
    # Generate response using Gemini
    if _gemini_model and candidate_results:
        try:
            # Prepare context from search results
            context_data = []
            for result in candidate_results[:10]:  # Use top 10 results
                if result.get('data'):
                    context_data.append(result['data'])
            
            # Create context string
            context_str = "\n\n".join([
                f"Data Point {i+1}:\n" + "\n".join([f"{k}: {v}" for k, v in data.items() if v is not None and str(v).strip() != ''])
                for i, data in enumerate(context_data)
            ])
            
            # Generate response
            prompt = f"""
            You are a groundwater expert. Based on the following groundwater data, provide a comprehensive analysis for the query: "{translated_query}"
            
            Groundwater Data:
            {context_str}
            
            Please provide:
            1. A detailed analysis of the groundwater situation
            2. Key findings and trends
            3. District-wise breakdown if multiple districts are mentioned
            4. Specific metrics and statistics
            5. Recommendations if applicable
            
            Format your response as a professional groundwater analysis report.
            """
            
            response = _gemini_model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"Gemini error: {e}")
            # Fallback to simple response
            return f"Based on the available groundwater data, I found {len(candidate_results)} relevant records. Here's a summary of the key findings:\n\n" + \
                   "\n".join([f"• {result.get('data', {}).get('DISTRICT', 'Unknown')} - {result.get('data', {}).get('STATE', 'Unknown')}" 
                             for result in candidate_results[:5]])
    
    return f"I found {len(candidate_results)} relevant groundwater records, but couldn't generate a detailed analysis at this time."

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Groundwater RAG API with ChromaDB is running"}

@app.post("/ingres/query", response_model=QueryResponse)
async def query_groundwater(request: QueryRequest):
    """Query groundwater data"""
    try:
        response = answer_query(request.query, request.user_language, request.user_id)
        return QueryResponse(
            response=response,
            language=request.user_language,
            sources=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ingres/states")
async def get_states():
    """Get list of states"""
    try:
        if _master_df is not None:
            states = _master_df['STATE'].unique().tolist()
            states = [state for state in states if pd.notna(state)]
            return {"states": sorted(states)}
        return {"states": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ingres/districts/{state}")
async def get_districts(state: str):
    """Get districts for a state"""
    try:
        if _master_df is not None:
            districts = _master_df[_master_df['STATE'] == state.upper()]['DISTRICT'].unique().tolist()
            districts = [district for district in districts if pd.notna(district)]
            return {"districts": sorted(districts)}
        return {"districts": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Initialize components
    if _init_components():
        print("🚀 Starting Groundwater RAG API with ChromaDB...")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("❌ Failed to initialize. Please check your setup.")

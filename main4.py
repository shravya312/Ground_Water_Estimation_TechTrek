#!/usr/bin/env python3
"""
Groundwater Estimation API - ChromaDB Primary with CSV Fallback
Optimized for ingris_rag_ready_complete.csv data
"""

import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
import re
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Groundwater Estimation API - ChromaDB", version="4.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
_chromadb_client = None
_collection = None
_model = None
_csv_data = None

# Configuration
COLLECTION_NAME = "ingris_groundwater_collection"
CSV_FILE_PATH = "ingris_rag_ready_complete.csv"
MODEL_NAME = "all-MiniLM-L6-v2"

class QueryRequest(BaseModel):
    query: str
    year: Optional[int] = None
    state: Optional[str] = None
    district: Optional[str] = None

class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_results: int
    search_method: str
    processing_time: float
    query_info: Dict[str, Any]

def _init_components():
    """Initialize ChromaDB, model, and CSV data"""
    global _chromadb_client, _collection, _model, _csv_data
    
    try:
        # Initialize ChromaDB
        if _chromadb_client is None:
            _chromadb_client = chromadb.PersistentClient(path="./chroma_db")
            logger.info("ChromaDB client initialized")
        
        # Get collection
        if _collection is None:
            try:
                _collection = _chromadb_client.get_collection(COLLECTION_NAME)
                logger.info(f"ChromaDB collection '{COLLECTION_NAME}' loaded")
            except Exception as e:
                logger.warning(f"ChromaDB collection not found: {e}")
                _collection = None
        
        # Initialize embedding model
        if _model is None:
            _model = SentenceTransformer(MODEL_NAME)
            logger.info(f"Embedding model '{MODEL_NAME}' loaded")
        
        # Load CSV data as fallback
        if _csv_data is None and os.path.exists(CSV_FILE_PATH):
            _csv_data = pd.read_csv(CSV_FILE_PATH)
            logger.info(f"CSV fallback data loaded: {len(_csv_data)} records")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise

def search_chromadb(query_text: str, year: Optional[int] = None, 
                   target_state: Optional[str] = None, 
                   target_district: Optional[str] = None, 
                   limit: int = 20) -> List[Dict[str, Any]]:
    """Search ChromaDB collection with optional filters"""
    try:
        if _collection is None:
            logger.warning("ChromaDB collection not available")
            return []
        
        # Generate query embedding
        query_embedding = _model.encode([query_text])[0].tolist()
        
        # Build where clause for filtering
        where_clause = {}
        if target_state:
            where_clause["STATE"] = target_state.upper()
        if target_district:
            where_clause["DISTRICT"] = target_district
        if year:
            where_clause["Assessment_Year"] = year
        
        # Search ChromaDB
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_clause if where_clause else None
        )
        
        # Process results
        processed_results = []
        if results['ids'] and results['ids'][0]:
            for i, (doc_id, distance, metadata) in enumerate(zip(
                results['ids'][0],
                results['distances'][0],
                results['metadatas'][0]
            )):
                result = {
                    'id': doc_id,
                    'score': 1 - distance,  # Convert distance to similarity score
                    'metadata': metadata,
                    'text': metadata.get('combined_text', ''),
                    'state': metadata.get('STATE', metadata.get('state', 'N/A')),
                    'district': metadata.get('DISTRICT', metadata.get('district', 'N/A')),
                    'year': metadata.get('Assessment_Year', metadata.get('year', 'N/A')),
                    'serial_number': metadata.get('serial_number', 'N/A')
                }
                processed_results.append(result)
        
        logger.info(f"ChromaDB search returned {len(processed_results)} results")
        return processed_results
        
    except Exception as e:
        logger.error(f"ChromaDB search failed: {e}")
        return []

def search_csv_fallback(query_text: str, year: Optional[int] = None,
                       target_state: Optional[str] = None,
                       target_district: Optional[str] = None,
                       limit: int = 20) -> List[Dict[str, Any]]:
    """Fallback search using CSV data with semantic similarity"""
    try:
        if _csv_data is None:
            logger.warning("CSV data not available for fallback")
            return []
        
        # Filter data based on criteria
        filtered_data = _csv_data.copy()
        
        if target_state:
            filtered_data = filtered_data[filtered_data['state'].str.upper() == target_state.upper()]
        
        if target_district:
            filtered_data = filtered_data[filtered_data['district'].str.upper() == target_district.upper()]
        
        if year:
            filtered_data = filtered_data[filtered_data['year'] == year]
        
        if len(filtered_data) == 0:
            logger.warning("No data found matching criteria in CSV")
            return []
        
        # Generate embeddings for filtered data
        combined_texts = filtered_data['combined_text'].fillna('').tolist()
        data_embeddings = _model.encode(combined_texts)
        query_embedding = _model.encode([query_text])[0]
        
        # Calculate similarities
        similarities = []
        for i, (_, row) in enumerate(filtered_data.iterrows()):
            similarity = float(query_embedding @ data_embeddings[i])
            similarities.append((i, similarity, row))
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:limit]
        
        # Process results
        processed_results = []
        for i, (original_idx, similarity, row) in enumerate(top_results):
            result = {
                'id': f"csv_{original_idx}",
                'score': similarity,
                'metadata': row.to_dict(),
                'text': row.get('combined_text', ''),
                'state': row.get('state', 'N/A'),
                'district': row.get('district', 'N/A'),
                'year': row.get('year', 'N/A'),
                'serial_number': row.get('serial_number', 'N/A')
            }
            processed_results.append(result)
        
        logger.info(f"CSV fallback search returned {len(processed_results)} results")
        return processed_results
        
    except Exception as e:
        logger.error(f"CSV fallback search failed: {e}")
        return []

def extract_location_info(query: str) -> Dict[str, str]:
    """Extract state and district information from query"""
    query_upper = query.upper()
    
    # State mapping
    state_mapping = {
        'KARNATAKA': 'KARNATAKA',
        'KERALA': 'KERALA',
        'TAMILNADU': 'TAMILNADU',
        'ANDHRA PRADESH': 'ANDHRA PRADESH',
        'TELANGANA': 'TELANGANA',
        'MAHARASHTRA': 'MAHARASHTRA',
        'GUJARAT': 'GUJARAT',
        'RAJASTHAN': 'RAJASTHAN',
        'UTTAR PRADESH': 'UTTAR PRADESH',
        'BIHAR': 'BIHAR',
        'MADHYA PRADESH': 'MADHYA PRADESH',
        'ODISHA': 'ODISHA',
        'JHARKHAND': 'JHARKHAND',
        'HARYANA': 'HARYANA',
        'PUNJAB': 'PUNJAB',
        'HIMACHAL PRADESH': 'HIMACHAL PRADESH',
        'JAMMU AND KASHMIR': 'JAMMU AND KASHMIR',
        'DELHI': 'DELHI',
        'GOA': 'GOA',
        'ANDAMAN AND NICOBAR ISLANDS': 'ANDAMAN AND NICOBAR ISLANDS',
        'LAKSHDWEEP': 'LAKSHDWEEP'
    }
    
    # Extract state
    detected_state = None
    for state_name, state_code in state_mapping.items():
        if state_name in query_upper:
            detected_state = state_code
            break
    
    # Extract district (simplified - look for common district patterns)
    district_patterns = [
        r'\b(\w+)\s+district\b',
        r'\b(\w+)\s+taluk\b',
        r'\b(\w+)\s+block\b'
    ]
    
    detected_district = None
    for pattern in district_patterns:
        match = re.search(pattern, query_upper)
        if match:
            detected_district = match.group(1).title()
            break
    
    return {
        'state': detected_state,
        'district': detected_district
    }

def search_groundwater_data(query_text: str, year: Optional[int] = None,
                           target_state: Optional[str] = None,
                           target_district: Optional[str] = None) -> List[Dict[str, Any]]:
    """Main search function - ChromaDB primary with CSV fallback"""
    start_time = datetime.now()
    
    try:
        # Initialize components
        _init_components()
        
        # Extract location info from query if not provided
        if not target_state or not target_district:
            location_info = extract_location_info(query_text)
            if not target_state:
                target_state = location_info['state']
            if not target_district:
                target_district = location_info['district']
        
        logger.info(f"Searching for: '{query_text}' | State: {target_state} | District: {target_district} | Year: {year}")
        
        # Try ChromaDB first
        results = search_chromadb(query_text, year, target_state, target_district, limit=20)
        
        if results:
            logger.info(f"ChromaDB search successful: {len(results)} results")
            return results
        else:
            logger.info("ChromaDB search returned no results, trying CSV fallback")
            # Fallback to CSV
            results = search_csv_fallback(query_text, year, target_state, target_district, limit=20)
            if results:
                logger.info(f"CSV fallback successful: {len(results)} results")
            else:
                logger.warning("Both ChromaDB and CSV search returned no results")
            
            return results
            
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Groundwater Estimation API - ChromaDB Version",
        "version": "4.0",
        "status": "running",
        "primary_search": "ChromaDB",
        "fallback_search": "CSV"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        _init_components()
        
        chromadb_status = "connected" if _collection else "not_available"
        csv_status = "loaded" if _csv_data is not None else "not_loaded"
        model_status = "loaded" if _model else "not_loaded"
        
        return {
            "status": "healthy",
            "chromadb": chromadb_status,
            "csv_fallback": csv_status,
            "model": model_status,
            "collection_name": COLLECTION_NAME,
            "csv_records": len(_csv_data) if _csv_data is not None else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/search", response_model=QueryResponse)
async def search_groundwater(request: QueryRequest):
    """Search groundwater data using ChromaDB with CSV fallback"""
    start_time = datetime.now()
    
    try:
        # Perform search
        results = search_groundwater_data(
            query_text=request.query,
            year=request.year,
            target_state=request.state,
            target_district=request.district
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Determine search method used
        search_method = "ChromaDB" if results and any('csv_' not in r['id'] for r in results) else "CSV_Fallback"
        
        # Prepare query info
        query_info = {
            "original_query": request.query,
            "filters_applied": {
                "year": request.year,
                "state": request.state,
                "district": request.district
            },
            "results_found": len(results)
        }
        
        return QueryResponse(
            results=results,
            total_results=len(results),
            search_method=search_method,
            processing_time=processing_time,
            query_info=query_info
        )
        
    except Exception as e:
        logger.error(f"Search API error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get collection statistics"""
    try:
        _init_components()
        
        stats = {
            "chromadb": {
                "available": _collection is not None,
                "collection_name": COLLECTION_NAME
            },
            "csv_fallback": {
                "available": _csv_data is not None,
                "records": len(_csv_data) if _csv_data is not None else 0
            },
            "model": {
                "available": _model is not None,
                "name": MODEL_NAME
            }
        }
        
        if _collection:
            try:
                chromadb_count = _collection.count()
                stats["chromadb"]["record_count"] = chromadb_count
            except:
                stats["chromadb"]["record_count"] = "unknown"
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@app.get("/test")
async def test_search():
    """Test search functionality"""
    try:
        # Test with a simple query
        test_query = "groundwater estimation in Karnataka"
        results = search_groundwater_data(test_query)
        
        return {
            "test_query": test_query,
            "results_found": len(results),
            "sample_results": results[:3] if results else [],
            "status": "success"
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

if __name__ == "__main__":
    print("🚀 Starting Groundwater Estimation API - ChromaDB Version")
    print("=" * 60)
    print(f"Primary Search: ChromaDB Collection '{COLLECTION_NAME}'")
    print(f"Fallback Search: CSV File '{CSV_FILE_PATH}'")
    print(f"Embedding Model: {MODEL_NAME}")
    print("=" * 60)
    
    uvicorn.run(
        "main4:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

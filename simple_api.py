#!/usr/bin/env python3
"""
Simple, reliable API for Karnataka groundwater search
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Simple Karnataka Groundwater API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
_qdrant_client = None
_model = None

def init_components():
    """Initialize Qdrant and model"""
    global _qdrant_client, _model
    
    if _qdrant_client is None:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        _qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, check_compatibility=False)
    
    if _model is None:
        _model = SentenceTransformer('all-mpnet-base-v2')

class QueryRequest(BaseModel):
    query: str
    state: str = None
    limit: int = 10

class QueryResponse(BaseModel):
    success: bool
    results: list
    total_found: int
    query: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Simple Karnataka API is running"}

@app.post("/search", response_model=QueryResponse)
async def search_groundwater(request: QueryRequest):
    """Search groundwater data with simple, reliable method"""
    try:
        init_components()
        
        # Create query vector
        query_vector = _model.encode([request.query])[0].tolist()
        
        # Create filter if state is specified
        query_filter = None
        if request.state:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="STATE",
                        match=MatchValue(value=request.state.upper())
                    )
                ]
            )
        
        # Search with very low threshold to ensure we get results
        results = _qdrant_client.search(
            collection_name="ingris_groundwater_collection",
            query_vector=query_vector,
            query_filter=query_filter,
            limit=request.limit,
            with_payload=True,
            score_threshold=0.0  # Very low threshold
        )
        
        # Format results
        formatted_results = []
        for result in results:
            payload = result.payload
            formatted_results.append({
                "score": result.score,
                "state": payload.get("STATE", "N/A"),
                "district": payload.get("DISTRICT", "N/A"),
                "year": payload.get("Assessment_Year", "N/A"),
                "rainfall_mm": payload.get("rainfall_mm", "N/A"),
                "ground_water_recharge_ham": payload.get("ground_water_recharge_ham", "N/A"),
                "stage_of_extraction": payload.get("stage_of_ground_water_extraction_", "N/A"),
                "categorization": payload.get("categorization_of_assessment_unit", "N/A"),
                "text_preview": payload.get("text", "N/A")[:200] + "..." if len(payload.get("text", "")) > 200 else payload.get("text", "N/A")
            })
        
        return QueryResponse(
            success=True,
            results=formatted_results,
            total_found=len(formatted_results),
            query=request.query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/karnataka")
async def get_karnataka_data():
    """Get sample Karnataka data"""
    try:
        init_components()
        
        # Get Karnataka data
        karnataka_filter = Filter(
            must=[
                FieldCondition(
                    key="STATE",
                    match=MatchValue(value="KARNATAKA")
                )
            ]
        )
        
        results = _qdrant_client.scroll(
            collection_name="ingris_groundwater_collection",
            scroll_filter=karnataka_filter,
            limit=20,
            with_payload=True
        )
        
        formatted_results = []
        for point in results[0]:
            payload = point.payload
            formatted_results.append({
                "district": payload.get("DISTRICT", "N/A"),
                "year": payload.get("Assessment_Year", "N/A"),
                "rainfall_mm": payload.get("rainfall_mm", "N/A"),
                "ground_water_recharge_ham": payload.get("ground_water_recharge_ham", "N/A"),
                "stage_of_extraction": payload.get("stage_of_ground_water_extraction_", "N/A"),
                "categorization": payload.get("categorization_of_assessment_unit", "N/A")
            })
        
        return {
            "success": True,
            "total_districts": len(formatted_results),
            "data": formatted_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Karnataka data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

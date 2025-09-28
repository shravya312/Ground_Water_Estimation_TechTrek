# Advanced RAG (Retrieval-Augmented Generation) Documentation

## Overview

This document describes the advanced RAG implementation for the Groundwater Estimation System, featuring hybrid search, reranking, and query expansion capabilities.

## Features

### 1. Hybrid Search
Combines dense (vector) and sparse (BM25) retrieval methods for comprehensive search coverage.

**Components:**
- **Dense Retrieval**: Uses sentence transformers for semantic similarity
- **Sparse Retrieval**: Uses BM25 for keyword-based matching
- **Hybrid Scoring**: Weighted combination of both methods (default: 60% dense, 40% sparse)

### 2. Query Expansion
Enhances user queries with domain-specific terms using Gemini AI.

**Features:**
- Domain-specific prompts for groundwater terminology
- Technical term expansion
- Synonym and alternative phrasing generation
- Configurable number of expansion terms (default: 3)

### 3. Advanced Reranking
Improves result relevance through semantic similarity and additional factors.

**Features:**
- Cosine similarity calculation
- Minimum similarity thresholds
- Combined scoring (70% semantic + 30% original score)
- Returns ALL chunks above minimum similarity threshold (no top-k limit)

## Configuration Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `HYBRID_ALPHA` | 0.6 | 0.0-1.0 | Weight for dense vs sparse retrieval |
| `QUERY_EXPANSION_TERMS` | 5 | 0-10 | Number of terms to add via expansion |
| `RERANK_MIN_SIMILARITY` | 0.3 | 0.0-1.0 | Minimum similarity for reranking (returns ALL chunks above threshold) |
| `MIN_SIMILARITY_SCORE` | 0.7 | 0.0-1.0 | Minimum similarity for dense search (high precision) |

## API Endpoints

### 1. Advanced Query Endpoint
```
POST /ingres/advanced-query
```

**Request Body:**
```json
{
  "query": "groundwater estimation in Chikkamagaluru",
  "language": "en",
  "user_id": "optional_user_id",
  "state": "optional_state",
  "district": "optional_district",
  "year": "optional_year"
}
```

**Response:**
```json
{
  "success": true,
  "response": "Detailed groundwater analysis...",
  "state": "Karnataka",
  "district": "Chikkamagaluru",
  "year": "2024",
  "language": "en",
  "query": "groundwater estimation in Chikkamagaluru",
  "user_id": "optional_user_id",
  "timestamp": "2024-01-01T12:00:00",
  "rag_method": "advanced_hybrid_rerank"
}
```

### 2. RAG Configuration Endpoints

#### Get Configuration
```
GET /rag/config
```

#### Update Configuration
```
POST /rag/config
```

**Request Body:**
```json
{
  "hybrid_alpha": 0.7,
  "rerank_top_k": 15,
  "query_expansion_terms": 5,
  "rerank_min_similarity": 0.3,
  "min_similarity_score": 0.2
}
```

## Implementation Details

### 1. Hybrid Search Pipeline

```python
def hybrid_search(query_text, year=None, target_state=None, target_district=None, extracted_parameters=None):
    # Step 1: Dense Retrieval (Vector Search)
    # - Qdrant vector search
    # - ChromaDB fallback
    
    # Step 2: Sparse Retrieval (BM25)
    # - Load BM25 model
    # - Calculate BM25 scores
    
    # Step 3: Hybrid Scoring
    # - Normalize scores
    # - Weighted combination
    # - Sort by combined score
```

### 2. Query Expansion

```python
def enhanced_query_expansion(query, num_terms=3):
    # Domain-specific prompt for groundwater data
    # Focus on technical terms, synonyms, alternatives
    # Return expanded query with additional terms
```

### 3. Advanced Reranking

```python
def advanced_rerank(query_text, candidate_results, top_k=5):
    # Extract text from candidates
    # Calculate semantic similarity
    # Apply minimum threshold
    # Combine with original scores
    # Return top-k reranked results
```

## Usage Examples

### 1. Basic Usage

```python
import requests

# Test advanced RAG
response = requests.post("http://localhost:8000/ingres/advanced-query", json={
    "query": "groundwater extraction in Karnataka",
    "language": "en"
})

result = response.json()
print(result["response"])
```

### 2. Configuration Management

```python
# Get current configuration
config_response = requests.get("http://localhost:8000/rag/config")
current_config = config_response.json()

# Update configuration
new_config = {
    "hybrid_alpha": 0.7,
    "rerank_min_similarity": 0.3
}
update_response = requests.post("http://localhost:8000/rag/config", json=new_config)
```

### 3. Testing Script

```bash
python test_advanced_rag.py
```

## Performance Optimization

### 1. Threshold Tuning
Start with low thresholds (0.1) and gradually increase:
- 0.1: Broad search, more results
- 0.3: Balanced search
- 0.5: Focused search
- 0.7: High precision search

### 2. Hybrid Alpha Tuning
- 0.0: Pure sparse (BM25) search
- 0.5: Balanced hybrid
- 1.0: Pure dense (vector) search

### 3. Query Expansion
- 0 terms: No expansion
- 3 terms: Light expansion (default)
- 5+ terms: Heavy expansion

## Monitoring and Debugging

### 1. Log Messages
The system provides detailed logging:
- 🔍 Search progress
- 📊 Dense retrieval results
- 📝 Sparse retrieval results
- 🔄 Hybrid scoring
- ✅ Success indicators
- ❌ Error messages

### 2. Performance Metrics
- Response time
- Number of results found
- Similarity scores
- RAG method used

## Troubleshooting

### 1. No Results Found
- Lower `RERANK_MIN_SIMILARITY` threshold
- Lower `MIN_SIMILARITY_SCORE` threshold
- Check if data is properly indexed

### 2. Poor Quality Results
- Increase `RERANK_MIN_SIMILARITY` threshold
- Adjust `HYBRID_ALPHA` for better balance
- Increase `QUERY_EXPANSION_TERMS`

### 3. Slow Performance
- Reduce `RERANK_TOP_K`
- Reduce `QUERY_EXPANSION_TERMS`
- Check vector database performance

## Future Enhancements

### 1. Advanced Reranking
- Cross-encoder models for better reranking
- Learning-to-rank algorithms
- User feedback integration

### 2. Query Understanding
- Named entity recognition
- Intent classification
- Query reformulation

### 3. Multi-modal Search
- Image-based queries
- Geographic search
- Temporal search

## Dependencies

- `sentence-transformers`: For dense retrieval
- `rank-bm25`: For sparse retrieval
- `qdrant-client`: For vector storage
- `chromadb`: For fallback storage
- `google-generativeai`: For query expansion
- `numpy`: For numerical operations

## Conclusion

The advanced RAG system provides a robust, configurable solution for groundwater data retrieval with improved accuracy and relevance through hybrid search, query expansion, and advanced reranking techniques.

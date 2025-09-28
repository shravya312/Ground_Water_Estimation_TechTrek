# main4.py - ChromaDB Groundwater Estimation API

A streamlined groundwater estimation API that uses ChromaDB as the primary search method with CSV fallback, optimized for the `ingris_rag_ready_complete.csv` dataset.

## 🚀 Key Features

- **ChromaDB Primary**: Uses local ChromaDB collection as the main search engine
- **CSV Fallback**: Falls back to CSV data when ChromaDB is unavailable
- **Real-time Upload Support**: Works with data currently uploading to ChromaDB
- **State Filtering**: Proper filtering by state, district, and year
- **FastAPI**: Modern, fast API with automatic documentation
- **Health Monitoring**: Built-in health checks and statistics

## 📋 Requirements

```bash
pip install fastapi uvicorn chromadb sentence-transformers pandas requests
```

## 🚀 Quick Start

### Option 1: Using Launcher (Recommended)
```bash
python run_main4.py
```

### Option 2: Direct Start
```bash
python main4.py
```

### Option 3: With Uvicorn
```bash
uvicorn main4:app --host 0.0.0.0 --port 8000 --reload
```

## 🔧 API Endpoints

### 1. Health Check
```http
GET /health
```
Returns system status and component availability.

### 2. Search Groundwater Data
```http
POST /search
Content-Type: application/json

{
    "query": "groundwater estimation in Karnataka",
    "year": 2022,
    "state": "KARNATAKA",
    "district": "Bangalore Urban"
}
```

### 3. Statistics
```http
GET /stats
```
Returns collection statistics and data counts.

### 4. Test Search
```http
GET /test
```
Runs a test search to verify functionality.

## 🔍 Search Capabilities

### Primary Search (ChromaDB)
- **Vector Search**: Semantic similarity using embeddings
- **Metadata Filtering**: Filter by state, district, year
- **Real-time Data**: Uses data currently uploading to ChromaDB
- **Fast Response**: Optimized for speed

### Fallback Search (CSV)
- **Semantic Search**: Vector similarity on CSV data
- **Complete Dataset**: All 162,632 records available
- **Reliable**: Always available as backup

## 📊 Data Structure

### ChromaDB Records
- **ID**: Unique document identifier
- **Score**: Similarity score (0-1)
- **Metadata**: 44+ fields including STATE, DISTRICT, Assessment_Year
- **Text**: Combined searchable text
- **Vector**: 384-dimensional embedding

### CSV Fallback
- **Same Structure**: Maintains compatibility with ChromaDB format
- **Complete Data**: All records from ingris_rag_ready_complete.csv
- **Filtering**: State, district, and year filtering

## 🎯 Usage Examples

### 1. Basic Search
```python
import requests

response = requests.post("http://localhost:8000/search", json={
    "query": "groundwater estimation in Karnataka"
})

results = response.json()
print(f"Found {results['total_results']} results")
```

### 2. Filtered Search
```python
response = requests.post("http://localhost:8000/search", json={
    "query": "groundwater recharge",
    "state": "KARNATAKA",
    "year": 2022
})
```

### 3. Health Check
```python
response = requests.get("http://localhost:8000/health")
status = response.json()
print(f"ChromaDB: {status['chromadb']}")
print(f"CSV Records: {status['csv_records']}")
```

## 🔄 Upload Integration

The API is designed to work with the ongoing ChromaDB upload:

1. **Real-time Compatibility**: Works with partially uploaded data
2. **Progressive Enhancement**: Performance improves as more data is uploaded
3. **Fallback Safety**: Always falls back to CSV if ChromaDB is incomplete
4. **State Awareness**: Knows which states have been uploaded

## 📈 Performance

### ChromaDB Search
- **Speed**: ~100-500ms per query
- **Accuracy**: High semantic similarity
- **Scalability**: Handles large datasets efficiently

### CSV Fallback
- **Speed**: ~1-3 seconds per query
- **Completeness**: All data available
- **Reliability**: Always works

## 🛠️ Configuration

### Environment Variables
- No environment variables required
- Uses local ChromaDB instance
- CSV file path: `ingris_rag_ready_complete.csv`

### Customization
- **Collection Name**: `COLLECTION_NAME = "ingris_groundwater_collection"`
- **Model**: `MODEL_NAME = "all-MiniLM-L6-v2"`
- **Port**: Default 8000 (configurable in uvicorn.run)

## 🔍 Monitoring

### Health Endpoint
```http
GET /health
```
Returns:
- ChromaDB connection status
- CSV data availability
- Model loading status
- Record counts

### Stats Endpoint
```http
GET /stats
```
Returns:
- Collection statistics
- Data availability
- Performance metrics

## 🚨 Troubleshooting

### Common Issues

1. **ChromaDB Not Available**
   - Check if collection exists
   - Verify data is uploading
   - API will fallback to CSV

2. **CSV Not Found**
   - Ensure `ingris_rag_ready_complete.csv` exists
   - Check file permissions
   - Verify file path

3. **Model Loading Issues**
   - Check internet connection
   - Verify sentence-transformers installation
   - Model will download on first use

### Debug Mode
```bash
uvicorn main4:app --reload --log-level debug
```

## 🔄 Integration with Upload Process

The API is designed to work seamlessly with the ChromaDB upload:

1. **Start API**: Run `python main4.py`
2. **Monitor Upload**: Use `/health` endpoint to check progress
3. **Test Search**: Use `/test` endpoint to verify functionality
4. **Scale Up**: Performance improves as more data is uploaded

## 📊 Expected Results

### With Partial Upload (Current)
- **ChromaDB**: 7,500+ records from 12 states
- **CSV Fallback**: All 162,632 records
- **Karnataka**: 30+ records in ChromaDB, 1,186 in CSV

### With Complete Upload
- **ChromaDB**: All 162,632 records
- **Fast Search**: Sub-second response times
- **Complete Coverage**: All states and districts

## 🎯 Next Steps

1. **Start the API**: `python run_main4.py`
2. **Test Functionality**: Use the test endpoints
3. **Monitor Upload**: Check health status regularly
4. **Scale Up**: Performance improves as upload completes

---

**Ready to serve groundwater data with ChromaDB! 🚀**

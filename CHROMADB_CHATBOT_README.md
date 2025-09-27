# 💧 INGRIS Groundwater Chatbot - ChromaDB Version

A comprehensive groundwater data chatbot using ChromaDB vector database with 162,632 INGRIS records.

## 🚀 Features

- **Complete Dataset**: 162,632 groundwater records from INGRIS
- **Vector Search**: Semantic search using all-mpnet-base-v2 embeddings (768D)
- **AI Analysis**: Powered by Gemini 2.0 Flash
- **Interactive UI**: Streamlit-based chat interface
- **Real-time Search**: Fast vector similarity search
- **Comprehensive Analysis**: Criticality assessment, trends, recommendations

## 📊 Dataset Information

- **Source**: INGRIS (Indian Groundwater Resource Information System)
- **Records**: 162,632 groundwater data points
- **Coverage**: All Indian states and union territories
- **Data Types**: Recharge, extraction, trends, quality, administrative divisions
- **Time Period**: 2016-2024
- **Vector Dimensions**: 768 (all-mpnet-base-v2)

## 🛠️ Setup Instructions

### 1. Prerequisites

```bash
# Install required packages
pip install streamlit chromadb sentence-transformers google-generativeai pandas numpy
```

### 2. Environment Setup

```bash
# Set Gemini API key
export GEMINI_API_KEY="your_gemini_api_key_here"
```

### 3. Data Upload (if not already done)

```bash
# Upload data to ChromaDB
python upload_to_chromadb.py
```

### 4. Test Setup

```bash
# Test ChromaDB and Gemini setup
python test_chromadb_setup.py
```

### 5. Launch Chatbot

```bash
# Start the chatbot
python run_chromadb_chatbot.py
```

## 🎯 Usage

### Sample Queries

- "groundwater estimation in Karnataka"
- "critical groundwater areas in Maharashtra"
- "water quality issues in Tamil Nadu"
- "groundwater recharge trends in Rajasthan"
- "over-exploited districts in India"
- "groundwater availability in coastal areas"
- "semi-critical zones in Uttar Pradesh"

### Features

1. **Semantic Search**: Natural language queries
2. **Context-Aware**: AI understands groundwater terminology
3. **Comprehensive Analysis**: Detailed reports with tables and insights
4. **Criticality Assessment**: Safe, Semi-Critical, Critical, Over-exploited
5. **Management Recommendations**: Data-driven suggestions
6. **Interactive Interface**: Easy-to-use chat interface

## 📁 File Structure

```
Ground_Water_Estimation_TechTrek/
├── ingris_chromadb_chatbot.py      # Main chatbot application
├── run_chromadb_chatbot.py         # Launcher script
├── test_chromadb_setup.py          # Setup verification
├── upload_to_chromadb.py           # Data upload script
├── ingris_rag_ready_complete.csv   # Source data (162,632 records)
├── chroma_db/                      # ChromaDB storage directory
└── CHROMADB_CHATBOT_README.md      # This file
```

## 🔧 Technical Details

### Vector Database
- **Engine**: ChromaDB (Persistent)
- **Collection**: ingris_groundwater_collection
- **Embeddings**: all-mpnet-base-v2 (768 dimensions)
- **Search**: Cosine similarity

### AI Integration
- **Model**: Gemini 2.0 Flash
- **Features**: Natural language processing, data analysis
- **Output**: Structured reports with tables and insights

### Search Pipeline
1. **Query Processing**: Natural language input
2. **Embedding Generation**: Convert to 768D vector
3. **Vector Search**: Find similar documents in ChromaDB
4. **Context Assembly**: Prepare relevant data for AI
5. **Response Generation**: AI analysis and recommendations

## 🚨 Troubleshooting

### Common Issues

1. **Dimension Mismatch Error**
   ```
   Collection expecting embedding with dimension of 768, got 384
   ```
   **Solution**: Use all-mpnet-base-v2 model (768D) instead of all-MiniLM-L6-v2 (384D)

2. **ChromaDB Connection Error**
   ```
   Collection not found
   ```
   **Solution**: Run `python upload_to_chromadb.py` first

3. **Gemini API Error**
   ```
   GEMINI_API_KEY not found
   ```
   **Solution**: Set environment variable with valid API key

4. **Memory Issues**
   ```
   Out of memory
   ```
   **Solution**: Reduce batch size in upload script or use smaller embedding model

### Performance Optimization

- **Search Speed**: ChromaDB provides fast vector search
- **Memory Usage**: Persistent storage reduces memory footprint
- **Scalability**: Can handle millions of records
- **Caching**: Streamlit caches components for better performance

## 📈 Performance Metrics

- **Upload Time**: ~2-3 hours for 162,632 records
- **Search Speed**: <100ms for typical queries
- **Memory Usage**: ~2-4GB during operation
- **Storage**: ~500MB for ChromaDB database
- **Accuracy**: High semantic similarity matching

## 🔮 Future Enhancements

1. **Multi-language Support**: Hindi, Tamil, Telugu queries
2. **Advanced Filtering**: State, district, year-based filters
3. **Visualization**: Charts and graphs for trends
4. **Export Features**: PDF/Excel report generation
5. **API Integration**: REST API for external access
6. **Real-time Updates**: Live data synchronization

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all prerequisites are installed
3. Ensure data upload completed successfully
4. Test with sample queries

## 📄 License

This project is part of the Ground Water Estimation TechTrek initiative.

---

**Note**: This chatbot provides comprehensive groundwater analysis based on the INGRIS dataset. Always verify critical decisions with official sources and local authorities.

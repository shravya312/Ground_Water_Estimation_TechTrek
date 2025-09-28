# ChromaDB Smart Upload Tracker

A comprehensive Streamlit application for uploading complete groundwater data to ChromaDB in the same format as Qdrant.

## 🚀 Features

- **Complete Data Upload**: Uploads the entire `ingris_rag_ready_complete.csv` dataset
- **Qdrant-Compatible Format**: Maintains the same payload structure as Qdrant
- **Smart Batching**: Processes data in configurable batches for optimal performance
- **Real-time Progress**: Live progress tracking and statistics
- **Search Testing**: Built-in search functionality to verify uploads
- **Error Handling**: Robust error handling and recovery
- **Collection Management**: Automatic collection creation and management

## 📋 Requirements

- Python 3.8+
- Streamlit
- ChromaDB
- Sentence Transformers
- Pandas
- NumPy

## 🛠️ Installation

1. **Install requirements:**
   ```bash
   pip install -r requirements_chromadb_uploader.txt
   ```

2. **Run the launcher:**
   ```bash
   python run_chromadb_uploader.py
   ```

3. **Or run directly:**
   ```bash
   streamlit run chromadb_smart_uploader.py --server.port 8502
   ```

## 📊 Data Structure

The uploader creates ChromaDB records with the following structure:

### Payload Fields (Qdrant-Compatible)
- `STATE`: State name (e.g., "KARNATAKA")
- `DISTRICT`: District name
- `Assessment_Year`: Assessment year
- `serial_number`: Serial number
- `combined_text`: Combined text for search
- `original_data`: Original CSV data as JSON
- All other groundwater parameters

### Vector Embeddings
- Uses `all-MiniLM-L6-v2` model (384 dimensions)
- Generated from `combined_text` field
- Compatible with existing search functionality

## 🔧 Usage

### 1. Upload Data
1. Open the Streamlit app
2. Upload your CSV file in the sidebar
3. Adjust batch size (recommended: 100)
4. Click "Start Upload"
5. Monitor progress in real-time

### 2. Monitor Progress
- **Total Processed**: Records processed so far
- **Successful**: Successfully uploaded records
- **Failed**: Failed uploads
- **Duration**: Total upload time

### 3. Test Search
- Use the search test feature to verify uploads
- Test with queries like "groundwater estimation in Karnataka"
- View results with metadata and distances

## 📈 Performance

### Recommended Settings
- **Batch Size**: 100-200 records
- **Memory**: 8GB+ RAM recommended
- **Storage**: ~2GB for complete dataset

### Upload Time Estimates
- **Small dataset** (1K records): ~2-3 minutes
- **Medium dataset** (10K records): ~15-20 minutes
- **Large dataset** (100K+ records): ~2-3 hours

## 🔍 Search Capabilities

The uploaded data supports:
- **Semantic Search**: Vector-based similarity search
- **Metadata Filtering**: Filter by state, district, year
- **Combined Text Search**: Full-text search across all fields
- **Hybrid Search**: Combination of vector and metadata search

## 🛡️ Error Handling

- **Connection Issues**: Automatic retry with exponential backoff
- **Memory Issues**: Configurable batch sizes
- **Data Validation**: Automatic data type conversion
- **Recovery**: Resume from last successful batch

## 📁 File Structure

```
Ground_Water_Estimation_TechTrek/
├── chromadb_smart_uploader.py      # Main Streamlit app
├── run_chromadb_uploader.py        # Launcher script
├── requirements_chromadb_uploader.txt  # Dependencies
├── CHROMADB_UPLOADER_README.md     # This documentation
└── chroma_db/                      # ChromaDB storage (created automatically)
```

## 🔧 Configuration

### Environment Variables
- No environment variables required
- Uses local ChromaDB instance
- Persistent storage in `./chroma_db/`

### Customization
- **Batch Size**: Adjust in the sidebar
- **Model**: Change in `initialize_model()` method
- **Collection Name**: Modify in `initialize_chromadb()` method

## 🚨 Troubleshooting

### Common Issues

1. **Memory Error**
   - Reduce batch size
   - Close other applications
   - Use a machine with more RAM

2. **Upload Fails**
   - Check CSV file format
   - Verify all required columns exist
   - Check disk space

3. **Search Returns No Results**
   - Verify data was uploaded successfully
   - Check collection statistics
   - Try different search queries

### Debug Mode
Enable debug logging by adding:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Data Verification

After upload, verify the data:

1. **Check Collection Stats**: Should show total record count
2. **Test Search**: Search for known states/districts
3. **Verify Metadata**: Check that all fields are populated
4. **Compare with CSV**: Ensure no data loss

## 🔄 Integration

This uploader is designed to work with the existing groundwater estimation system:

1. **Replace Qdrant**: Use ChromaDB instead of Qdrant
2. **Same API**: Maintains compatibility with existing search functions
3. **Better Performance**: Local storage, no network issues
4. **Reliable**: No cloud service dependencies

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all requirements are installed
3. Check the Streamlit logs for error messages
4. Ensure sufficient disk space and memory

## 🎯 Next Steps

After successful upload:
1. Update the main system to use ChromaDB
2. Test the search functionality
3. Verify state filtering works correctly
4. Deploy to production environment

---

**Note**: This uploader creates a local ChromaDB instance. For production use, consider using ChromaDB Cloud or a dedicated server instance.

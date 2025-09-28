# Smart INGRIS Complete Upload Tracker

## Overview
This tool uploads the complete INGRIS groundwater data (`ingris_rag_ready_complete.csv`) to Qdrant with the exact payload structure that matches your existing Qdrant collection.

## Features
- **Exact Payload Structure**: Matches the structure you showed from Qdrant
- **Smart Resume**: Continues from where it left off if interrupted
- **Progress Tracking**: Real-time upload progress and statistics
- **Karnataka Testing**: Built-in test to verify Karnataka data upload
- **Batch Processing**: Configurable batch sizes for optimal performance
- **Error Handling**: Robust error handling and recovery

## Payload Structure
The uploader creates payloads with this exact structure:
```json
{
  "STATE": "KARNATAKA",
  "DISTRICT": "Bangalore",
  "Assessment_Year": "2022",
  "serial_number": "12345",
  "text": "combined text...",
  "combined_text": "combined text...",
  "original_data": {...},
  "field1": "value1",
  "field2": "value2",
  ...
}
```

## Quick Start

### 1. Prerequisites
```bash
pip install streamlit pandas qdrant-client sentence-transformers python-dotenv
```

### 2. Environment Setup
Create `.env` file:
```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_api_key
GEMINI_API_KEY=your_gemini_key
```

### 3. Run the Uploader
```bash
python run_complete_uploader.py
```

Or directly:
```bash
streamlit run smart_upload_tracker_complete.py
```

## Usage

### 1. Setup Collection
- Click "Setup Collection" to create/verify the Qdrant collection
- Ensures correct vector size (768) and configuration

### 2. Start Upload
- Click "Start Complete Upload" to begin uploading
- Monitor progress in real-time
- Upload will resume from last batch if interrupted

### 3. Test Karnataka Data
- Use "Test Karnataka Data" to verify Karnataka records
- Ensures proper filtering and search functionality

### 4. Monitor Progress
- Real-time progress bar and statistics
- Batch-by-batch status updates
- Success/failure tracking

## Configuration

### Batch Size
- Default: 25 records per batch
- Adjustable via slider (10-100)
- Larger batches = faster upload, more memory usage

### Progress Tracking
- Progress saved to `upload_progress_complete.json`
- Automatic resume from last successful batch
- Manual progress reset available

## Data Structure

### Input CSV
- File: `ingris_rag_ready_complete.csv`
- Expected columns: `state`, `district`, `year`, `serial_number`, etc.
- All columns become individual payload fields

### Qdrant Collection
- Name: `ingris_groundwater_collection`
- Vector size: 768 (all-MiniLM-L6-v2)
- Distance: COSINE
- Payload: Structured fields + original data

## Troubleshooting

### Common Issues

1. **Qdrant Connection Error**
   - Check QDRANT_URL and QDRANT_API_KEY in .env
   - Verify Qdrant service is running

2. **CSV Not Found**
   - Ensure `ingris_rag_ready_complete.csv` exists
   - Check file path and permissions

3. **Vector Size Mismatch**
   - Delete existing collection and recreate
   - Use "Clear Collection" button

4. **Upload Interrupted**
   - Progress is automatically saved
   - Restart will resume from last batch
   - Use "Reset Progress" to start over

### Performance Tips

1. **Batch Size**
   - Start with 25 for stability
   - Increase to 50-100 for faster upload
   - Monitor memory usage

2. **Network**
   - Stable internet connection recommended
   - Qdrant Cloud may have rate limits

3. **Memory**
   - Large batches use more memory
   - Monitor system resources

## Verification

### Check Upload Success
1. Use "Check Collection Status" to see total points
2. Use "Test Karnataka Data" to verify Karnataka records
3. Check Qdrant dashboard for collection details

### Expected Results
- Total records: ~162,632 (from CSV)
- Karnataka records: ~1,186
- All states and districts properly indexed
- Search functionality working

## Next Steps

After successful upload:
1. **Test RAG System**: Use main2.py or main4.py
2. **Verify Queries**: Test Karnataka-specific queries
3. **Check Filtering**: Ensure state/district filtering works
4. **Monitor Performance**: Check search speed and accuracy

## Support

If you encounter issues:
1. Check the error messages in the interface
2. Verify all prerequisites are installed
3. Check Qdrant connection and credentials
4. Review the progress log for specific errors

## Files Created
- `upload_progress_complete.json`: Progress tracking
- Qdrant collection: `ingris_groundwater_collection`
- Logs: Check Streamlit console output

---

**Smart INGRIS Complete Upload Tracker** - Upload with exact Qdrant payload structure!

# Map Feature Testing Guide

## Overview
This guide will help you test the map feature integration that allows users to click on the India map to get groundwater analysis for specific locations.

## Features Implemented
- ✅ **Coordinate Detection**: Captures lat/lng when clicking on map
- ✅ **State Detection**: Uses Gemini API to identify Indian state from coordinates
- ✅ **Data Filtering**: Filters groundwater data by detected state
- ✅ **RAG Analysis**: Provides comprehensive groundwater analysis using RAG
- ✅ **Error Handling**: Graceful handling of edge cases

## Prerequisites
1. **Environment Variables**: Ensure `GEMINI_API_KEY` is set in `.env` file
2. **Data Files**: Ensure `master_groundwater_data.csv` exists
3. **Dependencies**: All required Python packages installed

## Testing Steps

### 1. Start Backend Server

**Option A: FastAPI (main.py)**
```bash
cd Ground_Water_Estimation_TechTrek
uvicorn main:app --reload --port 8000
```

**Option B: Flask (server.py)**
```bash
cd Ground_Water_Estimation_TechTrek
python server.py
```

### 2. Start Frontend
```bash
cd Ground_Water_Estimation_TechTrek/frontend
npm run dev
```

### 3. Run Automated Tests
```bash
cd Ground_Water_Estimation_TechTrek
python test_map_integration.py
```

### 4. Manual Testing

1. **Open Browser**: Go to `http://localhost:5173` (or port shown by Vite)
2. **Navigate to Map**: Go to the Groundwater page or wherever the map is displayed
3. **Test Coordinates**: Click on these locations to test:

| City | Coordinates | Expected State |
|------|-------------|----------------|
| Mumbai | 19.0760, 72.8777 | Maharashtra |
| Bangalore | 12.9716, 77.5946 | Karnataka |
| Delhi | 28.7041, 77.1025 | Delhi |
| Kolkata | 22.5726, 88.3639 | West Bengal |
| Chennai | 13.0827, 80.2707 | Tamil Nadu |
| Hyderabad | 17.3850, 78.4867 | Telangana |
| Jaipur | 26.9124, 75.7873 | Rajasthan |
| Chandigarh | 30.7333, 76.7794 | Punjab |

### 5. Expected Behavior

When you click on the map, you should see:

1. **Immediate Response**:
   - Coordinates displayed
   - Loading indicator appears
   - "Analyzing groundwater data..." message

2. **Analysis Results**:
   - State name detected
   - Number of data points found
   - Districts covered
   - Years of data available
   - Comprehensive groundwater analysis including:
     - Water availability and recharge rates
     - Extraction patterns and stage of extraction
     - Key metrics and statistics
     - Recommendations for groundwater management

3. **Visual Elements**:
   - Red marker placed on clicked location
   - Analysis results displayed below the map
   - Error messages if something goes wrong

## Troubleshooting

### Common Issues

1. **"Backend server not running"**
   - Solution: Start the backend server first
   - Check: `http://localhost:8000/health`

2. **"Could not determine state from coordinates"**
   - Solution: Check if coordinates are within India
   - Check: Gemini API key is set correctly

3. **"No groundwater data found for [state]"**
   - Solution: Check if state name matches data
   - Check: `master_groundwater_data.csv` exists and is loaded

4. **Map not loading**
   - Solution: Check Google Maps API key
   - Check: Internet connection
   - Check: Browser console for errors

5. **Slow response**
   - Solution: Check Gemini API quota
   - Solution: Check network connection
   - Solution: Check server logs for errors

### Debug Information

**Backend Logs**: Look for these messages in the console:
- `"Gemini found state: [state] for coordinates (lat, lng)"`
- `"Found state: [state] for coordinates (lat, lng)"`
- `"Error using Gemini for state detection: [error]"`

**Frontend Logs**: Check browser console for:
- `"Loading Google Maps..."`
- `"Google Maps loaded successfully"`
- `"Error analyzing location: [error]"`

## API Endpoints

### POST /analyze-location
**Request:**
```json
{
  "lat": 19.0760,
  "lng": 72.8777
}
```

**Response:**
```json
{
  "state": "Maharashtra",
  "data_points": 150,
  "summary": {
    "districts_covered": 25,
    "years_covered": [2020, 2021, 2022, 2023, 2024],
    "total_assessment_units": 150
  },
  "analysis": "Comprehensive groundwater analysis...",
  "key_metrics": {
    "Annual Ground water Recharge (ham) - Total - Total": {
      "mean": 1250.5,
      "min": 800.2,
      "max": 1800.7,
      "count": 150
    }
  }
}
```

## Performance Notes

- **State Detection**: Gemini API provides more accurate results than boundary mapping
- **Data Processing**: RAG service efficiently retrieves relevant groundwater data
- **Response Time**: Typically 2-5 seconds depending on data size and API response
- **Caching**: Consider implementing caching for frequently requested locations

## Success Criteria

✅ Map loads and displays India correctly
✅ Clicking on map captures coordinates
✅ State is detected accurately (90%+ accuracy)
✅ Groundwater data is retrieved for detected state
✅ Comprehensive analysis is generated
✅ Results are displayed clearly to user
✅ Error handling works for edge cases
✅ Performance is acceptable (< 10 seconds response time)

## Next Steps

After successful testing:
1. Deploy to production environment
2. Monitor performance and accuracy
3. Collect user feedback
4. Implement additional features like:
   - District-level analysis
   - Historical data trends
   - Interactive visualizations
   - Export functionality

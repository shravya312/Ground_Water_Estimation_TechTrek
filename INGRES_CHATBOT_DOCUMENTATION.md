# 🤖 INGRES CHATBOT DOCUMENTATION

## 📋 **OVERVIEW**

The INGRES ChatBOT is an AI-driven virtual assistant for the India Ground Water Resource Estimation System (INGRES). It provides intelligent query handling for groundwater data, real-time access to current and historical assessment results, interactive scientific diagrams and visualizations, and seamless integration with the INGRES database.

## 🎯 **KEY FEATURES**

### ✅ **Implemented Features**
- **Intelligent Query Handling**: Natural language processing for groundwater queries
- **Real-time Data Access**: Direct integration with INGRES database
- **Criticality Assessment**: Automatic categorization as Safe/Semi-Critical/Critical/Over-Exploited
- **Interactive Visualizations**: Scientific diagrams and charts
- **Location-based Analysis**: Coordinate-based groundwater analysis
- **Comprehensive Recommendations**: Actionable improvement suggestions
- **Water Quality Analysis**: Detection of contamination issues

### 🔄 **Pending Features**
- **Multilingual Support**: Regional language capabilities
- **Historical Data Access**: Time-series analysis
- **Advanced Analytics**: Trend analysis and predictions

## 🚀 **API ENDPOINTS**

### **1. Groundwater Query**
```http
POST /ingres/query
```

**Request Body:**
```json
{
    "query": "What is the groundwater status in Maharashtra?",
    "state": "MAHARASHTRA",
    "district": "MUMBAI",
    "assessment_unit": "MUMBAI_CITY",
    "include_visualizations": true,
    "language": "en"
}
```

**Response:**
```json
{
    "data": {
        "state": "MAHARASHTRA",
        "district": "MUMBAI",
        "extraction_stage": 45.2,
        "annual_recharge": 125000.5,
        "extractable_resource": 110000.0,
        "total_extraction": 50000.0,
        "future_availability": 60000.0,
        "rainfall": 2200.5,
        "total_area": 600000.0,
        "criticality_status": "Safe",
        "criticality_emoji": "🟢",
        "quality_issues": []
    },
    "criticality_status": "Safe",
    "criticality_emoji": "🟢",
    "numerical_values": {
        "extraction_stage": 45.2,
        "annual_recharge": 125000.5,
        "extractable_resource": 110000.0,
        "total_extraction": 50000.0,
        "future_availability": 60000.0,
        "rainfall": 2200.5,
        "total_area": 600000.0
    },
    "recommendations": [
        "Continue current water management practices",
        "Implement preventive measures to maintain current status",
        "Monitor groundwater levels regularly"
    ],
    "visualizations": [...],
    "comparison_data": {...},
    "quality_issues": []
}
```

### **2. Location Analysis**
```http
POST /ingres/location-analysis
```

**Request Body:**
```json
{
    "lat": 19.0760,
    "lng": 72.8777,
    "include_visualizations": true,
    "language": "en"
}
```

### **3. Get Available States**
```http
GET /ingres/states
```

**Response:**
```json
{
    "states": ["ANDHRA PRADESH", "ARUNACHAL PRADESH", ...],
    "count": 37
}
```

### **4. Get Districts by State**
```http
GET /ingres/districts/{state}
```

**Response:**
```json
{
    "state": "MAHARASHTRA",
    "districts": ["MUMBAI", "PUNE", "NAGPUR", ...],
    "count": 36
}
```

### **5. Criticality Summary**
```http
GET /ingres/criticality-summary
```

**Response:**
```json
{
    "total_districts": 4482,
    "criticality_distribution": {
        "safe": {
            "count": 3212,
            "percentage": 71.7
        },
        "semi_critical": {
            "count": 424,
            "percentage": 9.5
        },
        "critical": {
            "count": 846,
            "percentage": 18.9
        },
        "over_exploited": {
            "count": 702,
            "percentage": 15.7
        }
    },
    "national_average_extraction": 60.81
}
```

## 📊 **CRITICALITY ASSESSMENT**

### **Categorization Criteria**
- **🟢 Safe**: <70% extraction (71.7% of districts)
- **🟡 Semi-Critical**: 70-90% extraction (9.5% of districts)
- **🔴 Critical**: 90-100% extraction (18.9% of districts)
- **⚫ Over-Exploited**: ≥100% extraction (15.7% of districts)

### **Most Critical States**
1. **PUNJAB**: 163.6% average extraction (Over-Exploited)
2. **RAJASTHAN**: 145.9% average extraction (Over-Exploited)
3. **HARYANA**: 133.4% average extraction (Over-Exploited)
4. **DAMAN AND DIU**: 118.8% average extraction (Over-Exploited)
5. **DELHI**: 101.5% average extraction (Over-Exploited)

## 💧 **WATER QUALITY ANALYSIS**

### **Detected Parameters**
- **Arsenic (As)**: Contamination in some areas
- **Fluoride (F)**: Affecting certain regions
- **Salinity**: Partly saline conditions
- **Iron (Fe)**: Present in groundwater
- **Manganese (Mn)**: Found in some regions

### **Quality Recommendations**
- Arsenic removal technologies
- Fluoride removal systems
- Desalination or alternative water sources
- Iron removal filters
- Manganese removal treatment

## 🔧 **RECOMMENDATIONS BY STATUS**

### **Safe Areas (<70% extraction)**
- Continue current water management practices
- Implement preventive measures
- Monitor groundwater levels regularly
- Promote water conservation awareness

### **Semi-Critical Areas (70-90% extraction)**
- Implement water conservation measures
- Promote rainwater harvesting systems
- Optimize irrigation practices and crop patterns
- Monitor groundwater extraction rates closely
- Consider artificial recharge techniques

### **Critical Areas (90-100% extraction)**
- Immediate water conservation measures required
- Implement artificial recharge techniques
- Optimize crop patterns to reduce water demand
- Strict monitoring and regulation of extraction
- Emergency water management protocols

### **Over-Exploited Areas (≥100% extraction)**
- Emergency water management measures required
- Immediate artificial recharge implementation
- Strict extraction controls and regulations
- Crop diversification to water-efficient varieties
- Community awareness and participation programs
- Consider alternative water sources

## 📈 **VISUALIZATIONS**

### **Available Chart Types**
1. **Criticality Status Pie Chart**: Shows current status with color coding
2. **Resource Balance Bar Chart**: Compares recharge, extraction, and availability
3. **Extraction Efficiency Gauge**: Real-time extraction percentage with thresholds

### **Visualization Features**
- Interactive charts using Plotly
- Color-coded criticality levels
- Threshold indicators for safe/critical zones
- Comparative analysis charts
- Export capabilities

## 🗄️ **DATA SOURCES**

### **Primary Dataset**
- **File**: `master_groundwater_data.csv`
- **Records**: 4,639 districts
- **States**: 37 states and union territories
- **Assessment Year**: 2024
- **Columns**: 154 metrics covering all aspects of groundwater

### **Key Metrics**
- Stage of Ground Water Extraction (%)
- Annual Ground water Recharge (ham)
- Annual Extractable Ground water Resource (ham)
- Ground Water Extraction for all uses (ha.m)
- Net Annual Ground Water Availability for Future Use (ham)
- Rainfall (mm)
- Total Geographical Area (ha)
- Water Quality Parameters

## 🚀 **USAGE EXAMPLES**

### **Python Example**
```python
import requests

# Query groundwater data
response = requests.post("http://localhost:8000/ingres/query", json={
    "query": "What is the groundwater status in Chhattisgarh?",
    "state": "CHHATTISGARH",
    "include_visualizations": True,
    "language": "en"
})

data = response.json()
print(f"Status: {data['criticality_emoji']} {data['criticality_status']}")
print(f"Extraction: {data['numerical_values']['extraction_stage']:.1f}%")
```

### **JavaScript Example**
```javascript
const response = await fetch('http://localhost:8000/ingres/query', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        query: "What is the groundwater status in Punjab?",
        state: "PUNJAB",
        include_visualizations: true,
        language: "en"
    })
});

const data = await response.json();
console.log(`Status: ${data.criticality_emoji} ${data.criticality_status}`);
```

## 🧪 **TESTING**

### **Run Tests**
```bash
python test_ingres_chatbot.py
```

### **Test Coverage**
- Health check validation
- State and district data retrieval
- Criticality summary analysis
- Groundwater query processing
- Location-based analysis
- Critical states verification
- Specific query testing

## 🔧 **INSTALLATION & SETUP**

### **Prerequisites**
- Python 3.8+
- FastAPI
- Pandas
- Plotly
- Requests

### **Installation**
```bash
pip install fastapi pandas plotly requests
```

### **Start Server**
```bash
uvicorn main:app --reload --port 8000
```

### **Access API**
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **INGRES Endpoints**: http://localhost:8000/ingres/

## 📚 **INTEGRATION GUIDELINES**

### **Frontend Integration**
1. Use the `/ingres/query` endpoint for user queries
2. Display criticality status with appropriate emojis
3. Show numerical values in user-friendly format
4. Render visualizations using Plotly.js
5. Present recommendations as actionable items

### **Mobile App Integration**
1. Implement location-based analysis using GPS coordinates
2. Cache frequently accessed data
3. Provide offline access to critical information
4. Use push notifications for critical alerts

### **Web Portal Integration**
1. Embed the ChatBOT in existing INGRES portal
2. Provide seamless navigation between features
3. Maintain user session state
4. Support multiple languages

## 🎯 **FUTURE ENHANCEMENTS**

### **Planned Features**
- **Multilingual Support**: Hindi, Tamil, Telugu, Bengali, etc.
- **Historical Data**: Time-series analysis and trends
- **Predictive Analytics**: Future groundwater predictions
- **Mobile App**: Native mobile application
- **API Rate Limiting**: Enhanced security and performance
- **Caching**: Redis-based caching for improved performance
- **Real-time Updates**: WebSocket support for live data

### **Advanced Analytics**
- Machine learning models for prediction
- Anomaly detection in groundwater data
- Seasonal trend analysis
- Climate change impact assessment
- Policy recommendation engine

## 📞 **SUPPORT & CONTRIBUTION**

### **Documentation**
- API documentation available at `/docs`
- Code comments and docstrings
- Test cases and examples
- Integration guides

### **Contributing**
1. Fork the repository
2. Create feature branches
3. Add comprehensive tests
4. Submit pull requests
5. Follow coding standards

---

**The INGRES ChatBOT provides a comprehensive solution for groundwater data analysis, making complex scientific data accessible through intelligent querying and visualization. It serves as a bridge between technical groundwater assessment data and practical decision-making for water resource management.**

# 🌊 Groundwater Estimation TechTrek - SIH Project

An AI-powered web application for groundwater analysis and visualization across India using RAG techniques interactive visualizations, and location-based analysis.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### 1. Clone & Setup
```bash
git clone <repository-url>
cd Ground_Water_Estimation_TechTrek
```

### 2. Backend Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate.ps1 
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
# QDRANT_URL=your-qdrant-url
# QDRANT_API_KEY=your-qdrant-api-key
# GEMINI_API_KEY=your-gemini-api-key

# Prepare data
python excel_parser.py
python excel_ingestor.py
```

### 3. Frontend Setup
```bash
cd frontend
npm install

```

## 🏃‍♂️ Running the Application

### Terminal 1: Start Backend
```bash
cd Ground_Water_Estimation_TechTrek
uvicorn main:app --reload --port 8000
```

### Terminal 2: Start Frontend
```bash
cd Ground_Water_Estimation_TechTrek/frontend
npm run dev
```

### Access the Application
- **Main App**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health



## 📁 Project Structure
```
Ground_Water_Estimation_TechTrek/
├── main.py                    # FastAPI backend
├── app4.py                    # Streamlit app
├── excel_parser.py            # Data processing
├── requirements.txt           # Python dependencies
├── frontend/                  # React frontend
│   ├── src/
│   ├── package.json
│   └── .env
└── README.md
```

## 🔑 Required API Keys
- **Qdrant Cloud**: Vector database
- **Google Gemini**: AI chat functionality  
- **Google Maps**: Interactive mapping

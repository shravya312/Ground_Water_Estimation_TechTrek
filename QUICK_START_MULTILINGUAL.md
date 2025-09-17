# 🚀 Quick Start Guide - Multilingual Groundwater Chatbot

## Prerequisites

1. **Python 3.8+** installed
2. **Required dependencies** installed:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_multilingual.txt
   ```

## 🛠️ Setup

### 1. Fix Meta Tensor Issues (If Needed)
If you encounter meta tensor errors, run the fix script first:
```bash
python fix_meta_tensors.py
```

### 2. Set Up Environment Variables
Create a `.env` file with your API keys:
```bash
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### 3. Start the Application
```bash
streamlit run app3.py
```

## 🌐 Using Multilingual Features

### Language Selection
- Use the sidebar to select your preferred language
- The app supports 20+ languages including Hindi, Tamil, Telugu, etc.

### Asking Questions
1. **In English**: Ask directly in English
2. **In Other Languages**: Ask in your native language
   - The app will automatically detect the language
   - Translate your question to English for processing
   - Translate the answer back to your language

### Example Queries
- **English**: "What is the groundwater level in Karnataka?"
- **Hindi**: "कर्नाटक में भूजल स्तर क्या है?"
- **Tamil**: "கர்நாடகாவில் நிலத்தடி நீர் மட்டம் என்ன?"

## 🔧 Troubleshooting

### Common Issues

1. **Meta Tensor Error**:
   ```bash
   python fix_meta_tensors.py
   ```

2. **Translation Not Working**:
   - Check if translation libraries are installed
   - Use the "Translation Test" section in the sidebar

3. **Model Loading Failed**:
   - Click "🔄 Retry Model Initialization" in the sidebar
   - The app will work with BM25-only search if dense embeddings fail

### Status Indicators
- ✅ **Green**: Feature working properly
- ⚠️ **Yellow**: Feature has issues but app still works
- ❌ **Red**: Feature failed, using fallback

## 📊 Features

### Search Capabilities
- **Hybrid Search**: Combines dense embeddings + BM25 for best results
- **BM25-Only**: Fallback mode if dense embeddings fail
- **Multilingual**: Automatic language detection and translation

### Supported Languages
- English (en)
- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Bengali (bn)
- Marathi (mr)
- Gujarati (gu)
- Kannada (kn)
- Malayalam (ml)
- Punjabi (pa)
- And 10+ more languages

## 🎯 Tips for Best Results

1. **Be Specific**: Include state/district names in your questions
2. **Use Keywords**: Include terms like "groundwater", "extraction", "recharge"
3. **Check Status**: Monitor the status indicators in the sidebar
4. **Test Translation**: Use the translation test feature if needed

## 📁 File Structure

```
Ground_Water_Estimation_TechTrek/
├── app3.py                          # Main multilingual app
├── fix_meta_tensors.py              # Meta tensor fix script
├── requirements_multilingual.txt    # Additional dependencies
├── README_MULTILINGUAL.md          # Detailed documentation
├── TROUBLESHOOTING.md              # Troubleshooting guide
└── QUICK_START_MULTILINGUAL.md     # This file
```

## 🆘 Getting Help

1. **Check Status**: Look at the status indicators in the sidebar
2. **Use Translation Test**: Test translation capabilities
3. **Check Logs**: Look for error messages in the console
4. **Read Documentation**: Check README_MULTILINGUAL.md and TROUBLESHOOTING.md

## 🎉 Success!

If everything is working, you should see:
- ✅ Dense embeddings available (hybrid search enabled)
- 🌐 Translation methods available: googletrans, deep-translator, Gemini
- Language selection dropdown in the sidebar
- Translation test section for debugging

Happy querying! 🚀

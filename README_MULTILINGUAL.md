# Groundwater Chatbot - Multilingual Version (app3.py)

This is the multilingual version of the Groundwater Chatbot that supports multiple languages using Gemini for translation.

## Features

- **Multilingual Support**: Ask questions in Hindi, Bengali, Tamil, Telugu, Malayalam, Gujarati, Marathi, Punjabi, Kannada, Odia, Assamese, Urdu, Nepali, Sinhala, or English
- **Automatic Language Detection**: The system automatically detects the language of your question
- **Smart Translation**: Uses Gemini AI to translate queries to English for processing and responses back to your language
- **All Original Features**: Retains all features from app2.py including authentication, chat history, and hybrid search

## Supported Languages

- English (en)
- Hindi (hi)
- Bengali (bn)
- Tamil (ta)
- Telugu (te)
- Malayalam (ml)
- Gujarati (gu)
- Marathi (mr)
- Punjabi (pa)
- Kannada (kn)
- Odia (or)
- Assamese (as)
- Urdu (ur)
- Nepali (ne)
- Sinhala (si)

## Installation

1. Install the additional dependencies:
```bash
pip install -r requirements_multilingual.txt
```

2. Make sure you have all the original dependencies from `requirements.txt`

### Translation Libraries

The app supports multiple translation methods with automatic fallback:
- **googletrans**: Primary translation service (fast and reliable)
- **deep-translator**: Secondary fallback translation service
- **Gemini AI**: Final fallback using AI for translation

## Usage

1. Run the multilingual version:
```bash
streamlit run app3.py
```

2. Select your preferred language from the sidebar
3. Ask questions in your chosen language
4. The system will automatically detect the language and provide answers in the same language
5. Use the "Translation Test" section in the sidebar to test translation capabilities

## How It Works

1. **Language Detection**: Uses `langdetect` library to automatically detect the language of your question
2. **Query Translation**: Translates non-English queries to English for processing with the groundwater data
3. **Data Processing**: Processes the translated query using the same hybrid search (dense + sparse) and reranking
4. **Response Generation**: Uses Gemini to generate answers in the original language
5. **Display**: Shows both the original question and translated version (if applicable)

## Example Usage

- Ask in Hindi: "कर्नाटक में भूजल की स्थिति क्या है?"
- Ask in Tamil: "தமிழ்நாட்டில் நிலத்தடி நீர் நிலை என்ன?"
- Ask in Bengali: "পশ্চিমবঙ্গে ভূগর্ভস্থ জলের অবস্থা কী?"

The system will automatically detect the language and provide comprehensive answers in the same language.

## Notes

- The system uses Gemini AI for translation, so you need a valid GEMINI_API_KEY
- Translation quality depends on Gemini's capabilities
- All original features (authentication, chat history, etc.) work the same way
- The language detection is automatic - you don't need to manually select the language for each question

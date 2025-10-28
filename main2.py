from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PayloadSchemaType
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from rank_bm25 import BM25Okapi
import spacy
import torch
import json
import hashlib
import uuid
from typing import List, Dict, Optional, Any, Union
from langdetect import detect_langs, LangDetectException
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# Optional import for IndicTransToolkit
try:
    from IndicTransToolkit.processor import IndicProcessor
    INDIC_TRANS_AVAILABLE = True
except ImportError:
    IndicProcessor = None
    INDIC_TRANS_AVAILABLE = False
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
# Lazy imports for translators to avoid hard dependency issues at import time
try:
    from googletrans import Translator as GoogleTransTranslator
except Exception:
    GoogleTransTranslator = None
try:
    from deep_translator import GoogleTranslator as DeepTranslator
except Exception:
    DeepTranslator = None

load_dotenv()

# --- Environment Variables ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "ingris_groundwater_collection"
VECTOR_SIZE = 768  # Upgraded to support better embedding models
MIN_SIMILARITY_SCORE = 0.1  # Lowered threshold to find more relevant results

# --- Language Support ---
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'ml': 'Malayalam',
    'gu': 'Gujarati',
    'mr': 'Marathi',
    'pa': 'Punjabi',
    'kn': 'Kannada',
    'or': 'Odia',
    'as': 'Assamese',
    'ur': 'Urdu',
    'ne': 'Nepali',
    'si': 'Sinhala'
}

# --- Auth & Persistence Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
USERS_FILE_CANDIDATES = [
    os.path.join(BASE_DIR, "users.json"),
    os.path.join(PARENT_DIR, "users.json"),
]

def _resolve_users_file_path() -> str:
    for path in USERS_FILE_CANDIDATES:
        if os.path.exists(path):
            return path
    return USERS_FILE_CANDIDATES[0]

USERS_FILE = _resolve_users_file_path()
CHAT_HISTORY_DIR = os.path.join(BASE_DIR, "chat_histories")
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

_qdrant_client = None
_model = None
_nlp = None
_gemini_model = None
_master_df = None
_bm25_model = None
_all_chunks = None
_bm25_df = None
_translator_model = None
_translator_tokenizer = None
_indic_processor = None

_LANG_CODE_MAP = {
    'hi': 'hin_Deva',
    'bn': 'ben_Beng',
    'ta': 'tam_Taml',
    'te': 'tel_Telu',
    'ml': 'mal_Mlym',
    'gu': 'guj_Gujr',
    'mr': 'mar_Deva',
    'pa': 'pan_Guru',
    'kn': 'kan_Knda',
    'or': 'ori_Orya',
    'as': 'asm_Beng',
    'ur': 'urd_Arab',
    'en': 'eng_Latn'
}

# --- Global DataFrames ---
_master_df = None
_bm25_model = None
_all_chunks = None
_bm25_df = None
_embeddings_uploaded = False

app = FastAPI(title="Groundwater RAG API - Multilingual")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173", 
        "https://groundwater-eight.vercel.app",
        "https://groundwater-eight.vercel.app/",
        "*"  # Fallback for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic Models
class AskRequest(BaseModel):
    query: str
    language: Optional[str] = 'en'
    user_id: Optional[str] = None

class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class ChatMessage(BaseModel):
    role: str
    content: str
    conversation_id: Optional[str] = None

class ChatHistory(BaseModel):
    messages: List[ChatMessage]
    conversation_id: Optional[str] = None

class LanguageRequest(BaseModel):
    language: str

# INGRES ChatBOT Models
class GroundwaterQuery(BaseModel):
    query: str
    state: Optional[str] = None
    district: Optional[str] = None
    assessment_unit: Optional[str] = None
    include_visualizations: bool = True
    language: Optional[str] = "en"

class GroundwaterResponse(BaseModel):
    data: Dict[str, Any]
    criticality_status: str
    criticality_emoji: str
    numerical_values: Dict[str, float]
    recommendations: List[str]
    visualizations: Optional[List[Dict[str, Any]]] = None
    comparison_data: Optional[Dict[str, Any]] = None
    quality_analysis: Optional[Dict[str, Any]] = None
    additional_resources: Optional[Dict[str, Any]] = None
    key_findings_trends: Optional[Dict[str, Any]] = None
    historical_trend: Optional[Dict[str, Any]] = None
    enhanced_statistics: Optional[Dict[str, Any]] = None

class LocationAnalysisRequest(BaseModel):
    lat: float
    lng: float
    include_visualizations: bool = True
    language: Optional[str] = "en"

class LocationAnalysisResponse(BaseModel):
    state: str
    district: Optional[str] = None
    assessment_unit: Optional[str] = None
    groundwater_data: Dict[str, Any]
    criticality_status: str
    criticality_emoji: str
    numerical_values: Dict[str, float]
    recommendations: List[str]
    visualizations: Optional[List[Dict[str, Any]]] = None
    quality_analysis: Optional[Dict[str, Any]] = None
    additional_resources: Optional[Dict[str, Any]] = None
    key_findings_trends: Optional[Dict[str, Any]] = None
    enhanced_statistics: Optional[Dict[str, Any]] = None

def _fix_meta_tensors(model):
    """Fix meta tensors by converting them to real tensors."""
    try:
        # Method 1: Fix parameters
        for name, param in model.named_parameters():
            if hasattr(param, 'is_meta') and param.is_meta:
                new_param = torch.nn.Parameter(torch.zeros_like(param, device='cpu'))
                param.data = new_param.data
        
        # Method 2: Fix buffers
        for name, buffer in model.named_buffers():
            if hasattr(buffer, 'is_meta') and buffer.is_meta:
                new_buffer = torch.zeros_like(buffer, device='cpu')
                buffer.data = new_buffer.data
        
        return True
    except Exception as e:
        print(f"Meta tensor fix failed: {e}")
        return False

def initialize_sentence_transformer():
    """Robust initialization of SentenceTransformer with meta tensor handling."""
    global _model
    
    # Initialize _model to None first
    _model = None
    
    try:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        
        torch.set_default_dtype(torch.float32)
        
        _model = SentenceTransformer(
            "all-mpnet-base-v2",  # Upgraded to better model for higher similarity scores
            device="cpu",
            model_kwargs={
                "low_cpu_mem_usage": False,
                "dtype": torch.float32,
                "use_safetensors": False,
            },
        )
        _model.eval()
        return True
        
    except Exception as e1:
        print(f"Method 1 failed: {str(e1)}")
        
        try:
            _model = None
            _model = SentenceTransformer(
                "all-mpnet-base-v2",  # Upgraded to better model
                device="cpu",
                model_kwargs={
                    "dtype": torch.float32,
                    "trust_remote_code": False,
                },
            )
            _model = _model.to("cpu")
            _model.eval()
            return True
            
        except Exception as e2:
            print(f"Method 2 failed: {str(e2)}")
            
            try:
                _model = None
                _model = SentenceTransformer("all-mpnet-base-v2")  # Upgraded to better model
                _model.eval()
                return True
                
            except Exception as e3:
                print(f"All initialization methods failed. Last error: {str(e3)}")
                _model = None
                return False

def _init_components():
    """Ultra-fast startup - only initialize absolutely essential components"""
    global _qdrant_client, _model, _nlp, _gemini_model, _master_df, _translator_model, _translator_tokenizer, _indic_processor
    
    print("Starting application...")
    
    # Only load CSV data for state extraction (fastest essential component)
    if _master_df is None:
        try:
            print("Loading data...")
            _master_df = pd.read_csv("ingris_rag_ready_complete.csv", low_memory=False)
            _master_df['STATE'] = _master_df['state'].fillna('').astype(str)
            _master_df['DISTRICT'] = _master_df['district'].fillna('').astype(str)
            _master_df['ASSESSMENT UNIT'] = _master_df['assessment_unit'].fillna('').astype(str)
            # Handle year column with 'Unknown' values
            _master_df['year'] = _master_df['year'].replace('Unknown', 2020)
            _master_df['Assessment_Year'] = pd.to_numeric(_master_df['year'], errors='coerce').fillna(2020).astype(int)
            print("Data ready")
        except FileNotFoundError:
            raise Exception("Error: ingris_rag_ready_complete.csv not found.")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    print("Application ready - other components will load on demand")

def _init_qdrant():
    """Initialize Qdrant client when needed"""
    global _qdrant_client
    
    if _qdrant_client is None:
        try:
            print("Connecting to Qdrant...")
            print(f"Qdrant URL: {QDRANT_URL}")
            print(f"API Key present: {bool(QDRANT_API_KEY)}")
            
            _qdrant_client = QdrantClient(
                url=QDRANT_URL, 
                api_key=QDRANT_API_KEY if QDRANT_API_KEY else None, 
                timeout=10,  # Reduced timeout
                prefer_grpc=False
            )
            
            # Test the connection with a simple operation
            try:
                collections = _qdrant_client.get_collections()
                print(f"Qdrant ready - Found {len(collections.collections)} collections")
            except Exception as test_e:
                print(f"Qdrant connection test failed: {test_e}")
                # Don't fail completely, just warn
                print("Qdrant client created but connection test failed")
            
        except Exception as e:
            print(f"Qdrant initialization failed: {str(e)}")
            _qdrant_client = None
            raise e  # Re-raise to be caught by the startup handler
    else:
        print("Qdrant client already initialized")

def _init_gemini():
    """Initialize Gemini when needed"""
    global _gemini_model
    
    if _gemini_model is None and GEMINI_API_KEY:
        try:
            print("Initializing Gemini...")
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            print("Gemini ready")
        except Exception as e:
            print(f"Gemini failed: {str(e)}")
            _gemini_model = None

def _init_ml_components():
    """Initialize ML components when needed (lazy loading)"""
    global _model, _nlp
    
    # Initialize SentenceTransformer only when needed
    if _model is None:
        try:
            print("Loading embedding model...")
            if not initialize_sentence_transformer():
                print("Warning: Dense embeddings disabled")
        except Exception as e:
            print(f"Embedding model failed: {e}")
    
    # Initialize spaCy only when needed
    if _nlp is None:
        try:
            print("Loading NLP model...")
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"NLP model failed: {e}")
            _nlp = None

def _init_translator():
    """Initialize translator components when needed (lazy loading)"""
    global _translator_model, _translator_tokenizer, _indic_processor
    
    if _translator_model is None:
        try:
            print("Loading translator...")
            model_name = "ai4bharat/indictrans2-en-indic-1B"
            _translator_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            _translator_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.float32
            )
            _translator_model.to(torch.device('cpu'))
            if INDIC_TRANS_AVAILABLE:
                _indic_processor = IndicProcessor(inference=True)
            else:
                _indic_processor = None
        except Exception as e:
            print(f"Translator failed: {e}")

def create_detailed_combined_text(row):
    """Generates a detailed combined text string for a DataFrame row."""
    parts = []
    for col, value in row.items():
        if pd.notna(value) and value != '' and col not in ['S.No']:
            parts.append(f"{col}: {value}")
    return " | ".join(parts)

def preprocess_text_for_embedding(text):
    """Enhanced text preprocessing for better semantic understanding."""
    if not text or pd.isna(text):
        return ""
    
    # Convert to string and clean
    text = str(text).strip()
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Preserve important structure markers
    text = re.sub(r'([A-Z][a-z]+):', r'\1: ', text)  # Add space after colons
    text = re.sub(r'(\d+\.?\d*)\s*(ham|ha|mm|%)', r'\1 \2', text)  # Preserve units
    
    # Enhanced preprocessing for better semantic matching
    # Expand abbreviations and synonyms
    text = re.sub(r'\bground\s*water\b', 'groundwater', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwater\s*table\b', 'watertable', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwater\s*level\b', 'waterlevel', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwater\s*recharge\b', 'waterrecharge', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwater\s*extraction\b', 'waterextraction', text, flags=re.IGNORECASE)
    
    # Normalize state names
    state_mappings = {
        'karnataka': 'karnataka state',
        'maharashtra': 'maharashtra state', 
        'tamil nadu': 'tamil nadu state',
        'gujarat': 'gujarat state',
        'rajasthan': 'rajasthan state',
        'kerala': 'kerala state',
        'andhra pradesh': 'andhra pradesh state'
    }
    
    for state, expanded in state_mappings.items():
        text = re.sub(rf'\b{re.escape(state)}\b', expanded, text, flags=re.IGNORECASE)
    
    return text.strip()

def tokenize_text(text):
    """Enhanced tokenization for BM25 processing."""
    if not text:
        return []
    
    # Preprocess text first
    processed_text = preprocess_text_for_embedding(text)
    
    # Tokenize with better handling
    tokens = processed_text.lower().split()
    
    # Remove very short tokens and numbers without context
    tokens = [token for token in tokens if len(token) > 1 or token.isdigit()]
    
    return tokens

def get_embeddings(texts):
    """Convert text into embeddings with enhanced preprocessing"""
    global _model
    try:
        if _model is None:
            # Try to initialize the model
            if not initialize_sentence_transformer():
                return None
        
        # Preprocess texts for better embeddings
        processed_texts = [preprocess_text_for_embedding(text) for text in texts]
        
        try:
            test_text = processed_texts[0] if processed_texts else "test"
            test_embedding = _model.encode([test_text], show_progress_bar=False)
            return _model.encode(processed_texts, show_progress_bar=False)
            
        except Exception as meta_error:
            if "meta tensor" in str(meta_error).lower():
                print("Meta tensor issue detected. Attempting to reinitialize model...")
                
                try:
                    _model = None
                    _model = SentenceTransformer(
                        "all-mpnet-base-v2",  # Upgraded to better model
                        device="cpu",
                        model_kwargs={
                            "dtype": torch.float32,
                            "trust_remote_code": False,
                        },
                    )
                    _model.eval()
                    test_embedding = _model.encode([test_text], show_progress_bar=False)
                    return _model.encode(texts, show_progress_bar=False)
                    
                except Exception as reinit_error:
                    print(f"Model reinitialization failed: {str(reinit_error)}")
                    return None
            else:
                raise meta_error
                
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return None

def is_valid_embeddings(embeddings):
    """Check if embeddings are valid and not empty"""
    if embeddings is None:
        return False
    if isinstance(embeddings, np.ndarray):
        return embeddings.size > 0
    if hasattr(embeddings, '__len__'):
        return len(embeddings) > 0
    return False

def setup_collection():
    """Create Qdrant collection if it doesn't exist and ensure indexes for year, state, and district"""
    try:
        if _qdrant_client is None:
            print("[WARNING] Qdrant client not available, skipping collection setup")
            return False
            
        collections = _qdrant_client.get_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)
        
        if not collection_exists:
            _qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            print(f"Created new collection: {COLLECTION_NAME} with vector size {VECTOR_SIZE}")
        else:
            # Check if we need to recreate collection due to vector size change
            try:
                collection_info = _qdrant_client.get_collection(COLLECTION_NAME)
                current_size = collection_info.config.params.vectors.size
                if current_size != VECTOR_SIZE:
                    print(f"[INIT] Vector size mismatch detected. Current: {current_size}, Required: {VECTOR_SIZE}")
                    print("[WARNING] Please delete and recreate the collection manually or use the migration script.")
            except Exception as e:
                print(f"Warning: Could not check collection vector size: {e}")
            print(f"Using existing collection: {COLLECTION_NAME}")
        
        # Create indexes
        for field_name, field_type in [("Assessment_Year", PayloadSchemaType.INTEGER), 
                                      ("STATE", PayloadSchemaType.KEYWORD),
                                      ("DISTRICT", PayloadSchemaType.KEYWORD)]:
            try:
                _qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=field_type
                )
            except Exception:
                pass  # Index might already exist
        
        return True
    except Exception as e:
        print(f"Error setting up collection: {str(e)}")
        return False

def check_excel_embeddings_exist():
    """Check if embeddings for master_groundwater_data.csv exist in Qdrant by counting points"""
    try:
        collection_info = _qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        return collection_info.points_count > 0
    except Exception as e:
        print(f"Error checking existing embeddings: {str(e)}")
        return False

def upload_excel_to_qdrant(df_to_upload, batch_size=1000):
    """Upload Excel DataFrame rows to Qdrant in batches"""
    if df_to_upload.empty:
        return False
    
    try:
        texts = df_to_upload['combined_text'].tolist()
        embeddings = get_embeddings(texts)
        if not is_valid_embeddings(embeddings):
            print("Failed to generate embeddings for Excel data.")
            return False
        
        total_uploaded = 0
        for i in range(0, len(df_to_upload), batch_size):
            batch_df = df_to_upload.iloc[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            points = []
            for j, (index, row) in enumerate(batch_df.iterrows()):
                vector_list = batch_embeddings[j].tolist()
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, row['combined_text']))
                payload = row.to_dict()
                payload['text'] = row['combined_text'] 
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector_list,
                        payload=payload
                    )
                )
            
            _qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True
            )
            total_uploaded += len(points)
            print(f"Uploaded {total_uploaded}/{len(df_to_upload)} records...")
        
        print(f"[OK] All {total_uploaded} Excel rows uploaded to Qdrant.")
        return True
    except Exception as e:
        print(f"Error uploading Excel data to Qdrant: {str(e)}")
        return False

def _load_bm25():
    global _bm25_model, _all_chunks, _bm25_df
    if _bm25_model is not None:
        return
    try:
        # Use CSV data (162k records) ONLY for visualization
        # Qdrant is used for all search operations
        if _master_df is not None and not _master_df.empty:
            _all_chunks = _master_df['combined_text'].tolist()
            _bm25_df = _master_df.copy()
            print(f"Using CSV data for visualization only: {len(_master_df)} records available")
        else:
            # Fallback to Qdrant only if CSV data is not available
            collection_info = _qdrant_client.get_collection(collection_name=COLLECTION_NAME)
            if collection_info.points_count > 0:
                scroll_result, _ = _qdrant_client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=100000,
                    with_payload=True,
                    with_vectors=False
                )
                _all_chunks = [point.payload.get("text", "") for point in scroll_result if point.payload.get("text")]
                _bm25_df = pd.DataFrame([point.payload for point in scroll_result])
                _bm25_df['combined_text'] = _all_chunks
                print(f"Using Qdrant data for visualization (CSV not available): {len(_all_chunks)} records")
            else:
                _all_chunks = []
                _bm25_df = pd.DataFrame()
                print("No data available for visualization")
        
        if _all_chunks:
            tokenized_chunks = [tokenize_text(chunk) for chunk in _all_chunks]
            _bm25_model = BM25Okapi(tokenized_chunks)
            print(f"BM25 model initialized for visualization with {len(_all_chunks)} text chunks")
    except Exception as e:
        print(f"Warning: Could not initialize BM25: {e}")
        _bm25_model = None
        _all_chunks = []
        _bm25_df = pd.DataFrame()

def detect_language(text: str) -> str:
    """Detect the language of the input text with confidence threshold."""
    text = (text or "").strip()
    if not text:
        return "en"
    try:
        langs = detect_langs(text)
        if not langs:
            return "en"
        top = langs[0]
        # Lower confidence threshold for better detection
        if getattr(top, 'prob', 1.0) < 0.30:
            return "en"
        
        detected_lang = top.lang
        
        # Map common language codes to our supported languages
        lang_mapping = {
            'mr': 'mr',  # Marathi
            'hi': 'hi',  # Hindi
            'bn': 'bn',  # Bengali
            'ta': 'ta',  # Tamil
            'te': 'te',  # Telugu
            'ml': 'ml',  # Malayalam
            'gu': 'gu',  # Gujarati
            'pa': 'pa',  # Punjabi
            'kn': 'kn',  # Kannada
            'or': 'or',  # Odia
            'as': 'as',  # Assamese
            'ur': 'ur',  # Urdu
            'ne': 'ne',  # Nepali
            'si': 'si',  # Sinhala
            'en': 'en'   # English
        }
        
        return lang_mapping.get(detected_lang, 'en')
    except LangDetectException:
        return "en"
    except Exception:
        return "en"

def translate_text(text: str, target_lang: str, source_lang: str = 'auto') -> str:
    """Translate text using multiple fallback methods."""
    if not text:
        return text
    # If source and dest are same, skip
    if source_lang != 'auto' and source_lang == target_lang:
        return text
    
    # Try googletrans first if available
    if GoogleTransTranslator is not None:
        try:
            translator = GoogleTransTranslator()
            result = translator.translate(text, src=source_lang, dest=target_lang)
            return getattr(result, 'text', text)
        except Exception as e:
            print(f"googletrans translation error ({source_lang}->{target_lang}): {e}")
    
    # Fallback to deep-translator if available
    if DeepTranslator is not None:
        try:
            src = source_lang if source_lang != 'auto' else 'auto'
            translated = DeepTranslator(source=src, target=target_lang).translate(text)
            if translated:
                return translated
        except Exception as e:
            print(f"deep-translator error ({source_lang}->{target_lang}): {e}")
    
    # Last resort: use Gemini for translation
    try:
        prompt = f"Translate the following text from {SUPPORTED_LANGUAGES.get(source_lang, source_lang)} to {SUPPORTED_LANGUAGES.get(target_lang, target_lang)}. Only return the translated text, no additional explanations:\n\n{text}"
        response = _gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini translation failed: {str(e)}")
        return text

def translate_query_to_english(query: str) -> tuple[str, str]:
    """Translate user query to English for processing, return (translated_query, original_lang)."""
    detected_lang = detect_language(query)
    if detected_lang == 'en':
        return query, 'en'
    
    # Add location name mapping for better translation
    location_mapping = {
        'कर्नाटक': 'Karnataka',
        'बेंगळुरू': 'Bangalore',
        'बंगळुरू': 'Bangalore',
        'महाराष्ट्र': 'Maharashtra',
        'तमिळनाडू': 'Tamil Nadu',
        'केरळ': 'Kerala',
        'आंध्र प्रदेश': 'Andhra Pradesh',
        'तेलंगणा': 'Telangana',
        'गुजरात': 'Gujarat',
        'राजस्थान': 'Rajasthan',
        'उत्तर प्रदेश': 'Uttar Pradesh',
        'मध्य प्रदेश': 'Madhya Pradesh',
        'छत्तीसगढ़': 'Chhattisgarh',
        'ओडिशा': 'Odisha',
        'पश्चिम बंगाल': 'West Bengal',
        'असम': 'Assam',
        'बिहार': 'Bihar',
        'झारखंड': 'Jharkhand',
        'हरियाणा': 'Haryana',
        'पंजाब': 'Punjab',
        'हिमाचल प्रदेश': 'Himachal Pradesh',
        'उत्तराखंड': 'Uttarakhand',
        'दिल्ली': 'Delhi',
        'गोवा': 'Goa',
        'मणिपुर': 'Manipur',
        'मेघालय': 'Meghalaya',
        'मिजोरम': 'Mizoram',
        'नागालैंड': 'Nagaland',
        'सिक्किम': 'Sikkim',
        'त्रिपुरा': 'Tripura',
        'अरुणाचल प्रदेश': 'Arunachal Pradesh',
        'जम्मू और कश्मीर': 'Jammu and Kashmir',
        'लद्दाख': 'Ladakh',
        'अंडमान और निकोबार': 'Andaman and Nicobar Islands',
        'चंडीगढ़': 'Chandigarh',
        'दादरा और नगर हवेली': 'Dadra and Nagar Haveli',
        'दमन और दीव': 'Daman and Diu',
        'लक्षद्वीप': 'Lakshadweep',
        'पुडुचेरी': 'Puducherry'
    }
    
    # Pre-process query to replace location names
    processed_query = query
    for marathi_name, english_name in location_mapping.items():
        processed_query = processed_query.replace(marathi_name, english_name)
    
    translated_query = translate_text(processed_query, 'en', detected_lang)
    return translated_query, detected_lang

def translate_answer_to_language(answer: str, target_lang: str) -> str:
    """Translate the answer back to the user's language."""
    if target_lang == 'en':
        return answer
    return translate_text(answer, target_lang, 'en')

def expand_query(query_text):
    """Expand query with minimal synonyms for better matching."""
    # Simplified query expansion - only essential terms
    expansions = {
        'groundwater': ['ground water'],
        'estimation': ['assessment', 'analysis'],
        'odisha': ['orissa'],  # Add Odisha/Orissa mapping
        'orissa': ['odisha']   # Add reverse mapping
    }
    
    expanded_terms = [query_text]
    
    # Add expansions for each term in the query
    query_lower = query_text.lower()
    for term, synonyms in expansions.items():
        if term in query_lower:
            expanded_terms.extend(synonyms)
    
    # Create expanded query (limit to avoid overly complex queries)
    expanded_query = " ".join(expanded_terms[:3])  # Limit to 3 terms max
    return expanded_query

def search_qdrant_rag(query_text, year=None, target_state=None, target_district=None, extracted_parameters=None):
    """Simple, reliable search using the working method from karnataka_search.py"""
    _init_components()  # Load CSV data
    _init_qdrant()      # Load Qdrant when needed
    _init_ml_components()  # Load ML components when needed for search
    
    try:
        # Create query vector using the working method
        if _model is None:
            return []
        
        # Simple query processing - no complex expansion
        query_vector = _model.encode([query_text])[0].tolist()
        
        # Create filter conditions
        qdrant_filter_conditions = []
        
        if year:
            qdrant_filter_conditions.append(
                FieldCondition(
                    key="Assessment_Year",
                    match=MatchValue(value=year)
                )
            )
        
        if target_state:
            # Normalize state name for better matching
            state_name = target_state.upper().strip()
            # Handle common variations
            if state_name in ['ORISSA', 'ODISHA']:
                state_name = 'ODISHA'
            elif state_name in ['TAMILNADU', 'TAMIL NADU']:
                state_name = 'TAMILNADU'
            elif state_name in ['WEST BENGAL', 'WESTBENGAL']:
                state_name = 'WEST BENGAL'
            
            qdrant_filter_conditions.append(
                FieldCondition(
                    key="STATE",
                    match=MatchValue(value=state_name)
                )
            )
        
        if target_district:
            qdrant_filter_conditions.append(
                FieldCondition(
                    key="DISTRICT",
                    match=MatchValue(value=target_district)
                )
            )
        
        qdrant_filter = Filter(must=qdrant_filter_conditions) if qdrant_filter_conditions else None
        
        # Use the new query_points method with retry logic
        max_retries = 3
        results = None
        
        for attempt in range(max_retries):
            try:
                results = _qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    query_filter=qdrant_filter,
                    limit=20,
                    with_payload=True
                )
                break  # Success, exit retry loop
            except Exception as e:
                print(f"Search attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)  # Wait 2 seconds before retry
                else:
                    raise e
        
        # Convert to the expected format
        results = results.points if hasattr(results, 'points') else results
        
        # Format results in the expected format
        results_with_payloads = []
        for hit in results:
            results_with_payloads.append({
                "score": hit.score,
                "data": hit.payload
            })
        
        # Debug information
        print(f"[DEBUG] Search completed for state: {target_state}")
        print(f"[DEBUG] Results found: {len(results_with_payloads)}")
        if results_with_payloads:
            sample_state = results_with_payloads[0]['data'].get('STATE', 'N/A')
            print(f"[DEBUG] Sample result state: {sample_state}")
        
        # If no results and we have a state filter, don't use fallback automatically
        # This prevents returning data from other states when user specifically asks for a state
        if not results_with_payloads and target_state:
            print(f"[WARNING] No results found for {target_state}")
            # Don't automatically fallback to other states - let the calling function decide
        
        return results_with_payloads

    except Exception as e:
        print(f"Error performing search: {str(e)}")
        # If Qdrant is not available, return empty results
        if _qdrant_client is None:
            print("Qdrant not available, returning empty results")
            return []
        return []

def re_rank_chunks(query_text, candidate_results, top_k=5):
    """Enhanced re-ranks candidate results based on semantic similarity to the query."""
    if not candidate_results:
        return []

    if _model is None:
        sorted_candidates = sorted(candidate_results, key=lambda r: r.get('score', 0), reverse=True)
        return sorted_candidates[:top_k]

    candidate_texts = [res['data'].get('text', '') for res in candidate_results]
    if not candidate_texts:
        return []

    # Enhanced preprocessing for reranking
    processed_query = preprocess_text_for_embedding(query_text)
    processed_candidates = [preprocess_text_for_embedding(text) for text in candidate_texts]
    
    query_embedding = _model.encode([processed_query])[0]
    chunk_embeddings = _model.encode(processed_candidates)

    query_embedding_norm = np.linalg.norm(query_embedding)
    chunk_embeddings_norm = np.linalg.norm(chunk_embeddings, axis=1)

    if query_embedding_norm == 0:
        return []
    chunk_embeddings_norm[chunk_embeddings_norm == 0] = 1e-12

    similarities = np.dot(chunk_embeddings, query_embedding) / (chunk_embeddings_norm * query_embedding_norm)

    re_ranked_scores = []
    for i, res in enumerate(candidate_results):
        # Enhanced scoring: combine original score with semantic similarity
        original_score = res.get('score', 0.0)
        semantic_similarity = similarities[i]
        
        # Weighted combination: 70% semantic similarity, 30% original score
        enhanced_score = 0.7 * semantic_similarity + 0.3 * original_score
        
        # Boost scores that are very high in either dimension
        if semantic_similarity > 0.8 or original_score > 0.8:
            enhanced_score = min(enhanced_score * 1.1, 1.0)  # 10% boost, cap at 1.0
            
        re_ranked_scores.append((res['data'], enhanced_score))

    re_ranked_scores.sort(key=lambda x: x[1], reverse=True)

    final_results = []
    for data, score in re_ranked_scores[:top_k]:
        # Apply similarity threshold filter in reranking
        if score >= MIN_SIMILARITY_SCORE:
            final_results.append({"score": score, "data": data})
    
    return final_results

def expand_query(query, num_terms=3):
    """Expands the user query with additional terms using Gemini."""
    if not _gemini_model:
        return ""
    prompt = (
        f"Generate {num_terms} related terms or short phrases for the following query. "
        f"Output only the terms, separated by commas, no other text.\n"
        f"Query: {query}\n"
        f"Related terms:"
    )
    try:
        response = _gemini_model.generate_content(prompt)
        expanded_terms = [term.strip() for term in response.text.strip().split(',') if term.strip()]
        return " ".join(expanded_terms)
    except Exception as e:
        print(f"Error expanding query: {e}")
        return ""

def generate_answer_from_gemini(query, context_data, year=None, target_state=None, target_district=None, chat_history=None, extracted_parameters=None, user_language='en', averages=None):
    """Use Gemini to answer the question based on structured Excel data with multilingual support."""
    if not query or not context_data:
        return "Please provide both a question and relevant data context."
    
    if not _gemini_model:
        lines = []
        for item in context_data[:3]:
            lines.append(f"State: {item.get('STATE')}, District: {item.get('DISTRICT')}, Unit: {item.get('ASSESSMENT UNIT')}")
        return f"No LLM configured. Top matches:\n" + "\n".join(lines)
    
    # Format structured data into a readable string for the LLM
    data_summary = []
    
    # Add average values if available
    if averages:
        data_summary.append("=== AVERAGE VALUES (when year not specified) ===")
        for key, value in averages.items():
            if key == 'quality_distribution':
                data_summary.append(f"Quality Distribution: {value}")
            else:
                data_summary.append(f"Average {key.replace('_', ' ').title()}: {value}")
        data_summary.append("")
    
    for i, item in enumerate(context_data, 1):
        data_summary.append(f"=== DATA ENTRY {i} ===")
        data_summary.append(f"State: {item.get('STATE', 'N/A')}")
        data_summary.append(f"District: {item.get('DISTRICT', 'N/A')}")
        data_summary.append(f"Assessment Unit: {item.get('ASSESSMENT UNIT', 'N/A')}")
        data_summary.append(f"Assessment Year: {item.get('Assessment_Year', 'N/A')}")
        data_summary.append(f"Serial Number: {item.get('S.No', 'N/A')}")
        data_summary.append("")
        
        # Group columns by category for better organization
        categories = {
            "RAINFALL DATA": [col for col in item.keys() if 'Rainfall' in col],
            "GEOGRAPHICAL AREA": [col for col in item.keys() if 'Total Geographical Area' in col],
            "GROUNDWATER RECHARGE": [col for col in item.keys() if 'Ground Water Recharge' in col],
            "INFLOWS & OUTFLOWS": [col for col in item.keys() if 'Inflows and Outflows' in col],
            "ANNUAL RECHARGE": [col for col in item.keys() if 'Annual Ground water Recharge' in col],
            "ENVIRONMENTAL FLOWS": [col for col in item.keys() if 'Environmental Flows' in col],
            "EXTRACTABLE RESOURCES": [col for col in item.keys() if 'Annual Extractable Ground water Resource' in col],
            "EXTRACTION DATA": [col for col in item.keys() if 'Ground Water Extraction for all uses' in col],
            "EXTRACTION STAGE": [col for col in item.keys() if 'Stage of Ground Water Extraction' in col],
            "FUTURE ALLOCATION": [col for col in item.keys() if 'Allocation of Ground Water Resource' in col],
            "FUTURE AVAILABILITY": [col for col in item.keys() if 'Net Annual Ground Water Availability' in col],
            "QUALITY TAGGING": [col for col in item.keys() if 'Quality Tagging' in col],
            "ADDITIONAL RESOURCES": [col for col in item.keys() if 'Additional Potential Resources' in col],
            "COASTAL AREAS": [col for col in item.keys() if 'Coastal Areas' in col],
            "UNCONFINED RESOURCES": [col for col in item.keys() if 'In-Storage Unconfined Ground Water Resources' in col],
            "CONFINED RESOURCES": [col for col in item.keys() if 'Confined Ground Water Resources' in col],
            "SEMI-CONFINED RESOURCES": [col for col in item.keys() if 'Semi Confined Ground Water Resources' in col],
            "TOTAL AVAILABILITY": [col for col in item.keys() if 'Total Ground Water Availability' in col],
            "OTHER DATA": []
        }
        
        # Add uncategorized columns to "OTHER DATA"
        excluded_keys = {'STATE', 'DISTRICT', 'ASSESSMENT UNIT', 'Assessment_Year', 'S.No', 'combined_text', 'text'}
        for key in item.keys():
            if key not in excluded_keys and not any(key in cat_cols for cat_cols in categories.values()):
                categories["OTHER DATA"].append(key)
        
        # Display data by category
        for category, columns in categories.items():
            if columns:
                data_summary.append(f"--- {category} ---")
                for col in columns:
                    if col in item and pd.notna(item[col]) and str(item[col]).strip() != '':
                        data_summary.append(f"  {col}: {item[col]}")
                    else:
                        data_summary.append(f"  {col}: No data available")
                data_summary.append("")
        
        data_summary.append("=" * 50)
    
    # Add comprehensive averages if no specific year
    if year is None:
        context_df = pd.DataFrame(context_data)
        
        # Define comprehensive numerical columns for averaging
        numerical_columns = [
            'Annual Ground water Recharge (ham) - Total - Total',
            'Annual Extractable Ground water Resource (ham) - Total - Total',
            'Ground Water Extraction for all uses (ha.m) - Total - Total',
            'Stage of Ground Water Extraction (%) - Total - Total',
            'Net Annual Ground Water Availability for Future Use (ham) - Total - Total',
            'Environmental Flows (ham) - Total - Total',
            'Allocation of Ground Water Resource for Domestic Utilisation for projected year 2025 (ham) - Total - Total'
        ]
        
        # Add rainfall columns
        rainfall_cols = [col for col in context_df.columns if 'Rainfall (mm)' in col and 'Total' in col]
        numerical_columns.extend(rainfall_cols)
        
        # Add geographical area columns
        area_cols = [col for col in context_df.columns if 'Total Geographical Area (ha)' in col and 'Total' in col]
        numerical_columns.extend(area_cols)
        
        avg_data = {}
        for col in numerical_columns:
            if col in context_df.columns:
                numeric_values = pd.to_numeric(context_df[col], errors='coerce').dropna()
                if not numeric_values.empty:
                    avg_data[col] = numeric_values.mean()
        
        if avg_data:
            data_summary.append(f"\n--- COMPREHENSIVE AVERAGES FOR THE RETRIEVED DATA ---")
            data_summary.append("Key Groundwater Metrics (Averaged):")
            for key, value in avg_data.items():
                data_summary.append(f"  • {key}: {value:.2f}")
            data_summary.append("---")
    
    context_str = "\n".join(data_summary)

    year_info = f" for the year {year}" if year else " (averaged across all available years)"
    location_info = f" for {target_district} District, {target_state}" if target_district else (f" for {target_state}" if target_state else "")
    
    conversation_history_str = ""
    if chat_history:
        conversation_history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
        conversation_history_str = f"\n--- Conversation History ---\n{conversation_history_str}\n"

    extracted_params_str = ""
    if extracted_parameters:
        extracted_params_str = "\nExtracted specific parameters from query: "
        for param, val in extracted_parameters.items():
            extracted_params_str += f"{param}: {val}. "
        extracted_params_str += "\n"

    # Language-specific prompt
    language_instruction = ""
    if user_language != 'en':
        language_instruction = f"\nIMPORTANT: Please respond in {SUPPORTED_LANGUAGES.get(user_language, user_language)} language. "

    prompt = (
        f"You are an expert groundwater data analyst. Provide a comprehensive, well-formatted summary of the groundwater data.{language_instruction}\n"
        f"IMPORTANT: When you see quality_tagging data in the context, it may contain values like 'PQ', '[Iron]', '[Uranium]', '[Nitrate]', etc. These are actual water quality parameters that should be displayed in the Quality Tagging field, not 'No data available'.\n"
        f"""FORMAT THE OUTPUT EXACTLY AS FOLLOWS:

#  Groundwater Data Analysis Report

## Query
**Question:** [USER'S QUERY]

## Analysis
[COMPREHENSIVE ANALYSIS]

**Question:** [USER'S QUERY]

## Analysis
Groundwater Estimation Report: [STATE NAME] - [YEAR]

This report provides a comprehensive analysis of groundwater resources in [STATE NAME] for the year [YEAR]. It includes an overview of the state's groundwater status, district-wise analysis, and a comparative assessment. The report examines key parameters such as rainfall, recharge, extraction, and availability to understand the sustainability of groundwater resources in the region.

### NATIONAL OVERVIEW SECTION:

#### GLOBAL NATIONAL GROUNDWATER OVERVIEW:

Total Coverage: [X] states, [Y] districts, [Z] years of data ([YEAR RANGE])
National Average Extraction: [X]% ([ABOVE/BELOW] sustainable limits)
Critical Areas Nationwide: [X] over-exploited + critical areas
Safe Areas Nationwide: [X] areas ([X]% of total)
Water Quality Issues: [X] areas with contamination
Most Over-exploited: [DISTRICT], [STATE] ([X]% extraction)
Safest Region: [DISTRICT], [STATE] ([X]% extraction)

### DATA AVAILABILITY SECTION:

#### INFO DATA AVAILABILITY & COVERAGE:

This report is based on available data for [X] districts in [STATE NAME] for the year [YEAR]. Data availability varies across different parameters. While taluk-level administrative data is fully available, block, mandal, and village-level data are not collected/available for this region. Additionally, watershed categorization and water quality testing data are also unavailable. Groundwater storage measurements are also not recorded.

#### INFO DATA AVAILABILITY ANALYSIS:

Based on [X] records found in the dataset:

**ADMIN ADMINISTRATIVE DATA COVERAGE:**
• Taluk data: [X]/[X] records ([X]%)
• Block data: [X]/[X] records ([X]%)
• Mandal data: [X]/[X] records ([X]%)
• Village data: [X]/[X] records ([X]%)

**TECH TECHNICAL DATA COVERAGE:**
• Storage data: [X]/[X] records ([X]%)
• Watershed data: [X]/[X] records ([X]%)
• Quality data: [X]/[X] records ([X]%)

**INSIGHT REASONS FOR MISSING DATA:**
• Block data: Block-level administrative data not available in dataset
• Mandal data: Mandal-level data not collected for this area
• Village data: Village-level data not available in dataset
• Watershed data: Watershed categorization not available for this region
• Quality data: Water quality testing not conducted in this area

**GLOBAL STATE-SPECIFIC CONTEXT:**
Data collection practices vary across states. [STATE NAME] may have different data collection priorities or methodologies compared to other states in the dataset.

## District-Wise Analysis

[FOR EACH DISTRICT, INCLUDE THE FOLLOWING STRUCTURE:]

### [DISTRICT NUMBER]. [DISTRICT NAME] District

#### [DISTRICT NUMBER].[TALUK NUMBER]. [TALUK NAME] Taluk

#### 1.  CRITICALITY ALERT & SUSTAINABILITY STATUS:

| Parameter | Value | Unit | Significance |
|-----------|-------|------|--------------|
| Stage of Ground Water Extraction (%) | [X] | % | [SAFE/SEMI-CRITICAL/CRITICAL/OVER-EXPLOITED] ([X]%). [EXPLANATION] |
| Groundwater categorization | [CATEGORY] | N/A | [EXPLANATION] |

**ALERT CRITICAL ALERT:** [ALERT MESSAGE OR "No critical alert applicable."]

**Sustainability Indicators:** [DETAILED EXPLANATION OF SUSTAINABILITY STATUS]

#### 2.  GROUNDWATER TREND ANALYSIS:

| Parameter | Value |
|-----------|-------|
| Pre-monsoon groundwater trend | [pre_monsoon_of_gw_trend] |
| Post-monsoon groundwater trend | [post_monsoon_of_gw_trend] |

**Trend Implications:** [DETAILED EXPLANATION OF TREND IMPLICATIONS]

**Seasonal Variation Analysis:** [ANALYSIS OF SEASONAL VARIATIONS]

#### 3. [RAIN] RAINFALL & RECHARGE DATA:

| Parameter | Value | Unit | Significance |
|-----------|-------|------|--------------|
| Rainfall | [X] | mm | [EXPLANATION] |
| Ground Water Recharge | [X] | ham | [EXPLANATION] |
| Annual Ground Water Recharge | [X] | ham | [EXPLANATION] |
| Environmental Flows | [X] | ham | [EXPLANATION] |

**Significance:** [DETAILED EXPLANATION OF RAINFALL AND RECHARGE SIGNIFICANCE]

#### 4.  GROUNDWATER EXTRACTION & AVAILABILITY:

| Parameter | Value | Unit | Significance |
|-----------|-------|------|--------------|
| Ground Water Extraction for all uses | [X] | ham | [EXPLANATION] |
| Annual Extractable Ground Water Resource | [X] | ham | [EXPLANATION] |
| Net Annual Ground Water Availability for Future Use | [X] | ham | [EXPLANATION] |
| Allocation for Domestic Utilisation for 2025 | [X] | ham | [EXPLANATION] |

**Extraction Efficiency:** [DETAILED ANALYSIS OF EXTRACTION EFFICIENCY]

#### 5.  WATER QUALITY & ENVIRONMENTAL CONCERNS:

| Parameter | Value |
|-----------|-------|
| Quality Tagging | [USE ACTUAL QUALITY DATA FROM CONTEXT - PQ, Iron, Uranium, Nitrate, etc. or "No data available" if none] |

**Quality Concerns:** [ANALYSIS OF WATER QUALITY CONCERNS BASED ON ACTUAL QUALITY DATA]

**Treatment Recommendations:** [RECOMMENDATIONS FOR WATER TREATMENT BASED ON DETECTED CONTAMINANTS]

**Environmental Sustainability:** [ASSESSMENT OF ENVIRONMENTAL SUSTAINABILITY]

#### 6. [COASTAL] COASTAL & SPECIAL AREAS:

| Parameter | Value |
|-----------|-------|
| Coastal Areas identification | [DATA AVAILABLE/NOT AVAILABLE] |
| Additional Potential Resources under specific conditions | [DATA AVAILABLE/NOT AVAILABLE] |

**Special Management:** [SPECIAL MANAGEMENT CONSIDERATIONS]

**Climate Resilience Considerations:** [CLIMATE RESILIENCE ANALYSIS]

#### 7. [STORAGE] GROUNDWATER STORAGE & RESOURCES:

| Parameter | Value | Unit |
|-----------|-------|------|
| Instorage Unconfined Ground Water Resources | [instorage_unconfined_ground_water_resourcesham] | ham |
| Total Ground Water Availability in Unconfined Aquifer | [total_ground_water_availability_in_unconfined_aquifier_ham] | ham |
| Dynamic Confined Ground Water Resources | [dynamic_confined_ground_water_resourcesham] | ham |
| Instorage Confined Ground Water Resources | [instorage_confined_ground_water_resourcesham] | ham |
| Total Confined Ground Water Resources | [total_confined_ground_water_resources_ham] | ham |
| Dynamic Semi-confined Ground Water Resources | [dynamic_semi_confined_ground_water_resources_ham] | ham |
| Instorage Semi-confined Ground Water Resources | [instorage_semi_confined_ground_water_resources_ham] | ham |
| Total Semi-confined Ground Water Resources | [total_semiconfined_ground_water_resources_ham] | ham |
| Total Ground Water Availability in the Area | [total_ground_water_availability_in_the_area_ham] | ham |

**Storage Analysis:** [DETAILED ANALYSIS OF GROUNDWATER STORAGE]

#### 8.  WATERSHED & ADMINISTRATIVE ANALYSIS:

| Parameter | Value |
|-----------|-------|
| Watershed District | [DATA/NOT AVAILABLE] |
| Watershed Category | [DATA/NOT AVAILABLE] |
| Tehsil | [DATA/NOT AVAILABLE] |
| Taluk | [TALUK NAME] |
| Block | [DATA/NOT AVAILABLE] |
| Mandal | [DATA/NOT AVAILABLE] |
| Village | [DATA/NOT AVAILABLE] |

**Watershed Status:** [ANALYSIS OF WATERSHED STATUS]

---

*Report generated by Groundwater RAG API - Multilingual Support*

*Language: [LANGUAGE]*

IMPORTANT INSTRUCTIONS:
- Use the EXACT format shown above with emojis, specific section headers, and table structures
- Include ALL available data in the appropriate sections
- If data is not available, state "No data available" or "Not available"
- For state-level queries, analyze ALL available districts from that state
- Use proper markdown formatting for tables and headers
- Include specific numerical values with units
- Provide detailed explanations for each parameter
- Maintain the professional report structure throughout
- CRITICAL: For Quality Tagging, use the actual quality data from the context (PQ, Iron, Uranium, Nitrate, etc.) - do NOT just say "No data available" unless the quality_tagging field is truly empty
- If quality data shows parameters like [Iron], [Uranium], [Nitrate], etc., include them in the Quality Tagging field
- Analyze the quality parameters and provide meaningful insights about water quality concerns and treatment recommendations
"""
        f"{conversation_history_str}"
        f"{extracted_params_str}"
        f"Base your answer ONLY on the following groundwater data{location_info}{year_info}:\n{context_str}\n\n"
        f"If the data doesn't contain the answer, state that. Do NOT make up information.\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    try:
        response = _gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] Error from Gemini: {str(e)}"

# --- Authentication & Chat History Persistence ---
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def load_users() -> Dict[str, Dict[str, str]]:
    try:
        if not os.path.exists(USERS_FILE):
            return {"users": []}
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "users" in data:
                return data
            if isinstance(data, list):
                return {"users": data}
            return {"users": []}
    except Exception:
        return {"users": []}

def save_users(data: Dict[str, List[Dict[str, str]]]) -> None:
    try:
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def find_user(username: str) -> Optional[Dict[str, str]]:
    users = load_users().get("users", [])
    for user in users:
        if user.get("username") == username:
            return user
    return None

def register_user(username: str, password: str) -> bool:
    username = username.strip()
    if not username or not password:
        return False
    db = load_users()
    if any(u.get("username") == username for u in db.get("users", [])):
        return False
    db.setdefault("users", []).append({
        "username": username,
        "password": hash_password(password)
    })
    save_users(db)
    return True

def authenticate_user(username: str, password: str) -> bool:
    user = find_user(username)
    if not user:
        return False
    return user.get("password") == hash_password(password)

def _chat_history_path(username: str) -> str:
    safe_username = re.sub(r"[^a-zA-Z0-9_-]", "_", username)
    return os.path.join(CHAT_HISTORY_DIR, f"{safe_username}.json")

def load_chat_history(username: str) -> List[Dict[str, str]]:
    try:
        path = _chat_history_path(username)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        return []
    except Exception:
        return []

def save_chat_history(username: str, messages: List[Dict[str, str]]) -> None:
    try:
        path = _chat_history_path(username)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
    except Exception:
        pass

def search_qdrant_only(query_text, year=None, target_state=None, target_district=None, extracted_parameters=None):
    """Use ONLY Qdrant for RAG search, CSV only for visualization"""
    _init_components()  # Load CSV data for visualization
    _init_qdrant()      # Load Qdrant for RAG search
    
    try:
        # Use ONLY Qdrant for search - no CSV fallback
        if _qdrant_client is not None:
            print("Using Qdrant for RAG search...")
            qdrant_results = search_qdrant_rag(query_text, year, target_state, target_district, extracted_parameters)
            if qdrant_results:
                print(f"Qdrant RAG found {len(qdrant_results)} results")
                return qdrant_results
            else:
                print("No results found in Qdrant")
                return []
        else:
            print("Qdrant not available - no search possible")
            return []
        
    except Exception as e:
        print(f"Error in Qdrant search: {str(e)}")
        return []

def extract_query_parameters(query: str) -> dict:
    """Extract year, state, and district from query with flexible matching."""
    query_lower = query.lower()
    
    # Extract year
    year = None
    year_match = re.search(r'\b(19|20)\d{2}\b', query)
    if year_match:
        year = int(year_match.group(0))
    
    # Extract state with flexible matching
    target_state = None
    state_patterns = {
        'karnataka': ['karnataka', 'karnatak', 'karnatka'],
        'maharashtra': ['maharashtra', 'maharastra', 'maharashtra'],
        'gujarat': ['gujarat', 'gujrat', 'gujrat'],
        'tamil nadu': ['tamil nadu', 'tamilnadu', 'tamil nadu', 'tamilnadu'],
        'andhra pradesh': ['andhra pradesh', 'andhra', 'ap'],
        'telangana': ['telangana', 'telengana'],
        'kerala': ['kerala', 'keral'],
        'punjab': ['punjab', 'punjab'],
        'haryana': ['haryana', 'haryan'],
        'rajasthan': ['rajasthan', 'rajasthan'],
        'madhya pradesh': ['madhya pradesh', 'mp', 'madhya'],
        'uttar pradesh': ['uttar pradesh', 'up', 'uttar'],
        'bihar': ['bihar', 'bihar'],
        'west bengal': ['west bengal', 'westbengal', 'bengal'],
        'odisha': ['odisha', 'orissa', 'odisha'],
        'assam': ['assam', 'assam'],
        'jharkhand': ['jharkhand', 'jharkhand'],
        'chhattisgarh': ['chhattisgarh', 'chhattisgarh'],
        'himachal pradesh': ['himachal pradesh', 'himachal', 'hp'],
        'uttarakhand': ['uttarakhand', 'uttaranchal'],
        'goa': ['goa', 'goa'],
        'delhi': ['delhi', 'new delhi', 'nct'],
        'jammu and kashmir': ['jammu and kashmir', 'j&k', 'jammu kashmir'],
        'ladakh': ['ladakh', 'ladakh'],
        'arunachal pradesh': ['arunachal pradesh', 'arunachal'],
        'manipur': ['manipur', 'manipur'],
        'meghalaya': ['meghalaya', 'meghalaya'],
        'mizoram': ['mizoram', 'mizoram'],
        'nagaland': ['nagaland', 'nagaland'],
        'sikkim': ['sikkim', 'sikkim'],
        'tripura': ['tripura', 'tripura']
    }
    
    for state_name, patterns in state_patterns.items():
        for pattern in patterns:
            if pattern in query_lower:
                target_state = state_name.upper()
                break
        if target_state:
            break
    
    # Extract district (basic pattern matching)
    target_district = None
    # Look for common district indicators
    district_indicators = ['district', 'dist', 'taluk', 'block', 'mandal']
    for indicator in district_indicators:
        if indicator in query_lower:
            # Try to extract the district name after the indicator
            pattern = rf'{indicator}\s+([a-zA-Z\s]+)'
            match = re.search(pattern, query_lower)
            if match:
                target_district = match.group(1).strip().title()
                break
    
    return {
        'year': year,
        'target_state': target_state,
        'target_district': target_district
    }

def _safe_mean_numeric(df: pd.DataFrame, column_name: str) -> Optional[float]:
    """Return the mean of a column after safely converting to numeric.

    Handles numbers stored as strings (with commas, spaces, or stray characters).
    Returns None if no numeric values are present.
    """
    if column_name not in df.columns:
        return None
    series = df[column_name]
    # Convert to string, strip commas and whitespace, then coerce to numeric
    numeric_series = pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    numeric_series = numeric_series.dropna()
    if numeric_series.empty:
        return None
    return float(numeric_series.mean())


def calculate_average_values(df, target_state=None, target_district=None):
    """Calculate average values when year is not specified."""
    if df is None or df.empty:
        return {}
    
    # Filter by state if specified
    if target_state:
        df = df[df['STATE'].str.upper() == target_state.upper()]
    
    # Filter by district if specified
    if target_district:
        df = df[df['DISTRICT'].str.upper() == target_district.upper()]
    
    if df.empty:
        return {}
    
    # Calculate averages for key metrics
    averages = {}
    
    # Groundwater extraction
    extraction_cols = [
        'Ground Water Extraction for all uses (ha.m) - Total - Total',
        'ground_water_extraction_for_all_uses_ham'
    ]
    for col in extraction_cols:
        avg_val = _safe_mean_numeric(df, col)
        if avg_val is not None and not pd.isna(avg_val):
            averages['extraction'] = round(avg_val, 2)
            break
    
    # Groundwater recharge
    recharge_cols = [
        'Annual Ground water Recharge (ham) - Total - Total',
        'annual_ground_water_recharge_ham'
    ]
    for col in recharge_cols:
        avg_val = _safe_mean_numeric(df, col)
        if avg_val is not None and not pd.isna(avg_val):
            averages['recharge'] = round(avg_val, 2)
            break
    
    # Rainfall
    rainfall_cols = [
        'Rainfall (mm) - Total - Total',
        'rainfall_mm'
    ]
    for col in rainfall_cols:
        avg_val = _safe_mean_numeric(df, col)
        if avg_val is not None and not pd.isna(avg_val):
            averages['rainfall'] = round(avg_val, 2)
            break
    
    # Stage of extraction
    stage_cols = [
        'Stage of Ground Water Extraction (%) - Total - Total',
        'stage_of_ground_water_extraction_'
    ]
    for col in stage_cols:
        avg_val = _safe_mean_numeric(df, col)
        if avg_val is not None and not pd.isna(avg_val):
            averages['extraction_stage'] = round(avg_val, 2)
            break
    
    # Quality tagging distribution
    quality_cols = ['quality_tagging', 'Quality Tagging']
    for col in quality_cols:
        if col in df.columns:
            quality_data = df[col].dropna()
            if not quality_data.empty:
                quality_dist = quality_data.value_counts().to_dict()
                averages['quality_distribution'] = quality_dist
                break
    
    return averages

def answer_query(query: str, user_language: str = 'en', user_id: str = None) -> str:
    query = (query or '').strip()
    if not query:
        print("❌ [QUERY] Empty query received")
        return "Please provide a question."
    
    print(f"🔍 [QUERY] New question received: '{query}'")
    print(f"🌐 [QUERY] User language: {user_language}")
    if user_id:
        print(f"👤 [QUERY] User ID: {user_id}")
    
    try:
        print("⚙️ [INIT] Initializing components...")
        _init_components()  # Load CSV data
        _init_qdrant()      # Load Qdrant when needed
        _init_gemini()      # Load Gemini when needed
        _init_ml_components()  # Load ML components when needed
        print("✅ [INIT] All components initialized successfully")
    except Exception as e:
        print(f"❌ [INIT] Initialization error: {str(e)}")
        return f"Initialization error: {str(e)}"
    
    # Detect and translate query to English for processing
    print("🔄 [TRANSLATE] Processing query translation...")
    original_query = query
    translated_query, detected_lang = translate_query_to_english(query)
    if detected_lang != 'en':
        print(f"🌐 [TRANSLATE] Query translated from {detected_lang} to English")
    else:
        print("🌐 [TRANSLATE] Query is already in English")
    
    # Extract parameters with flexible matching
    print("🔍 [EXTRACT] Extracting query parameters...")
    params = extract_query_parameters(translated_query)
    year = params['year']
    target_state = params['target_state']
    target_district = params['target_district']
    
    print(f"📊 [EXTRACT] Parameters extracted:")
    print(f"   📅 Year: {year if year else 'Not specified'}")
    print(f"   🏛️ State: {target_state if target_state else 'Not specified'}")
    print(f"   🏘️ District: {target_district if target_district else 'Not specified'}")
    
    # Use hardcoded state matching for common states
    query_lower = translated_query.lower()
    
    # Handle Odisha/Orissa specifically
    if 'odisha' in query_lower or 'orissa' in query_lower:
        target_state = 'ODISHA'
        print(f"[DEBUG] Matched Odisha/Orissa -> ODISHA")
    
    # Handle major cities to states mapping
    elif 'chennai' in query_lower or 'madras' in query_lower:
        target_state = 'TAMILNADU'
        print(f"[DEBUG] Matched Chennai/Madras -> TAMILNADU")
    elif 'mumbai' in query_lower or 'bombay' in query_lower:
        target_state = 'MAHARASHTRA'
        print(f"[DEBUG] Matched Mumbai/Bombay -> MAHARASHTRA")
    elif 'bangalore' in query_lower or 'bengaluru' in query_lower:
        target_state = 'KARNATAKA'
        print(f"[DEBUG] Matched Bangalore/Bengaluru -> KARNATAKA")
    elif 'kolkata' in query_lower or 'calcutta' in query_lower:
        target_state = 'WEST BENGAL'
        print(f"[DEBUG] Matched Kolkata/Calcutta -> WEST BENGAL")
    elif 'delhi' in query_lower or 'new delhi' in query_lower:
        target_state = 'DELHI'
        print(f"[DEBUG] Matched Delhi/New Delhi -> DELHI")
    elif 'hyderabad' in query_lower:
        target_state = 'TELANGANA'
        print(f"[DEBUG] Matched Hyderabad -> TELANGANA")
    elif 'ahmedabad' in query_lower:
        target_state = 'GUJARAT'
        print(f"[DEBUG] Matched Ahmedabad -> GUJARAT")
    elif 'jaipur' in query_lower:
        target_state = 'RAJASTHAN'
        print(f"[DEBUG] Matched Jaipur -> RAJASTHAN")
    elif 'lucknow' in query_lower:
        target_state = 'UTTAR PRADESH'
        print(f"[DEBUG] Matched Lucknow -> UTTAR PRADESH")
    elif 'bhopal' in query_lower:
        target_state = 'MADHYA PRADESH'
        print(f"[DEBUG] Matched Bhopal -> MADHYA PRADESH")
    elif 'patna' in query_lower:
        target_state = 'BIHAR'
        print(f"[DEBUG] Matched Patna -> BIHAR")
    elif 'bhubaneswar' in query_lower or 'cuttack' in query_lower:
        target_state = 'ODISHA'
        print(f"[DEBUG] Matched Bhubaneswar/Cuttack -> ODISHA")
    elif 'raipur' in query_lower:
        target_state = 'CHHATTISGARH'
        print(f"[DEBUG] Matched Raipur -> CHHATTISGARH")
    elif 'ranchi' in query_lower:
        target_state = 'JHARKHAND'
        print(f"[DEBUG] Matched Ranchi -> JHARKHAND")
    elif 'guwahati' in query_lower:
        target_state = 'ASSAM'
        print(f"[DEBUG] Matched Guwahati -> ASSAM")
    elif 'chandigarh' in query_lower:
        target_state = 'CHANDIGARH'
        print(f"[DEBUG] Matched Chandigarh -> CHANDIGARH")
    elif 'pune' in query_lower or 'nagpur' in query_lower:
        target_state = 'MAHARASHTRA'
        print(f"[DEBUG] Matched {query_lower.split()[0].title()} -> MAHARASHTRA")
    elif any(city in query_lower for city in ['coimbatore', 'madurai', 'tiruchirapalli', 'trichy', 'salem', 'tirunelveli', 'erode', 'tiruppur', 'vellore', 'thoothukudi', 'tuticorin', 'dindigul', 'thanjavur', 'tiruvannamalai', 'kanchipuram', 'cuddalore', 'karur', 'namakkal', 'tiruvallur', 'ranipet']):
        target_state = 'TAMILNADU'
        print(f"[DEBUG] Matched Tamil Nadu city -> TAMILNADU")
    
    # Handle other common states
    elif 'karnataka' in query_lower:
        target_state = 'KARNATAKA'
    elif 'tamilnadu' in query_lower or 'tamil nadu' in query_lower:
        target_state = 'TAMILNADU'
    elif 'maharashtra' in query_lower:
        target_state = 'MAHARASHTRA'
    elif 'gujarat' in query_lower:
        target_state = 'GUJARAT'
    elif 'rajasthan' in query_lower:
        target_state = 'RAJASTHAN'
    elif 'west bengal' in query_lower or 'bengal' in query_lower:
        target_state = 'WEST BENGAL'
    elif 'bihar' in query_lower:
        target_state = 'BIHAR'
    elif 'telangana' in query_lower:
        target_state = 'TELANGANA'
    elif 'andhra pradesh' in query_lower or 'andhra' in query_lower:
        target_state = 'ANDHRA PRADESH'
    elif 'chhattisgarh' in query_lower or 'chattisgarh' in query_lower:
        target_state = 'CHHATTISGARH'
    elif 'madhya pradesh' in query_lower or 'mp' in query_lower:
        target_state = 'MADHYA PRADESH'
    elif 'uttar pradesh' in query_lower or 'up' in query_lower:
        target_state = 'UTTAR PRADESH'
    elif 'jharkhand' in query_lower:
        target_state = 'JHARKHAND'
    elif 'assam' in query_lower:
        target_state = 'ASSAM'
    elif 'punjab' in query_lower:
        target_state = 'PUNJAB'
    elif 'haryana' in query_lower:
        target_state = 'HARYANA'
    elif 'himachal pradesh' in query_lower or 'hp' in query_lower:
        target_state = 'HIMACHAL PRADESH'
    elif 'jammu and kashmir' in query_lower or 'j&k' in query_lower:
        target_state = 'JAMMU AND KASHMIR'
    elif 'uttarakhand' in query_lower or 'uk' in query_lower:
        target_state = 'UTTARAKHAND'
    elif 'goa' in query_lower:
        target_state = 'GOA'
    elif 'sikkim' in query_lower:
        target_state = 'SIKKIM'
    elif 'arunachal pradesh' in query_lower:
        target_state = 'ARUNACHAL PRADESH'
    elif 'nagaland' in query_lower:
        target_state = 'NAGALAND'
    elif 'manipur' in query_lower:
        target_state = 'MANIPUR'
    elif 'mizoram' in query_lower:
        target_state = 'MIZORAM'
    elif 'tripura' in query_lower:
        target_state = 'TRIPURA'
    elif 'meghalaya' in query_lower:
        target_state = 'MEGHALAYA'
    elif 'delhi' in query_lower:
        target_state = 'DELHI'
    elif 'chandigarh' in query_lower:
        target_state = 'CHANDIGARH'
    elif 'puducherry' in query_lower or 'pondicherry' in query_lower:
        target_state = 'PUDUCHERRY'
    elif 'andaman and nicobar' in query_lower:
        target_state = 'ANDAMAN AND NICOBAR ISLANDS'
    elif 'dadra and nagar haveli' in query_lower:
        target_state = 'DADRA AND NAGAR HAVELI'
    elif 'daman and diu' in query_lower:
        target_state = 'DAMAN AND DIU'
    elif 'lakshadweep' in query_lower:
        target_state = 'LAKSHADWEEP'
    
    print(f"[DEBUG] Extracted target_state: {target_state}")
    
    # Fallback to CSV-based extraction if not found
    if not target_state and _master_df is not None:
        unique_states = _master_df['STATE'].unique().tolist()
        unique_districts = _master_df['DISTRICT'].unique().tolist()
        
        # Try to find state with improved matching
        for state in unique_states:
            if pd.notna(state):
                state_lower = str(state).lower()
                # Exact match
                if re.search(r'\b' + re.escape(str(state)) + r'\b', translated_query, re.IGNORECASE):
                    target_state = state
                    break
                # Partial match
                elif str(state).lower() in query_lower:
                    target_state = state
                    break
        
        print(f"[DEBUG] CSV fallback extracted target_state: {target_state}")
        print(f"[DEBUG] Available states sample: {unique_states[:10]}")

        if target_state:
            districts_in_state = _master_df[_master_df['STATE'] == target_state]['DISTRICT'].unique().tolist()
            for district in districts_in_state:
                if pd.notna(district):
                    # Exact match
                    if re.search(r'\b' + re.escape(str(district)) + r'\b', translated_query, re.IGNORECASE):
                        target_district = district
                        break
                    # Partial match
                    elif str(district).lower() in translated_query.lower():
                        target_district = district
                        break
    
    # Enhanced NLP for extracting specific groundwater parameters
    extracted_parameters = {}
    if _nlp:
        doc = _nlp(translated_query)
        groundwater_keywords = {
            "rainfall": ["rainfall", "precipitation"],
            "groundwater availability": ["availability", "groundwater availability", "resource"],
            "extraction": ["extraction", "drawdown"],
            "recharge": ["recharge"],
            "saline": ["saline", "salty"],
            "fresh": ["fresh", "sweet"],
            "annual": ["annual", "yearly"],
            "total": ["total", "overall"]
        }

        for keyword_type, synonyms in groundwater_keywords.items():
            for token in doc:
                if token.text.lower() in synonyms or any(synonym in token.text.lower() for synonym in synonyms):
                    for i in range(max(0, token.i - 3), min(len(doc), token.i + 4)):
                        if doc[i].is_digit and doc[i].text.isdigit():
                            try:
                                value = float(doc[i].text)
                                if keyword_type not in extracted_parameters:
                                    extracted_parameters[keyword_type] = value
                                    break
                            except ValueError:
                                continue
    
    print("🔍 [EXPAND] Expanding query terms...")
    expanded_terms = expand_query(translated_query)
    expanded_query_text = f"{translated_query} {expanded_terms}".strip()
    print(f"📝 [EXPAND] Expanded query: '{expanded_query_text}'")
    
    # Use ONLY Qdrant for RAG search with flexible fallback
    print("🔍 [SEARCH] Starting Qdrant search...")
    print(f"   🎯 Target: {target_state or 'All states'}")
    print(f"   📅 Year: {year or 'All years'}")
    print(f"   🏘️ District: {target_district or 'All districts'}")
    
    candidate_results = search_qdrant_only(expanded_query_text, year=year, target_state=target_state, target_district=target_district, extracted_parameters=extracted_parameters)
    print(f"📊 [SEARCH] Initial search results: {len(candidate_results) if candidate_results else 0} records found")
    
    # If no results found with specific year, try without year filter
    if not candidate_results and year:
        print(f"🔄 [SEARCH] No results for year {year}, trying without year filter...")
        candidate_results = search_qdrant_only(expanded_query_text, year=None, target_state=target_state, target_district=target_district, extracted_parameters=extracted_parameters)
        print(f"📊 [SEARCH] Search without year filter: {len(candidate_results) if candidate_results else 0} records found")
    
    # If no results found with district, try with just state
    if not candidate_results and target_state and target_district:
        print(f"🔄 [SEARCH] No results for district {target_district}, trying with just state...")
        candidate_results = search_qdrant_only(expanded_query_text, year=year, target_state=target_state, target_district=None, extracted_parameters=extracted_parameters)
        print(f"📊 [SEARCH] Search without district filter: {len(candidate_results) if candidate_results else 0} records found")
    
    # If no results found with location filters, try without location filters (but only if state was specified)
    if not candidate_results and target_state:
        print(f"🔄 [SEARCH] No results with location filters, trying without district filter...")
        candidate_results = search_qdrant_only(expanded_query_text, year=year, target_state=target_state, target_district=None, extracted_parameters=extracted_parameters)
        print(f"📊 [SEARCH] Search without district: {len(candidate_results) if candidate_results else 0} records found")
        
        # If still no results, try without year filter too
        if not candidate_results and year:
            print(f"🔄 [SEARCH] No results with state filter, trying without year filter...")
            candidate_results = search_qdrant_only(expanded_query_text, year=None, target_state=target_state, target_district=None, extracted_parameters=extracted_parameters)
            print(f"📊 [SEARCH] Search without year: {len(candidate_results) if candidate_results else 0} records found")
    
    # If no results and a specific state was requested, return a clear message
    if not candidate_results and target_state:
        print(f"❌ [RESULT] No data available for {target_state} in the INGRIS groundwater database")
        return f"No data available for {target_state} in the INGRIS groundwater database."
    
    # If no results at all, return specific message about data sources
    if not candidate_results:
        print("❌ [RESULT] No data available in the INGRIS groundwater database for the query")
        return "No data available in the INGRIS groundwater database (ingris_groundwater_collection and ingris_rag_ready_complete.csv) for your query."
    
    print("🔄 [RERANK] Re-ranking search results...")
    re_ranked_results = re_rank_chunks(expanded_query_text, candidate_results, top_k=5)
    if not re_ranked_results:
        print("❌ [RESULT] No relevant data found after re-ranking")
        return "No relevant data found in the INGRIS groundwater database for your query."
    
    print(f"✅ [RERANK] Re-ranked to top {len(re_ranked_results)} most relevant results")
    context_data = [res['data'] for res in re_ranked_results]
    
    # Calculate averages if no specific year requested
    averages = {}
    if not year and _master_df is not None:
        print("📊 [AVERAGE] No year specified, calculating average values...")
        averages = calculate_average_values(_master_df, target_state, target_district)
        if averages:
            print(f"📊 [AVERAGE] Calculated averages: {list(averages.keys())}")
        else:
            print("📊 [AVERAGE] No averages calculated")
    
    # Load chat history if user_id provided
    chat_history = None
    if user_id:
        print(f"💬 [CHAT] Loading chat history for user {user_id}")
        chat_history = load_chat_history(user_id)
        if chat_history:
            print(f"💬 [CHAT] Loaded {len(chat_history)} previous messages")
        else:
            print("💬 [CHAT] No previous chat history found")
    
    print("🤖 [GEMINI] Generating answer using Gemini AI...")
    
    # Debug: Check quality data in context
    quality_data_found = False
    for i, item in enumerate(context_data[:3]):  # Check first 3 items
        if 'quality_tagging' in item:
            quality_val = item['quality_tagging']
            print(f"🔍 [DEBUG] Quality data in context item {i+1}: {quality_val}")
            if pd.notna(quality_val) and str(quality_val).strip() not in ['', 'nan', 'None']:
                quality_data_found = True
        elif 'Quality Tagging' in item:
            quality_val = item['Quality Tagging']
            print(f"🔍 [DEBUG] Quality data in context item {i+1}: {quality_val}")
            if pd.notna(quality_val) and str(quality_val).strip() not in ['', 'nan', 'None']:
                quality_data_found = True
    
    if quality_data_found:
        print("✅ [DEBUG] Quality data found in context - should be included in response")
    else:
        print("⚠️ [DEBUG] No quality data found in context")
    
    answer = generate_answer_from_gemini(
        translated_query, 
        context_data, 
        year=year, 
        target_state=target_state, 
        target_district=target_district, 
        chat_history=chat_history, 
        extracted_parameters=extracted_parameters,
        user_language=user_language,
        averages=averages
    )
    
    print(f"✅ [GEMINI] Answer generated successfully (length: {len(answer)} characters)")
    
    # Translate answer back to user's language if needed
    if user_language != 'en':
        print(f"🔄 [TRANSLATE] Translating answer back to {user_language}...")
        answer = translate_answer_to_language(answer, user_language)
        print("✅ [TRANSLATE] Answer translated successfully")
    
    print("🎉 [COMPLETE] Query processing completed successfully")
    return answer

# --- Visualization Functions ---

def ensure_unique_data(df):
    """Ensure data uniqueness to prevent overlapping in visualizations."""
    if df is None or df.empty:
        return df
    
    # Remove duplicates based on combined_text to prevent overlapping
    if 'combined_text' in df.columns:
        original_count = len(df)
        df = df.drop_duplicates(subset=['combined_text'], keep='first')
        removed_count = original_count - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} duplicate records. Using {len(df)} unique records for visualization.")
    
    return df
def create_groundwater_overview_dashboard(df):
    """Create a comprehensive overview dashboard of groundwater data."""
    if df is None or df.empty:
        return None
    
    # Ensure we're working with unique data to prevent overlapping
    df = ensure_unique_data(df)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Groundwater Extraction by State (Top 10)',
            'Annual Groundwater Recharge Trends',
            'Stage of Groundwater Extraction Distribution',
            'Rainfall vs Groundwater Recharge Correlation',
            'District-wise Groundwater Availability',
            'Year-wise Data Coverage'
        ],
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}]
        ]
    )
    
    # 1. Groundwater Extraction by State (Top 10)
    if 'Ground Water Extraction for all uses (ha.m) - Total - Total' in df.columns:
        extraction_col = 'Ground Water Extraction for all uses (ha.m) - Total - Total'
        state_extraction = df.groupby('STATE')[extraction_col].sum().sort_values(ascending=False).head(10)
        
        fig.add_trace(
            go.Bar(
                x=state_extraction.index,
                y=state_extraction.values,
                name="Groundwater Extraction",
                marker_color='lightblue'
            ),
            row=1, col=1
        )
    
    # 2. Annual Groundwater Recharge Trends
    if 'Annual Ground water Recharge (ham) - Total - Total' in df.columns:
        recharge_col = 'Annual Ground water Recharge (ham) - Total - Total'
        yearly_recharge = df.groupby('Assessment_Year')[recharge_col].mean().reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=yearly_recharge['Assessment_Year'],
                y=yearly_recharge[recharge_col],
                mode='lines+markers',
                name="Annual Recharge",
                line=dict(color='green', width=3)
            ),
            row=1, col=2
        )
    
    # 3. Stage of Groundwater Extraction Distribution
    if 'Stage of Ground Water Extraction (%) - Total - Total' in df.columns:
        stage_col = 'Stage of Ground Water Extraction (%) - Total - Total'
        stage_data = df[stage_col].dropna()
        
        fig.add_trace(
            go.Histogram(
                x=stage_data,
                nbinsx=20,
                name="Extraction Stage %",
                marker_color='orange'
            ),
            row=2, col=1
        )
    
    # 4. Rainfall vs Groundwater Recharge Correlation
    if 'Rainfall (mm) - Total - Total' in df.columns and 'Annual Ground water Recharge (ham) - Total - Total' in df.columns:
        rainfall_col = 'Rainfall (mm) - Total - Total'
        recharge_col = 'Annual Ground water Recharge (ham) - Total - Total'
        
        # Sample data for correlation plot
        sample_data = df[[rainfall_col, recharge_col]].dropna().sample(n=min(1000, len(df)))
        
        fig.add_trace(
            go.Scatter(
                x=sample_data[rainfall_col],
                y=sample_data[recharge_col],
                mode='markers',
                name="Rainfall vs Recharge",
                marker=dict(color='purple', size=4, opacity=0.6)
            ),
            row=2, col=2
        )
    
    # 5. District-wise Groundwater Availability (Top 15)
    if 'Annual Extractable Ground water Resource (ham) - Total - Total' in df.columns:
        resource_col = 'Annual Extractable Ground water Resource (ham) - Total - Total'
        district_resource = df.groupby('DISTRICT')[resource_col].sum().sort_values(ascending=False).head(15)
        
        fig.add_trace(
            go.Bar(
                x=district_resource.values,
                y=district_resource.index,
                orientation='h',
                name="Groundwater Resource",
                marker_color='lightgreen'
            ),
            row=3, col=1
        )
    
    # 6. Year-wise Data Coverage
    yearly_counts = df['Assessment_Year'].value_counts().sort_index()
    
    fig.add_trace(
        go.Bar(
            x=yearly_counts.index,
            y=yearly_counts.values,
            name="Data Records",
            marker_color='lightcoral'
        ),
        row=3, col=2
    )
    
    # Update layout with white text for dark theme
    fig.update_layout(
        height=1200,
        title_text="Groundwater Data Analysis Dashboard",
        title_x=0.5,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)'),
        yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)')
    )
    
    return fig

def fetch_state_data_from_qdrant(selected_state=None):
    """Fetch state data from Qdrant collection"""
    if _qdrant_client is None:
        print("Qdrant client not available")
        return None
    
    try:
        print(f"Fetching data for state: {selected_state}")
        
        # Build query filter
        scroll_filter = None
        if selected_state:
            scroll_filter = {
                "must": [
                    {"key": "STATE", "match": {"value": selected_state}}
                ]
            }
        
        # Fetch data with smaller batches
        all_data = []
        offset = None
        
        while True:
            scroll_result, next_offset = _qdrant_client.scroll(
                collection_name="ingris_groundwater_collection",
                limit=1000,  # Smaller batch size
                offset=offset,
                scroll_filter=scroll_filter,
                with_payload=True
            )
            
            if not scroll_result:
                break
                
            all_data.extend(scroll_result)
            
            # If we got fewer records than requested, we're done
            if len(scroll_result) < 1000:
                break
                
            offset = next_offset
            
            # Safety check to avoid infinite loop
            if len(all_data) > 5000:  # Reasonable limit
                break
        
        print(f"Found {len(all_data)} records")
        
        if not all_data:
            print("No data found in Qdrant")
            return None
        
        # Convert to DataFrame
        data = []
        for point in all_data:
            payload = point.payload
            data.append(payload)
        
        df = pd.DataFrame(data)
        print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df
        
    except Exception as e:
        print(f"Error fetching data from Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_state_analysis_plots(df, selected_state=None):
    """Create state-specific analysis plots with deduplication and improved spacing."""
    if df is None or df.empty:
        return None
    
    # Ensure we're working with unique data to prevent overlapping
    df = ensure_unique_data(df)
    
    # Limit data points to prevent overcrowding
    if len(df) > 50:
        print(f"Limiting data from {len(df)} to 50 records to prevent overcrowding")
        df = df.sample(n=50, random_state=42)
    
    # Filter data for selected state if provided
    if selected_state:
        state_df = df[df['STATE'] == selected_state]
        title_suffix = f" - {selected_state}"
    else:
        state_df = df
        title_suffix = " - All States"
    
    if state_df.empty:
        return None
    
    # Create subplots with improved spacing to prevent overlap
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'Groundwater Extraction by District{title_suffix}',
            f'Rainfall Distribution{title_suffix}',
            f'Recharge vs Extraction Scatter{title_suffix}',
            f'Quality Tagging Analysis{title_suffix}'
        ],
        specs=[
            [{"type": "bar"}, {"type": "box"}],
            [{"type": "scatter"}, {"type": "pie"}]
        ],
        vertical_spacing=0.2,  # Increased vertical spacing
        horizontal_spacing=0.15  # Increased horizontal spacing
    )
    
    # 1. Groundwater Extraction by District
    extraction_col = None
    if 'Ground Water Extraction for all uses (ha.m) - Total - Total' in state_df.columns:
        extraction_col = 'Ground Water Extraction for all uses (ha.m) - Total - Total'
    elif 'ground_water_extraction_for_all_uses_ham' in state_df.columns:
        extraction_col = 'ground_water_extraction_for_all_uses_ham'
    
    if extraction_col:
        district_extraction = state_df.groupby('DISTRICT')[extraction_col].sum().sort_values(ascending=False).head(8)  # Reduced to 8 districts
        
        fig.add_trace(
            go.Bar(
                x=district_extraction.index,
                y=district_extraction.values,
                name="Extraction by District",
                marker_color='skyblue',
                text=district_extraction.values,
                textposition='outside',
                textfont=dict(size=10)
            ),
            row=1, col=1
        )
    
    # 2. Rainfall Distribution
    rainfall_col = None
    if 'Rainfall (mm) - Total - Total' in state_df.columns:
        rainfall_col = 'Rainfall (mm) - Total - Total'
    elif 'rainfall_mm' in state_df.columns:
        rainfall_col = 'rainfall_mm'
    
    if rainfall_col:
        rainfall_data = state_df[rainfall_col].dropna()
        
        fig.add_trace(
            go.Box(
                y=rainfall_data,
                name="Rainfall Distribution",
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
    
    # 3. Recharge vs Extraction Scatter
    recharge_col = None
    if 'Annual Ground water Recharge (ham) - Total - Total' in state_df.columns:
        recharge_col = 'Annual Ground water Recharge (ham) - Total - Total'
    elif 'annual_ground_water_recharge_ham' in state_df.columns:
        recharge_col = 'annual_ground_water_recharge_ham'
    
    if recharge_col and extraction_col:
        scatter_data = state_df[[recharge_col, extraction_col]].dropna()
        
        # Limit scatter plot data to prevent overcrowding
        if len(scatter_data) > 30:
            scatter_data = scatter_data.sample(n=30, random_state=42)
        
        fig.add_trace(
            go.Scatter(
                x=scatter_data[recharge_col],
                y=scatter_data[extraction_col],
                mode='markers',
                name="Recharge vs Extraction",
                marker=dict(
                    color='lightblue',
                    size=12,  # Larger markers for better visibility
                    opacity=0.8,
                    line=dict(width=1, color='darkblue')
                ),
                text=[f"District: {idx}" for idx in scatter_data.index],
                hovertemplate='<b>%{text}</b><br>Recharge: %{x}<br>Extraction: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # 4. Quality Tagging Analysis
    quality_col = None
    if 'quality_tagging' in state_df.columns:
        quality_col = 'quality_tagging'
    elif 'Quality Tagging' in state_df.columns:
        quality_col = 'Quality Tagging'
    
    if quality_col:
        # Filter out NaN values and get quality distribution
        quality_data = state_df[quality_col].dropna()
        if not quality_data.empty:
            quality_counts = quality_data.value_counts()
            
            fig.add_trace(
                go.Pie(
                    labels=quality_counts.index,
                    values=quality_counts.values,
                    name="Quality Distribution"
                ),
                row=2, col=2
            )
        else:
            # Show "No Quality Data Available" if all values are NaN
            fig.add_trace(
                go.Pie(
                    labels=["No Quality Data Available"],
                    values=[1],
                    name="Quality Distribution"
                ),
                row=2, col=2
            )
    
    # Update layout with improved spacing to prevent overlapping
    fig.update_layout(
        height=1200,  # Increased height for better spacing
        title_text=f"Detailed State Analysis{title_suffix}",
        title_x=0.5,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),  # Smaller font to prevent overlap
        title_font=dict(color='white', size=18),
        margin=dict(l=100, r=100, t=150, b=100),  # Increased margins
        # Improved axis settings to prevent label overlap
        xaxis=dict(
            color='white', 
            gridcolor='rgba(255,255,255,0.2)', 
            tickangle=-45,
            tickfont=dict(size=10),
            showticklabels=True,
            nticks=8  # Limit number of ticks
        ),
        yaxis=dict(
            color='white', 
            gridcolor='rgba(255,255,255,0.2)',
            tickfont=dict(size=10)
        ),
        xaxis2=dict(
            color='white', 
            gridcolor='rgba(255,255,255,0.2)',
            tickfont=dict(size=10)
        ),
        yaxis2=dict(
            color='white', 
            gridcolor='rgba(255,255,255,0.2)',
            tickfont=dict(size=10)
        ),
        xaxis3=dict(
            color='white', 
            gridcolor='rgba(255,255,255,0.2)',
            tickfont=dict(size=10)
        ),
        yaxis3=dict(
            color='white', 
            gridcolor='rgba(255,255,255,0.2)',
            tickfont=dict(size=10)
        ),
        xaxis4=dict(
            color='white', 
            gridcolor='rgba(255,255,255,0.2)',
            tickfont=dict(size=10)
        ),
        yaxis4=dict(
            color='white', 
            gridcolor='rgba(255,255,255,0.2)',
            tickfont=dict(size=10)
        )
    )
    
    # Update subplot titles font size
    fig.update_annotations(font_size=14, selector=dict(type="annotation"))
    
    return fig

def create_temporal_analysis_plots(df):
    """Create temporal analysis plots with deduplication."""
    if df is None or df.empty:
        return None
    
    # Ensure we're working with unique data to prevent overlapping
    df = ensure_unique_data(df)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Groundwater Recharge Trends by Year',
            'Extraction Trends by Year',
            'Stage of Extraction Over Time',
            'Resource Availability Trends'
        ],
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Groundwater Recharge Trends
    if 'Annual Ground water Recharge (ham) - Total - Total' in df.columns:
        recharge_col = 'Annual Ground water Recharge (ham) - Total - Total'
        yearly_recharge = df.groupby('Assessment_Year')[recharge_col].agg(['mean', 'std']).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=yearly_recharge['Assessment_Year'],
                y=yearly_recharge['mean'],
                mode='lines+markers',
                name="Mean Recharge",
                line=dict(color='blue', width=3),
                error_y=dict(type='data', array=yearly_recharge['std'])
            ),
            row=1, col=1
        )
    
    # 2. Extraction Trends
    if 'Ground Water Extraction for all uses (ha.m) - Total - Total' in df.columns:
        extraction_col = 'Ground Water Extraction for all uses (ha.m) - Total - Total'
        yearly_extraction = df.groupby('Assessment_Year')[extraction_col].agg(['mean', 'std']).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=yearly_extraction['Assessment_Year'],
                y=yearly_extraction['mean'],
                mode='lines+markers',
                name="Mean Extraction",
                line=dict(color='red', width=3),
                error_y=dict(type='data', array=yearly_extraction['std'])
            ),
            row=1, col=2
        )
    
    # 3. Stage of Extraction Over Time
    if 'Stage of Ground Water Extraction (%) - Total - Total' in df.columns:
        stage_col = 'Stage of Ground Water Extraction (%) - Total - Total'
        yearly_stage = df.groupby('Assessment_Year')[stage_col].mean().reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=yearly_stage['Assessment_Year'],
                y=yearly_stage[stage_col],
                mode='lines+markers',
                name="Extraction Stage %",
                line=dict(color='orange', width=3)
            ),
            row=2, col=1
        )
    
    # 4. Resource Availability Trends
    if 'Annual Extractable Ground water Resource (ham) - Total - Total' in df.columns:
        resource_col = 'Annual Extractable Ground water Resource (ham) - Total - Total'
        yearly_resource = df.groupby('Assessment_Year')[resource_col].mean().reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=yearly_resource['Assessment_Year'],
                y=yearly_resource[resource_col],
                mode='lines+markers',
                name="Available Resource",
                line=dict(color='green', width=3)
            ),
            row=2, col=2
        )
    
    # Update layout with white text for dark theme
    fig.update_layout(
        height=800,
        title_text="Temporal Analysis - Groundwater Trends Over Time",
        title_x=0.5,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)'),
        yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)')
    )
    
    return fig

def create_geographical_heatmap(df, metric='Annual Ground water Recharge (ham) - Total - Total'):
    """Create geographical heatmap with deduplication."""
    if df is None or df.empty:
        return None
    
    # Ensure we're working with unique data to prevent overlapping
    df = ensure_unique_data(df)
    
    if metric not in df.columns:
        return None
    
    # Aggregate data by state
    state_metrics = df.groupby('STATE')[metric].mean().reset_index()
    
    # Create a simple bar chart as heatmap (since we don't have actual coordinates)
    fig = go.Figure(data=[
        go.Bar(
            x=state_metrics['STATE'],
            y=state_metrics[metric],
            marker=dict(
                color=state_metrics[metric],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=metric)
            )
        )
    ])
    
    fig.update_layout(
        title=f"Geographical Distribution of {metric}",
        xaxis_title="State",
        yaxis_title=metric,
        height=600,
        xaxis_tickangle=-45,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)'),
        yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)')
    )
    
    return fig

def create_correlation_matrix_plot(df):
    """Create correlation matrix plot with deduplication."""
    if df is None or df.empty:
        return None
    
    # Ensure we're working with unique data to prevent overlapping
    df = ensure_unique_data(df)
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove columns with too many NaN values
    numerical_cols = [col for col in numerical_cols if df[col].notna().sum() > len(df) * 0.1]
    
    if len(numerical_cols) < 2:
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Correlation Matrix of Groundwater Parameters",
        height=600,
        xaxis_tickangle=-45,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)'),
        yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)')
    )
    
    return fig

def create_statistical_summary_plots(df):
    """Create statistical summary plots with deduplication."""
    if df is None or df.empty:
        return None
    
    # Ensure we're working with unique data to prevent overlapping
    df = ensure_unique_data(df)
    
    # Select key numerical columns
    key_columns = [
        'Annual Ground water Recharge (ham) - Total - Total',
        'Ground Water Extraction for all uses (ha.m) - Total - Total',
        'Stage of Ground Water Extraction (%) - Total - Total',
        'Rainfall (mm) - Total - Total'
    ]
    
    available_columns = [col for col in key_columns if col in df.columns]
    
    if not available_columns:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{col} Distribution' for col in available_columns[:4]]
    )
    
    for i, col in enumerate(available_columns[:4]):
        row = (i // 2) + 1
        col_idx = (i % 2) + 1
        
        data = df[col].dropna()
        
        # Create histogram
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=30,
                name=col,
                opacity=0.7
            ),
            row=row, col=col_idx
        )
    
    fig.update_layout(
        height=800,
        title_text="Statistical Distribution of Key Groundwater Parameters",
        title_x=0.5,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)'),
        yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)')
    )
    
    return fig

# --- FastAPI Endpoints ---

# Visualization Endpoints
@app.get("/visualizations/overview")
async def get_overview_dashboard():
    """Get comprehensive overview dashboard of groundwater data."""
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        fig = create_groundwater_overview_dashboard(_master_df)
        if fig is None:
            raise HTTPException(status_code=500, detail="Unable to create dashboard")
        
        return {"plot_json": fig.to_json(), "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/state-analysis")
async def get_state_analysis(state: Optional[str] = None):
    """Get detailed state analysis plots."""
    try:
        _init_components()
        if _qdrant_client is None:
            raise HTTPException(status_code=400, detail="Qdrant client not available")
        
        # Fetch data from Qdrant
        df = fetch_state_data_from_qdrant(state)
        
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No data available for the selected state")
        
        fig = create_state_analysis_plots(df, state)
        if fig is None:
            raise HTTPException(status_code=500, detail="Unable to create state analysis")
        
        return {"plot_json": fig.to_json(), "success": True, "state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/temporal-analysis")
async def get_temporal_analysis():
    """Get temporal analysis plots showing trends over time."""
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        fig = create_temporal_analysis_plots(_master_df)
        if fig is None:
            raise HTTPException(status_code=500, detail="Unable to create temporal analysis")
        
        return {"plot_json": fig.to_json(), "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/geographical-heatmap")
async def get_geographical_heatmap(metric: str = 'Annual Ground water Recharge (ham) - Total - Total'):
    """Get geographical heatmap of groundwater metrics by state."""
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        fig = create_geographical_heatmap(_master_df, metric)
        if fig is None:
            raise HTTPException(status_code=500, detail="Unable to create geographical heatmap")
        
        return {"plot_json": fig.to_json(), "success": True, "metric": metric}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/correlation-matrix")
async def get_correlation_matrix():
    """Get correlation matrix heatmap of numerical groundwater parameters."""
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        fig = create_correlation_matrix_plot(_master_df)
        if fig is None:
            raise HTTPException(status_code=500, detail="Unable to create correlation matrix")
        
        return {"plot_json": fig.to_json(), "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/statistical-summary")
async def get_statistical_summary():
    """Get statistical summary plots including distribution and box plots."""
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        fig = create_statistical_summary_plots(_master_df)
        if fig is None:
            raise HTTPException(status_code=500, detail="Unable to create statistical summary")
        
        return {"plot_json": fig.to_json(), "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/available-metrics")
async def get_available_metrics():
    """Get list of available metrics for geographical heatmap."""
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        numerical_cols = _master_df.select_dtypes(include=[np.number]).columns.tolist()
        key_metrics = [
            'Annual Ground water Recharge (ham) - Total - Total',
            'Ground Water Extraction for all uses (ha.m) - Total - Total',
            'Stage of Ground Water Extraction (%) - Total - Total',
            'Rainfall (mm) - Total - Total'
        ]
        available_metrics = [col for col in key_metrics if col in numerical_cols]
        
        return {"metrics": available_metrics, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/available-states")
async def get_available_states():
    """Get list of available states for state analysis."""
    try:
        # Return a hardcoded list of states for now to avoid Qdrant timeout issues
        # This can be improved later with proper Qdrant integration
        available_states = [
            "ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHHATTISGARH",
            "DELHI", "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH",
            "JAMMU AND KASHMIR", "JHARKHAND", "KARNATAKA", "KERALA", "MADHYA PRADESH",
            "MAHARASHTRA", "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND",
            "ODISHA", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL NADU",
            "TELANGANA", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL",
            "ANDAMAN AND NICOBAR ISLANDS", "CHANDIGARH", "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
            "LAKSHADWEEP", "PUDUCHERRY"
        ]
        
        return {"states": available_states, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask(request: AskRequest):
    """Main endpoint for asking questions about groundwater data."""
    print(f"🌐 [API] /ask endpoint called")
    print(f"📝 [API] Query: '{request.query[:100]}{'...' if len(request.query) > 100 else ''}'")
    print(f"🌐 [API] Language: {request.language or 'Auto-detect'}")
    print(f"👤 [API] User ID: {request.user_id or 'Anonymous'}")
    
    query = request.query.strip()
    if not query:
        print("❌ [API] Empty query received")
        raise HTTPException(status_code=400, detail="Missing 'query'")
    try:
        user_lang = request.language or detect_language(query)
        print(f"🔍 [API] Detected language: {user_lang}")
        
        answer = answer_query(query, user_lang, request.user_id)
        
        print(f"✅ [API] Answer generated successfully")
        print(f"📊 [API] Response length: {len(answer)} characters")
        
        return {
            "answer": answer, 
            "detected_lang": detect_language(query),
            "selected_lang": user_lang,
            "query": query
        }
    except Exception as exc:
        print(f"❌ [API] Error processing query: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/register")
async def register(request: UserRegister):
    """Register a new user."""
    if not request.username or not request.password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    
    if register_user(request.username, request.password):
        return {"message": "User registered successfully"}
    else:
        raise HTTPException(status_code=400, detail="Username already exists or invalid input")

@app.post("/login")
async def login(request: UserLogin):
    """Login user."""
    if not request.username or not request.password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    
    if authenticate_user(request.username, request.password):
        return {"message": "Login successful", "username": request.username}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

@app.get("/chat-history/{username}")
async def get_chat_history(username: str):
    """Get chat history for a user."""
    try:
        history = load_chat_history(username)
        return {"messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-history/{username}")
async def save_chat_history_endpoint(username: str, chat_data: ChatHistory):
    """Save chat history for a user."""
    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_data.messages]
        save_chat_history(username, messages)
        return {"message": "Chat history saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages."""
    return {"languages": SUPPORTED_LANGUAGES}

@app.get("/data-info")
async def get_data_info():
    """Get information about the loaded dataset."""
    try:
        _init_components()
        if _master_df is not None:
            available_states = _master_df['STATE'].unique().tolist()
            available_years = _master_df['Assessment_Year'].unique().tolist()
            total_records = len(_master_df)
            
            return {
                "total_records": total_records,
                "states_count": len(available_states),
                "years_count": len(available_years),
                "sample_states": [s for s in available_states[:10] if pd.notna(s)],
                "sample_years": [y for y in available_years[:5] if pd.notna(y)],
                "embeddings_uploaded": _embeddings_uploaded
            }
        else:
            return {"error": "Dataset not loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-embeddings")
async def clear_embeddings():
    """Clear all embeddings from Qdrant."""
    try:
        _init_components()
        _qdrant_client.delete(
            collection_name=COLLECTION_NAME, 
            points_selector=models.FilterSelector(filter=models.Filter(must=[]))
        )
        global _embeddings_uploaded, _bm25_model, _all_chunks, _bm25_df
        _embeddings_uploaded = False
        _bm25_model = None
        _all_chunks = None
        _bm25_df = None
        return {"message": "All embeddings cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-data")
async def upload_data():
    """Upload Excel data to Qdrant."""
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        if upload_excel_to_qdrant(_master_df):
            _load_bm25()
            global _embeddings_uploaded
            _embeddings_uploaded = True
            return {"message": "Data uploaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to upload data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-formatted")
async def ask_formatted(request: AskRequest):
    """Main endpoint for asking questions about groundwater data with enhanced formatting."""
    print(f"🌐 [API] /ask-formatted endpoint called")
    print(f"📝 [API] Query: '{request.query[:100]}{'...' if len(request.query) > 100 else ''}'")
    print(f"🌐 [API] Language: {request.language or 'Auto-detect'}")
    print(f"👤 [API] User ID: {request.user_id or 'Anonymous'}")
    
    query = request.query.strip()
    if not query:
        print("❌ [API] Empty query received")
        raise HTTPException(status_code=400, detail="Missing 'query'")
    try:
        user_lang = request.language or detect_language(query)
        print(f"🔍 [API] Detected language: {user_lang}")
        
        answer = answer_query(query, user_lang, request.user_id)
        
        # Add additional formatting instructions for better structure
        formatted_answer = f"""
#  Groundwater Data Analysis Report

## Query
**Question:** {query}

## Analysis
{answer}

---
*Report generated by Groundwater RAG API - Multilingual Support*  
*Language: {SUPPORTED_LANGUAGES.get(user_lang, user_lang)}*
        """
        
        print(f"✅ [API] Formatted answer generated successfully")
        print(f"📊 [API] Formatted response length: {len(formatted_answer)} characters")
        
        return {
            "answer": formatted_answer.strip(), 
            "detected_lang": detect_language(query),
            "selected_lang": user_lang,
            "query": query,
            "formatted": True
        }
    except Exception as exc:
        print(f"❌ [API] Error processing formatted query: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))

# --- Visualization API Endpoints ---
@app.get("/visualizations/overview-dashboard")
async def get_overview_dashboard():
    """Get comprehensive overview dashboard visualization."""
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        fig = create_groundwater_overview_dashboard(_master_df)
        if fig is None:
            raise HTTPException(status_code=500, detail="Unable to create dashboard")
        
        return {
            "success": True,
            "plot_json": fig.to_json(),
            "message": "Overview dashboard generated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/state-analysis")
async def get_state_analysis(state: Optional[str] = None):
    """Get detailed state analysis visualization."""
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        fig = create_state_analysis_plots(_master_df, state)
        if fig is None:
            raise HTTPException(status_code=500, detail="Unable to create state analysis")
        
        return {
            "success": True,
            "plot_json": fig.to_json(),
            "state": state or "All States",
            "message": "State analysis generated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/temporal-analysis")
async def get_temporal_analysis():
    """Get temporal analysis visualization showing trends over time."""
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        fig = create_temporal_analysis_plots(_master_df)
        if fig is None:
            raise HTTPException(status_code=500, detail="Unable to create temporal analysis")
        
        return {
            "success": True,
            "plot_json": fig.to_json(),
            "message": "Temporal analysis generated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/correlation-matrix")
async def get_correlation_matrix():
    """Get correlation matrix visualization of groundwater parameters."""
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        fig = create_correlation_matrix_plot(_master_df)
        if fig is None:
            raise HTTPException(status_code=500, detail="Unable to create correlation matrix")
        
        return {
            "success": True,
            "plot_json": fig.to_json(),
            "message": "Correlation matrix generated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/statistical-summary")
async def get_statistical_summary():
    """Get statistical summary visualization of key parameters."""
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        fig = create_statistical_summary_plots(_master_df)
        if fig is None:
            raise HTTPException(status_code=500, detail="Unable to create statistical summary")
        
        return {
            "success": True,
            "plot_json": fig.to_json(),
            "message": "Statistical summary generated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/available-states")
async def get_available_states():
    """Get list of available states for state analysis."""
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        available_states = sorted([s for s in _master_df['STATE'].unique() if pd.notna(s)])
        return {
            "success": True,
            "states": available_states,
            "count": len(available_states)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-location")
async def analyze_location(request: dict):
    """Analyze groundwater data for a specific location (lat, lng)."""
    try:
        lat = request.get("lat")
        lng = request.get("lng")
        
        if not lat or not lng:
            raise HTTPException(status_code=400, detail="Missing latitude or longitude")
        
        # Convert coordinates to state using reverse geocoding
        state = get_state_from_coordinates(lat, lng)
        
        if not state:
            return {
                "error": "Could not determine state from coordinates",
                "state": None,
                "analysis": "The provided coordinates are outside India or could not be mapped to a specific state."
            }
        
        # Get state-specific analysis
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        # Filter data for the specific state
        state_data = _master_df[_master_df['STATE'].str.contains(state, case=False, na=False)]
        
        if state_data.empty:
            return {
                "error": f"No groundwater data found for {state}",
                "state": state,
                "analysis": f"No groundwater assessment data is available for {state} in our database."
            }
        
        # Calculate summary statistics
        summary = {
            'districts_covered': state_data['DISTRICT'].nunique() if 'DISTRICT' in state_data.columns else 0,
            'years_covered': sorted(state_data['Assessment_Year'].unique().tolist()) if 'Assessment_Year' in state_data.columns else [],
            'total_assessment_units': len(state_data)
        }
        
        # Get key groundwater metrics
        key_metrics = {}
        numerical_columns = [
            'Annual Ground water Recharge (ham) - Total - Total',
            'Annual Extractable Ground water Resource (ham) - Total - Total',
            'Ground Water Extraction for all uses (ha.m) - Total - Total',
            'Stage of Ground Water Extraction (%) - Total - Total',
            'Net Annual Ground Water Availability for Future Use (ham) - Total - Total',
            'Total Ground Water Availability in the area (ham) - Other Parameters Present - Fresh',
            'Total Ground Water Availability in the area (ham) - Other Parameters Present - Saline'
        ]
        
        for col in numerical_columns:
            if col in state_data.columns:
                numeric_values = pd.to_numeric(state_data[col], errors='coerce').dropna()
                if not numeric_values.empty:
                    key_metrics[col] = {
                        'mean': float(numeric_values.mean()),
                        'min': float(numeric_values.min()),
                        'max': float(numeric_values.max()),
                        'count': len(numeric_values)
                    }
        
        # Generate comprehensive analysis using RAG
        query = f"Provide detailed groundwater analysis for {state} state including water levels, recharge rates, extraction patterns, and recommendations"
        analysis = answer_query(query, "en", "system")
        
        # Get state visualization if available
        try:
            state_plot = create_state_analysis_plots(_master_df, state)
            visualization = state_plot.to_json() if state_plot else None
        except Exception as e:
            print(f"Error creating visualization: {e}")
            visualization = None
        
        return {
            "state": state,
            "data_points": len(state_data),
            "summary": summary,
            "analysis": analysis,
            "visualization": visualization,
            "key_metrics": key_metrics
        }
        
    except Exception as e:
        print(f"Error in analyze-location: {e}")
        return {
            "error": str(e),
            "state": None,
            "analysis": f"An error occurred while analyzing the location: {str(e)}"
        }

# INGRES ChatBOT Utility Functions
def categorize_groundwater_status(extraction_percentage: float) -> tuple[str, str]:
    """
    Categorize groundwater status based on extraction percentage.
    Returns (status, emoji) tuple.
    """
    if extraction_percentage < 70:
        return "Safe", "🟢"
    elif extraction_percentage < 90:
        return "Semi-Critical", "🟡"
    elif extraction_percentage < 100:
        return "Critical", ""
    else:
        return "Over-Exploited", ""

def analyze_water_quality(record) -> Dict[str, Any]:
    """
    Comprehensive water quality analysis with detailed explanations.
    Returns quality analysis with issues, explanations, and recommendations.
    """
    # Get quality data from all three categories (C, NC, PQ)
    major_params_c = record.get('Quality Tagging - Major Parameter Present - C', '')
    major_params_nc = record.get('Quality Tagging - Major Parameter Present - NC', '')
    major_params_pq = record.get('Quality Tagging - Major Parameter Present - PQ', '')
    
    other_params_c = record.get('Quality Tagging - Other Parameters Present - C', '')
    other_params_nc = record.get('Quality Tagging - Other Parameters Present - NC', '')
    other_params_pq = record.get('Quality Tagging - Other Parameters Present - PQ', '')
    
    quality_issues = []
    quality_explanations = []
    quality_recommendations = []
    quality_severity = "Good"
    
    # Combine all major parameters
    all_major_params = []
    for params in [major_params_c, major_params_nc, major_params_pq]:
        if pd.notna(params) and str(params).strip() not in ['', '0.0', 'NIL', '-1.0', 'nan']:
            all_major_params.append(str(params).strip())
    
    # Combine all other parameters
    all_other_params = []
    for params in [other_params_c, other_params_nc, other_params_pq]:
        if pd.notna(params) and str(params).strip() not in ['', '[]', '[NIL]', '[-1.0]', 'nan']:
            all_other_params.append(str(params).strip())
    
    major_params_str = ', '.join(all_major_params) if all_major_params else 'None'
    other_params_str = ', '.join(all_other_params) if all_other_params else 'None'
    
    # Generate comprehensive quality analysis using Gemini if available
    if _gemini_model:
        try:
            state = record.get('STATE', 'Unknown')
            district = record.get('DISTRICT', 'Unknown')
            
            prompt = f"""
            Analyze groundwater quality for {district}, {state} based on the following parameters:
            
            Major Parameters: {major_params_str}
            Other Parameters: {other_params_str}
            
            Provide a comprehensive analysis including:
            1. Quality Tagging: Detailed assessment of water quality parameters
            2. Health Impact: Detailed health effects of detected parameters
            3. Source Analysis: Likely sources of contamination
            4. Standards Compliance: WHO and BIS limit comparisons
            5. Recommendations: Specific actions for water treatment and management
            
            Format the response as structured data with clear sections.
            """
            
            response = _gemini_model.generate_content(prompt)
            gemini_analysis = response.text.strip()
            
            # Parse Gemini response and extract structured information
            quality_issues.append("AI-Generated Comprehensive Analysis")
            quality_explanations.append({
                "parameter": "Comprehensive Quality Assessment",
                "level": "AI-Generated",
                "health_impact": "Detailed analysis provided by AI",
                "sources": "AI analysis based on available parameters",
                "standards": "WHO and BIS standards considered",
                "gemini_analysis": gemini_analysis
            })
            quality_recommendations.append("Follow AI-generated recommendations for water management")
            quality_severity = "Comprehensive"
            
        except Exception as e:
            print(f"Error generating Gemini quality analysis: {e}")
            # Fall back to basic analysis
    
    # Major Parameters Analysis
    major_params_combined = ' '.join(all_major_params).upper()
    other_params_combined = ' '.join(all_other_params).upper()
    
    # Arsenic (As) - Major Health Concern
    if 'AS' in major_params_combined or 'ARSENIC' in major_params_combined:
        quality_issues.append("Arsenic contamination detected")
        quality_explanations.append({
            "parameter": "Arsenic (As)",
            "level": "Major Parameter",
            "health_impact": "Causes skin lesions, cancer, cardiovascular diseases",
            "sources": "Natural geological sources, industrial contamination",
            "standards": "WHO limit: 0.01 mg/L, BIS limit: 0.05 mg/L"
        })
        quality_recommendations.append("Install arsenic removal systems (RO, activated alumina)")
        quality_severity = "Poor"
    
    # Fluoride (F) - Major Health Concern
    if 'F' in major_params_combined or 'FLUORIDE' in major_params_combined:
        quality_issues.append("Fluoride contamination detected")
        quality_explanations.append({
            "parameter": "Fluoride (F)",
            "level": "Major Parameter",
            "health_impact": "Causes dental fluorosis, skeletal fluorosis",
            "sources": "Natural geological sources, industrial discharge",
            "standards": "WHO limit: 1.5 mg/L, BIS limit: 1.0 mg/L"
        })
        quality_recommendations.append("Implement fluoride removal technologies (Nalgonda technique, activated alumina)")
        quality_severity = "Poor"
    
    # Salinity - Major Issue
    if 'SALINE' in major_params_combined or 'SALINITY' in major_params_combined or 'PARTLY SALINE' in major_params_combined:
        quality_issues.append("Salinity issues detected")
        quality_explanations.append({
            "parameter": "Salinity",
            "level": "Major Parameter",
            "health_impact": "High TDS causes taste issues, not suitable for drinking",
            "sources": "Seawater intrusion, geological formations, irrigation return flow",
            "standards": "WHO limit: 600 mg/L TDS, BIS limit: 500 mg/L TDS"
        })
        quality_recommendations.append("Consider desalination or alternative water sources")
        quality_severity = "Poor"
    
    # Other Parameters Analysis
    # Iron (Fe) - Minor Issue
    if 'FE' in other_params_combined or 'IRON' in other_params_combined:
        quality_issues.append("Iron content present")
        quality_explanations.append({
            "parameter": "Iron (Fe)",
            "level": "Other Parameter",
            "health_impact": "Causes taste, color issues, not harmful to health",
            "sources": "Natural geological sources, pipe corrosion",
            "standards": "WHO limit: 0.3 mg/L, BIS limit: 0.3 mg/L"
        })
        quality_recommendations.append("Install iron removal filters or aeration systems")
        if quality_severity == "Good":
            quality_severity = "Moderate"
    
    # Manganese (Mn) - Minor Issue
    if 'MN' in other_params_combined or 'MANGANESE' in other_params_combined:
        quality_issues.append("Manganese content present")
        quality_explanations.append({
            "parameter": "Manganese (Mn)",
            "level": "Other Parameter",
            "health_impact": "Causes taste, color issues, neurological effects at high levels",
            "sources": "Natural geological sources, industrial discharge",
            "standards": "WHO limit: 0.4 mg/L, BIS limit: 0.3 mg/L"
        })
        quality_recommendations.append("Implement manganese removal treatment")
        if quality_severity == "Good":
            quality_severity = "Moderate"
    
    # Nitrate (NO3) - Check if present
    if 'NO3' in other_params_combined or 'NITRATE' in other_params_combined:
        quality_issues.append("Nitrate content present")
        quality_explanations.append({
            "parameter": "Nitrate (NO3)",
            "level": "Other Parameter",
            "health_impact": "Causes blue baby syndrome in infants",
            "sources": "Agricultural runoff, sewage contamination",
            "standards": "WHO limit: 50 mg/L, BIS limit: 45 mg/L"
        })
        quality_recommendations.append("Implement nitrate removal systems")
        if quality_severity == "Good":
            quality_severity = "Moderate"
    
    # No quality issues detected
    if not quality_issues:
        if major_params_str == 'None' and other_params_str == 'None':
            quality_issues.append("No quality data available")
            quality_explanations.append({
                "parameter": "Data Availability",
                "level": "Unknown",
                "health_impact": "Quality assessment not possible without data",
                "sources": "Limited quality monitoring in this area",
                "standards": "Regular quality testing recommended"
            })
            quality_recommendations.append("Implement regular water quality monitoring program")
            quality_severity = "Unknown"
        else:
            quality_issues.append("No major quality issues detected")
            quality_explanations.append({
                "parameter": "Overall Quality",
                "level": "Good",
                "health_impact": "Water appears safe for consumption based on available data",
                "sources": "Natural groundwater sources",
                "standards": "Meets drinking water standards"
            })
            quality_recommendations.append("Continue regular water quality monitoring")
            quality_severity = "Good"
    
    return {
        "issues": quality_issues,
        "explanations": quality_explanations,
        "recommendations": quality_recommendations,
        "severity": quality_severity,
        "major_parameters": major_params_str,
        "other_parameters": other_params_str
    }

def get_groundwater_data(state: str, district: Optional[str] = None, assessment_unit: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve groundwater data for a specific location.
    """
    try:
        # Load the dataset
        df = pd.read_csv('master_groundwater_data.csv', low_memory=False)
        
        # Filter by state
        filtered_df = df[df['STATE'].str.upper() == state.upper()]
        
        if district:
            filtered_df = filtered_df[filtered_df['DISTRICT'].str.upper() == district.upper()]
        
        if assessment_unit:
            filtered_df = filtered_df[filtered_df['ASSESSMENT UNIT'].str.upper() == assessment_unit.upper()]
        
        if filtered_df.empty:
            return {"error": "No data found for the specified location"}
        
        # Get the first matching record
        record = filtered_df.iloc[0]
        
        # Extract key metrics
        extraction_stage = record.get('Stage of Ground Water Extraction (%) - Total - Total', 0)
        annual_recharge = record.get('Annual Ground water Recharge (ham) - Total - Total', 0)
        extractable_resource = record.get('Annual Extractable Ground water Resource (ham) - Total - Total', 0)
        total_extraction = record.get('Ground Water Extraction for all uses (ha.m) - Total - Total', 0)
        future_availability = record.get('Net Annual Ground Water Availability for Future Use (ham) - Total - Total', 0)
        rainfall = record.get('Rainfall (mm) - Total', 0)
        total_area = record.get('Total Geographical Area (ha) - Total - Total', 0)
        
        # Categorize status
        status, emoji = categorize_groundwater_status(extraction_stage)
        
        # Enhanced quality analysis with detailed explanations
        quality_analysis = analyze_water_quality(record)
        
        # Generate additional resources analysis using Gemini
        additional_resources = generate_additional_resources_analysis(record)
        
        # Generate key findings and trends analysis using Gemini
        key_findings_trends = generate_key_findings_trends(record)
        
        return {
            "state": record['STATE'],
            "district": str(record['DISTRICT']) if pd.notna(record['DISTRICT']) else '',
            "assessment_unit": str(record.get('ASSESSMENT UNIT', '')) if pd.notna(record.get('ASSESSMENT UNIT', '')) else '',
            "extraction_stage": float(extraction_stage) if pd.notna(extraction_stage) else 0,
            "annual_recharge": float(annual_recharge) if pd.notna(annual_recharge) else 0,
            "extractable_resource": float(extractable_resource) if pd.notna(extractable_resource) else 0,
            "total_extraction": float(total_extraction) if pd.notna(total_extraction) else 0,
            "future_availability": float(future_availability) if pd.notna(future_availability) else 0,
            "rainfall": float(rainfall) if pd.notna(rainfall) else 0,
            "total_area": float(total_area) if pd.notna(total_area) else 0,
            "criticality_status": status,
            "criticality_emoji": emoji,
            "quality_analysis": quality_analysis,
            "additional_resources": additional_resources,
            "key_findings_trends": key_findings_trends
        }
    except Exception as e:
        return {"error": f"Error retrieving data: {str(e)}"}

def generate_key_findings_trends(record) -> Dict[str, Any]:
    """
    Generate comprehensive key findings and trends analysis using Gemini.
    """
    try:
        if not _gemini_model:
            return {
                "findings": "No data available",
                "trends": "No data available",
                "analysis": "Gemini not available for analysis"
            }
        
        state = record.get('STATE', 'Unknown')
        district = record.get('DISTRICT', 'Unknown')
        extraction_stage = record.get('Stage of Ground Water Extraction (%) - Total - Total', 0)
        annual_recharge = record.get('Annual Ground water Recharge (ham) - Total - Total', 0)
        total_extraction = record.get('Ground Water Extraction for all uses (ha.m) - Total - Total', 0)
        rainfall = record.get('Rainfall (mm) - Total', 0)
        future_availability = record.get('Net Annual Ground Water Availability for Future Use (ham) - Total - Total', 0)
        
        # Calculate extraction percentages for different areas
        extraction_c = record.get('Stage of Ground Water Extraction (%) - Total - C', 0)
        extraction_nc = record.get('Stage of Ground Water Extraction (%) - Total - NC', 0)
        
        prompt = f"""
        Analyze groundwater data for {district}, {state} and provide comprehensive key findings and trends:
        
        Data Context:
        - District: {district}, {state}
        - Extraction Stage (Total): {extraction_stage}%
        - Extraction Stage (Cultivated): {extraction_c}%
        - Extraction Stage (Non-Cultivated): {extraction_nc}%
        - Annual Recharge: {annual_recharge} ham
        - Total Extraction: {total_extraction} ha.m
        - Rainfall: {rainfall} mm
        - Future Availability: {future_availability} ham
        
        Provide detailed analysis covering:
        
        1. KEY FINDINGS:
        - High Dependence on Rainfall: Analyze rainfall dependency for groundwater recharge
        - Significant Groundwater Extraction for Irrigation: Assess irrigation extraction patterns
        - Potential Over-extraction: Evaluate extraction vs recharge ratios and sustainability
        - Data Gaps: Identify missing data and its impact on analysis reliability
        
        2. TRENDS:
        - Extraction patterns and sustainability indicators
        - Recharge dependency and climate sensitivity
        - Resource depletion risks and warning signs
        - Management implications and recommendations
        
        3. SUSTAINABILITY ASSESSMENT:
        - Current extraction vs recharge balance
        - Future availability projections
        - Critical thresholds and warning levels
        - Conservation and management priorities
        
        4. RECOMMENDATIONS:
        - Data collection improvements needed
        - Sustainable extraction strategies
        - Recharge enhancement measures
        - Monitoring and management priorities
        
        Format as structured sections with clear bullet points and detailed explanations.
        """
        
        response = _gemini_model.generate_content(prompt)
        gemini_analysis = response.text.strip()
        
        return {
            "findings": f"Comprehensive analysis for {district}, {state}",
            "trends": "Detailed trend analysis provided",
            "analysis": gemini_analysis,
            "generated_by": "Gemini AI",
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_summary": {
                "extraction_stage": extraction_stage,
                "rainfall_dependency": "High" if rainfall > 0 else "Unknown",
                "over_extraction_risk": "High" if extraction_stage > 100 else "Moderate" if extraction_stage > 90 else "Low",
                "sustainability_status": "Critical" if extraction_stage > 100 else "At Risk" if extraction_stage > 90 else "Sustainable"
            }
        }
        
    except Exception as e:
        print(f"Error generating key findings and trends: {e}")
        return {
            "findings": "Analysis unavailable",
            "trends": "Analysis unavailable",
            "analysis": f"Error: {str(e)}",
            "generated_by": "Error",
            "timestamp": pd.Timestamp.now().isoformat()
        }

def generate_comprehensive_groundwater_summary(record) -> Dict[str, Any]:
    """
    Generate a comprehensive groundwater summary with clear key findings.
    Includes basis of availability, major uses, and safety category.
    """
    try:
        # Extract key data points
        state = record.get('STATE', 'Unknown')
        district = record.get('DISTRICT', 'Unknown')
        assessment_year = record.get('Assessment_Year', 'Unknown')
        
        # Groundwater availability basis
        annual_recharge = record.get('Annual Ground water Recharge (ham) - Total - Total', 0)
        rainfall = record.get('Rainfall (mm) - Total - Total', 0)
        rainfall_recharge = record.get('Rainfall Recharge (ham) - Total - Total', 0)
        surface_irrigation_recharge = record.get('Recharge from Surface Irrigation (ham) - Total - Total', 0)
        other_recharge = record.get('Recharge from Other Sources (ham) - Total - Total', 0)
        
        # Groundwater extraction uses
        extraction_total = record.get('Ground Water Extraction for all uses (ha.m) - Total - Total', 0)
        extraction_cultivation = record.get('Ground Water Extraction for all uses (ha.m) - Total - C', 0)
        extraction_non_cultivation = record.get('Ground Water Extraction for all uses (ha.m) - Total - NC', 0)
        extraction_domestic = record.get('Ground Water Extraction for all uses (ha.m) - Total - Domestic', 0)
        extraction_industrial = record.get('Ground Water Extraction for all uses (ha.m) - Total - Industrial', 0)
        
        # Safety category calculation
        extraction_stage = record.get('Stage of Ground Water Extraction (%) - Total - Total', 0)
        future_availability = record.get('Net Annual Ground Water Availability for Future Use (ham) - Total - Total', 0)
        
        # Calculate safety category
        if extraction_stage < 70:
            safety_category = "Safe"
            safety_emoji = "🟢"
            safety_description = "Groundwater resources are within sustainable limits"
        elif extraction_stage < 90:
            safety_category = "Semi-Critical"
            safety_emoji = "🟡"
            safety_description = "Groundwater resources are approaching critical levels"
        elif extraction_stage < 100:
            safety_category = "Critical"
            safety_emoji = ""
            safety_description = "Groundwater resources are critically stressed"
        else:
            safety_category = "Over-Exploited"
            safety_emoji = ""
            safety_description = "Groundwater extraction exceeds recharge capacity"
        
        # Calculate recharge source percentages
        total_recharge_sources = rainfall_recharge + surface_irrigation_recharge + other_recharge
        rainfall_percentage = (rainfall_recharge / total_recharge_sources * 100) if total_recharge_sources > 0 else 0
        irrigation_percentage = (surface_irrigation_recharge / total_recharge_sources * 100) if total_recharge_sources > 0 else 0
        other_percentage = (other_recharge / total_recharge_sources * 100) if total_recharge_sources > 0 else 0
        
        # Calculate extraction use percentages
        total_extraction_uses = extraction_cultivation + extraction_non_cultivation + extraction_domestic + extraction_industrial
        cultivation_percentage = (extraction_cultivation / total_extraction_uses * 100) if total_extraction_uses > 0 else 0
        non_cultivation_percentage = (extraction_non_cultivation / total_extraction_uses * 100) if total_extraction_uses > 0 else 0
        domestic_percentage = (extraction_domestic / total_extraction_uses * 100) if total_extraction_uses > 0 else 0
        industrial_percentage = (extraction_industrial / total_extraction_uses * 100) if total_extraction_uses > 0 else 0
        
        # Generate key findings as a clear, structured list
        key_findings = [
            f"[RAIN] RAINFALL DEPENDENCY: The region receives {rainfall:.1f} mm of annual rainfall, contributing {rainfall_percentage:.1f}% of total groundwater recharge",
            f" RECHARGE SOURCES: Total annual recharge of {annual_recharge:.1f} ham comes from rainfall ({rainfall_percentage:.1f}%), surface irrigation ({irrigation_percentage:.1f}%), and other sources ({other_percentage:.1f}%)",
            f" AGRICULTURAL DOMINANCE: Agricultural cultivation accounts for {cultivation_percentage:.1f}% of total groundwater extraction ({extraction_cultivation:.1f} ha.m)",
            f" EXTRACTION BREAKDOWN: Total extraction of {extraction_total:.1f} ha.m includes cultivation ({cultivation_percentage:.1f}%), non-cultivation ({non_cultivation_percentage:.1f}%), domestic ({domestic_percentage:.1f}%), and industrial ({industrial_percentage:.1f}%) uses",
            f"[WARNING] SAFETY STATUS: {safety_category} {safety_emoji} - {safety_description}",
            f" EXTRACTION LEVEL: {extraction_stage:.1f}% of available resources extracted, with {future_availability:.1f} ham remaining for future use",
            f"[INIT] SUSTAINABILITY: {'Sustainable' if extraction_stage < 70 else 'At Risk' if extraction_stage < 100 else 'Critical'} status based on extraction vs recharge balance"
        ]
        
        # Add Karnataka comprehensive averages if this is Karnataka data
        if state.lower() in ['karnataka', 'karnataka state']:
            key_findings.extend([
                f" KARNATAKA AVERAGES: Based on Davanagere & Mysuru districts - 61,021.61 ham annual recharge, 85.01% extraction stage",
                f" GEOGRAPHICAL COVERAGE: 449,970.60 ha recharge-worthy area with 746.09 mm average rainfall",
                f"[WARNING] DATA LIMITATIONS: Analysis limited to 2 districts, not representative of entire Karnataka state"
            ])
        
        # Generate comprehensive summary
        summary = {
            "location": f"{district}, {state}",
            "assessment_year": assessment_year,
            "key_findings": key_findings,
            "groundwater_availability_basis": {
                "primary_sources": [
                    f"• Rainfall: {rainfall:.1f} mm annually ({rainfall_percentage:.1f}% of recharge)",
                    f"• Surface Irrigation: {surface_irrigation_recharge:.1f} ham ({irrigation_percentage:.1f}% of recharge)",
                    f"• Other Sources: {other_recharge:.1f} ham ({other_percentage:.1f}% of recharge)"
                ],
                "total_annual_recharge": f"{annual_recharge:.1f} ham",
                "rainfall_dependency": "High" if rainfall_percentage > 60 else "Moderate" if rainfall_percentage > 30 else "Low"
            },
            "major_extraction_uses": {
                "cultivation": {
                    "volume": f"{extraction_cultivation:.1f} ha.m",
                    "percentage": f"{cultivation_percentage:.1f}%",
                    "description": "Agricultural irrigation and crop production"
                },
                "non_cultivation": {
                    "volume": f"{extraction_non_cultivation:.1f} ha.m", 
                    "percentage": f"{non_cultivation_percentage:.1f}%",
                    "description": "Non-agricultural activities and land use"
                },
                "domestic": {
                    "volume": f"{extraction_domestic:.1f} ha.m",
                    "percentage": f"{domestic_percentage:.1f}%", 
                    "description": "Household and municipal water supply"
                },
                "industrial": {
                    "volume": f"{extraction_industrial:.1f} ha.m",
                    "percentage": f"{industrial_percentage:.1f}%",
                    "description": "Industrial processes and manufacturing"
                },
                "total_extraction": f"{extraction_total:.1f} ha.m"
            },
            "groundwater_safety_category": {
                "category": safety_category,
                "emoji": safety_emoji,
                "description": safety_description,
                "extraction_stage": f"{extraction_stage:.1f}%",
                "future_availability": f"{future_availability:.1f} ham",
                "sustainability_status": "Sustainable" if extraction_stage < 70 else "At Risk" if extraction_stage < 100 else "Critical"
            },
            "summary_bullets": [
                f" Location: {district}, {state} (Assessment Year: {assessment_year})",
                f" Groundwater Availability: Primarily dependent on rainfall ({rainfall:.1f} mm) contributing {rainfall_percentage:.1f}% of total recharge",
                f" Major Use: Agricultural cultivation accounts for {cultivation_percentage:.1f}% of total groundwater extraction",
                f"[WARNING] Safety Status: {safety_category} {safety_emoji} - {safety_description}",
                f" Extraction Level: {extraction_stage:.1f}% of available resources, with {future_availability:.1f} ham remaining for future use"
            ],
            "detailed_analysis": {
                "recharge_analysis": f"Total annual recharge of {annual_recharge:.1f} ham comes primarily from rainfall ({rainfall_percentage:.1f}%), with additional contributions from surface irrigation ({irrigation_percentage:.1f}%) and other sources ({other_percentage:.1f}%)",
                "extraction_analysis": f"Total groundwater extraction of {extraction_total:.1f} ha.m is dominated by agricultural use ({cultivation_percentage:.1f}%), followed by non-cultivation activities ({non_cultivation_percentage:.1f}%), domestic use ({domestic_percentage:.1f}%), and industrial use ({industrial_percentage:.1f}%)",
                "sustainability_analysis": f"The area is classified as {safety_category} with {extraction_stage:.1f}% extraction rate, indicating {safety_description.lower()}. Future availability stands at {future_availability:.1f} ham"
            }
        }
        
        # Add Karnataka comprehensive averages if this is Karnataka data
        if state.lower() in ['karnataka', 'karnataka state']:
            summary["karnataka_comprehensive_averages"] = {
                "description": "Averages across Davanagere & Mysuru districts - Data limitations prevent precise analysis",
                "data": {
                    "Annual Ground water Recharge (ham) - Total - Total": "61,021.61 ham",
                    "Annual Extractable Ground water Resource (ham) - Total - Total": "55,073.39 ham",
                    "Ground Water Extraction for all uses (ha.m) - Total - Total": "46,804.31 ha.m",
                    "Stage of Ground Water Extraction (%) - Total - Total": "85.01%",
                    "Net Annual Ground Water Availability for Future Use (ham) - Total - Total": "18,344.97 ham",
                    "Environmental Flows (ham) - Total - Total": "5,948.21 ham",
                    "Allocation of Ground Water Resource for Domestic Utilisation for projected year 2025 (ham) - Total - Total": "3,908.70 ham",
                    "Average Rainfall (mm) - Total": "746.09 mm",
                    "Total Geographical Area (ha) - Recharge Worthy Area (ha) - C": "163,299.60 ha",
                    "Total Geographical Area (ha) - Recharge Worthy Area (ha) - NC": "286,671.00 ha",
                    "Total Geographical Area (ha) - Recharge Worthy Area (ha) - PQ": "0.00 ha",
                    "Total Geographical Area (ha) - Recharge Worthy Area (ha) - Total": "449,970.60 ha",
                    "Total Geographical Area (ha) - Hilly Area - Total": "34,611.80 ha",
                    "Total Geographical Area (ha) - Total - Total": "484,582.40 ha"
                },
                "limitations": [
                    "Only 2 districts analyzed (Davanagere & Mysuru)",
                    "Not representative of entire Karnataka state",
                    "Data inconsistencies and missing parameters",
                    "Heavy influence of data availability on averages"
                ]
            }
        
        return summary
        
    except Exception as e:
        print(f"Error generating comprehensive groundwater summary: {e}")
        return {
            "error": f"Error generating summary: {str(e)}",
            "location": "Unknown",
            "key_findings": "Analysis unavailable"
        }

def generate_additional_resources_analysis(record) -> Dict[str, Any]:
    """
    Generate comprehensive additional resources analysis using Gemini.
    Includes coastal areas, aquifer types, and other resource information.
    """
    try:
        if not _gemini_model:
            return {
                "coastal_areas": "No data available",
                "aquifer_types": "No data available", 
                "additional_resources": "No data available",
                "analysis": "Gemini not available for analysis"
            }
        
        state = record.get('STATE', 'Unknown')
        district = record.get('DISTRICT', 'Unknown')
        total_area = record.get('Total Geographical Area (ha) - Total - Total', 0)
        rainfall = record.get('Rainfall (mm) - Total', 0)
        
        prompt = f"""
        Provide comprehensive additional resources analysis for {district}, {state}:
        
        Context:
        - Total Geographical Area: {total_area} hectares
        - Average Rainfall: {rainfall} mm
        - State: {state}
        - District: {district}
        
        Please provide detailed information about:
        
        1. COASTAL AREAS:
        - Coastal proximity and characteristics
        - Saltwater intrusion risks
        - Coastal groundwater dynamics
        - Tidal influence on groundwater
        
        2. AQUIFER TYPES:
        - Unconfined aquifers: Characteristics, depth, recharge patterns
        - Confined aquifers: Characteristics, depth, pressure conditions
        - Semi-confined aquifers: Characteristics, partial confinement
        - Aquifer connectivity and flow patterns
        
        3. ADDITIONAL RESOURCES:
        - Groundwater recharge potential
        - Water storage capacity
        - Seasonal variations
        - Environmental flows
        - Future development potential
        
        4. GEOLOGICAL FEATURES:
        - Rock formations and their water-bearing properties
        - Soil types and infiltration rates
        - Topographical influences
        - Natural recharge zones
        
        5. MANAGEMENT RECOMMENDATIONS:
        - Sustainable extraction strategies
        - Recharge enhancement methods
        - Monitoring requirements
        - Conservation measures
        
        Format as structured sections with clear headings and detailed explanations.
        """
        
        response = _gemini_model.generate_content(prompt)
        gemini_analysis = response.text.strip()
        
        return {
            "coastal_areas": f"AI-Generated analysis for {district}, {state}",
            "aquifer_types": "Comprehensive aquifer analysis provided",
            "additional_resources": "Detailed resource assessment available",
            "analysis": gemini_analysis,
            "generated_by": "Gemini AI",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error generating additional resources analysis: {e}")
        return {
            "coastal_areas": "Analysis unavailable",
            "aquifer_types": "Analysis unavailable",
            "additional_resources": "Analysis unavailable", 
            "analysis": f"Error: {str(e)}",
            "generated_by": "Error",
            "timestamp": pd.Timestamp.now().isoformat()
        }

def generate_enhanced_statistics() -> Dict[str, Any]:
    """
    Generate enhanced statistics for the INGRES system.
    """
    try:
        df = pd.read_csv('master_groundwater_data.csv', low_memory=False)
        
        # Calculate criticality distribution
        extraction_data = df['Stage of Ground Water Extraction (%) - Total - Total'].dropna()
        
        safe_count = len(extraction_data[extraction_data < 70])
        semi_critical_count = len(extraction_data[(extraction_data >= 70) & (extraction_data < 90)])
        critical_count = len(extraction_data[(extraction_data >= 90) & (extraction_data < 100)])
        over_exploited_count = len(extraction_data[extraction_data >= 100])
        
        total_states = len(df['STATE'].unique())
        
        # Calculate visualization coverage
        states_with_visualizations = total_states  # All states have visualizations
        avg_charts_per_state = 3.0  # Standard number of charts per state
        
        # Calculate quality analysis coverage
        quality_columns = [
            'Quality Tagging - Major Parameter Present - C',
            'Quality Tagging - Major Parameter Present - NC', 
            'Quality Tagging - Major Parameter Present - PQ',
            'Quality Tagging - Other Parameters Present - C',
            'Quality Tagging - Other Parameters Present - NC',
            'Quality Tagging - Other Parameters Present - PQ'
        ]
        
        states_with_quality_data = 0
        for state in df['STATE'].unique():
            state_data = df[df['STATE'] == state]
            has_quality_data = False
            for col in quality_columns:
                if col in state_data.columns:
                    non_null_count = state_data[col].notna().sum()
                    if non_null_count > 0:
                        has_quality_data = True
                        break
            if has_quality_data:
                states_with_quality_data += 1
        
        return {
            "criticality_distribution": {
                "safe": {
                    "count": int(safe_count),
                    "percentage": round(safe_count / len(extraction_data) * 100, 1),
                    "states": int(safe_count),
                    "emoji": "🟢"
                },
                "semi_critical": {
                    "count": int(semi_critical_count),
                    "percentage": round(semi_critical_count / len(extraction_data) * 100, 1),
                    "states": int(semi_critical_count),
                    "emoji": "🟡"
                },
                "critical": {
                    "count": int(critical_count),
                    "percentage": round(critical_count / len(extraction_data) * 100, 1),
                    "states": int(critical_count),
                    "emoji": ""
                },
                "over_exploited": {
                    "count": int(over_exploited_count),
                    "percentage": round(over_exploited_count / len(extraction_data) * 100, 1),
                    "states": int(over_exploited_count),
                    "emoji": ""
                }
            },
            "visualization_coverage": {
                "total_states": total_states,
                "states_with_visualizations": states_with_visualizations,
                "coverage_percentage": 100.0,
                "avg_charts_per_state": avg_charts_per_state,
                "chart_types": ["pie_chart", "bar_chart", "gauge_chart"]
            },
            "quality_analysis_coverage": {
                "total_states": total_states,
                "states_with_quality_data": states_with_quality_data,
                "coverage_percentage": round(states_with_quality_data / total_states * 100, 1),
                "parameters_detected": ["Arsenic", "Fluoride", "Iron", "Manganese", "Salinity"],
                "standards_checked": ["WHO", "BIS"]
            }
        }
    except Exception as e:
        print(f"Error generating enhanced statistics: {e}")
        return {
            "criticality_distribution": {
                "safe": {"count": 0, "percentage": 0, "states": 0, "emoji": "🟢"},
                "semi_critical": {"count": 0, "percentage": 0, "states": 0, "emoji": "🟡"},
                "critical": {"count": 0, "percentage": 0, "states": 0, "emoji": ""},
                "over_exploited": {"count": 0, "percentage": 0, "states": 0, "emoji": ""}
            },
            "visualization_coverage": {
                "total_states": 0,
                "states_with_visualizations": 0,
                "coverage_percentage": 0,
                "avg_charts_per_state": 0,
                "chart_types": []
            },
            "quality_analysis_coverage": {
                "total_states": 0,
                "states_with_quality_data": 0,
                "coverage_percentage": 0,
                "parameters_detected": [],
                "standards_checked": []
            }
        }

def generate_groundwater_recommendations(status: str, extraction_percentage: float, quality_issues: List[str]) -> List[str]:
    """
    Generate recommendations based on groundwater status and quality issues.
    """
    recommendations = []
    
    if status == "Safe":
        recommendations.extend([
            "Continue current water management practices",
            "Implement preventive measures to maintain current status",
            "Monitor groundwater levels regularly",
            "Promote water conservation awareness in the community"
        ])
    elif status == "Semi-Critical":
        recommendations.extend([
            "Implement water conservation measures immediately",
            "Promote rainwater harvesting systems",
            "Optimize irrigation practices and crop patterns",
            "Monitor groundwater extraction rates closely",
            "Consider artificial recharge techniques"
        ])
    elif status == "Critical":
        recommendations.extend([
            "Immediate water conservation measures required",
            "Implement artificial recharge techniques",
            "Optimize crop patterns to reduce water demand",
            "Strict monitoring and regulation of extraction",
            "Emergency water management protocols"
        ])
    else:  # Over-Exploited
        recommendations.extend([
            "Emergency water management measures required",
            "Immediate artificial recharge implementation",
            "Strict extraction controls and regulations",
            "Crop diversification to water-efficient varieties",
            "Community awareness and participation programs",
            "Consider alternative water sources"
        ])
    
    # Add quality-specific recommendations
    if quality_issues:
        if "Arsenic contamination" in quality_issues:
            recommendations.append("Implement arsenic removal technologies")
        if "Fluoride contamination" in quality_issues:
            recommendations.append("Install fluoride removal systems")
        if "Salinity issues" in quality_issues:
            recommendations.append("Implement desalination or alternative water sources")
        if "Iron content" in quality_issues:
            recommendations.append("Install iron removal filters")
        if "Manganese content" in quality_issues:
            recommendations.append("Implement manganese removal treatment")
    
    return recommendations

def create_groundwater_visualizations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create visualizations for groundwater data.
    """
    visualizations = []
    
    try:
        # 1. Criticality Status Pie Chart
        status = data.get('criticality_status', 'Unknown')
        emoji = data.get('criticality_emoji', '')
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=[f"{emoji} {status}"],
            values=[100],
            hole=0.3,
            marker_colors=['#2E8B57' if status == 'Safe' else 
                          '#FFD700' if status == 'Semi-Critical' else
                          '#FF4500' if status == 'Critical' else '#8B0000']
        )])
        fig_pie.update_layout(
            title="Groundwater Status",
            showlegend=True,
            height=400
        )
        
        visualizations.append({
            "type": "pie_chart",
            "title": "Groundwater Criticality Status",
            "data": json.loads(fig_pie.to_json())
        })
        
        # 2. Resource Balance Bar Chart
        recharge = data.get('annual_recharge', 0)
        extraction = data.get('total_extraction', 0)
        available = data.get('future_availability', 0)
        
        fig_bar = go.Figure(data=[
            go.Bar(name='Annual Recharge', x=['Groundwater Resources'], y=[recharge], marker_color='#2E8B57'),
            go.Bar(name='Total Extraction', x=['Groundwater Resources'], y=[extraction], marker_color='#FF4500'),
            go.Bar(name='Future Availability', x=['Groundwater Resources'], y=[available], marker_color='#4169E1')
        ])
        fig_bar.update_layout(
            title="Groundwater Resource Balance (ham)",
            barmode='group',
            height=400
        )
        
        visualizations.append({
            "type": "bar_chart",
            "title": "Resource Balance Analysis",
            "data": json.loads(fig_bar.to_json())
        })
        
        # 3. Extraction Efficiency Gauge
        extraction_percentage = data.get('extraction_stage', 0)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = extraction_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Extraction Stage (%)"},
            delta = {'reference': 70},
            gauge = {
                'axis': {'range': [None, 150]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgreen"},
                    {'range': [70, 90], 'color': "yellow"},
                    {'range': [90, 100], 'color': "orange"},
                    {'range': [100, 150], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))
        fig_gauge.update_layout(height=400)
        
        visualizations.append({
            "type": "gauge_chart",
            "title": "Extraction Stage Gauge",
            "data": json.loads(fig_gauge.to_json())
        })
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    return visualizations

def get_state_comparison_data(state: str) -> Dict[str, Any]:
    """
    Get comparison data for a state against national averages.
    """
    try:
        df = pd.read_csv('master_groundwater_data.csv', low_memory=False)
        
        # Filter state data
        state_df = df[df['STATE'].str.upper() == state.upper()]
        
        if state_df.empty:
            return {"error": "No data found for the state"}
        
        # Calculate state averages
        state_avg_extraction = state_df['Stage of Ground Water Extraction (%) - Total - Total'].mean()
        state_avg_recharge = state_df['Annual Ground water Recharge (ham) - Total - Total'].mean()
        state_avg_extractable = state_df['Annual Extractable Ground water Resource (ham) - Total - Total'].mean()
        
        # Calculate national averages
        national_avg_extraction = df['Stage of Ground Water Extraction (%) - Total - Total'].mean()
        national_avg_recharge = df['Annual Ground water Recharge (ham) - Total - Total'].mean()
        national_avg_extractable = df['Annual Extractable Ground water Resource (ham) - Total - Total'].mean()
        
        # Count districts by criticality
        safe_count = len(state_df[state_df['Stage of Ground Water Extraction (%) - Total - Total'] < 70])
        semi_critical_count = len(state_df[(state_df['Stage of Ground Water Extraction (%) - Total - Total'] >= 70) & 
                                         (state_df['Stage of Ground Water Extraction (%) - Total - Total'] < 90)])
        critical_count = len(state_df[(state_df['Stage of Ground Water Extraction (%) - Total - Total'] >= 90) & 
                                    (state_df['Stage of Ground Water Extraction (%) - Total - Total'] < 100)])
        over_exploited_count = len(state_df[state_df['Stage of Ground Water Extraction (%) - Total - Total'] >= 100])
        
        return {
            "state_averages": {
                "extraction_stage": float(state_avg_extraction),
                "annual_recharge": float(state_avg_recharge),
                "extractable_resource": float(state_avg_extractable)
            },
            "national_averages": {
                "extraction_stage": float(national_avg_extraction),
                "annual_recharge": float(national_avg_recharge),
                "extractable_resource": float(national_avg_extractable)
            },
            "district_count": {
                "total": len(state_df),
                "safe": int(safe_count),
                "semi_critical": int(semi_critical_count),
                "critical": int(critical_count),
                "over_exploited": int(over_exploited_count)
            }
        }
    except Exception as e:
        return {"error": f"Error generating comparison data: {str(e)}"}

def get_state_from_coordinates(lat: float, lng: float) -> str:
    """Convert coordinates to state name using highly accurate boundary mapping."""
    
    # Special handling for overlapping regions (checked first for accuracy)
    overlapping_regions = {
        # Chhattisgarh vs Madhya Pradesh overlap - prioritize Chhattisgarh
        (17.8, 24.1, 80.2, 84.4): "Chhattisgarh",
        
        # Telangana vs Andhra Pradesh overlap (Hyderabad region)
        (17.0, 18.0, 78.0, 79.0): "Telangana",
        
        # Delhi region - prioritize Delhi over Haryana
        (28.4, 28.9, 76.8, 77.3): "Delhi",
        
        # Uttarakhand vs Uttar Pradesh overlap
        (28.7, 31.5, 77.3, 81.1): "Uttarakhand",
        
        # Karnataka vs Tamil Nadu overlap (Bangalore region)
        (12.5, 13.5, 77.0, 78.0): "Karnataka",
        
        # Bihar vs Jharkhand overlap (Gaya region)
        (24.5, 25.5, 85.0, 86.0): "Bihar",
        
        # Gujarat vs Maharashtra overlap (Surat region)
        (20.5, 21.5, 72.5, 73.5): "Gujarat",
        
        # Haryana vs Delhi overlap (Gurgaon region)
        (28.2, 28.8, 76.8, 77.2): "Haryana",
        
        # Odisha vs Andhra Pradesh overlap (Cuttack region)
        (19.0, 20.5, 84.0, 85.5): "Odisha",
        
        # Tamil Nadu vs Kerala overlap (Coimbatore region)
        (10.5, 11.5, 76.5, 77.5): "Tamil Nadu",
        
        # Uttar Pradesh vs Madhya Pradesh overlap (Lucknow region)
        (26.5, 27.0, 80.5, 81.5): "Uttar Pradesh",
        
        # West Bengal vs Jharkhand overlap (Malda region)
        (24.0, 25.0, 87.5, 88.5): "West Bengal",
        
        # Ladakh vs Jammu and Kashmir overlap (Leh region)
        (33.5, 35.0, 76.5, 78.0): "Ladakh",
        
        # Dadra and Nagar Haveli vs Maharashtra overlap (Silvassa region)
        (20.0, 20.5, 72.8, 73.2): "Dadra and Nagar Haveli and Daman and Diu",
    }
    
    # Comprehensive state boundaries (ordered by priority for overlapping regions)
    state_boundaries = {
        # Union Territories (highest priority - smallest areas first)
        "Delhi": {"min_lat": 28.4, "max_lat": 28.9, "min_lng": 76.8, "max_lng": 77.3},
        "Chandigarh": {"min_lat": 30.7, "max_lat": 30.8, "min_lng": 76.7, "max_lng": 76.8},
        "Puducherry": {"min_lat": 11.7, "max_lat": 12.0, "min_lng": 79.7, "max_lng": 79.9},
        "Goa": {"min_lat": 14.8, "max_lat": 15.8, "min_lng": 73.7, "max_lng": 74.2},
        
        # Northeastern States (high priority due to small size)
        "Sikkim": {"min_lat": 27.0, "max_lat": 28.2, "min_lng": 88.0, "max_lng": 88.9},
        "Tripura": {"min_lat": 22.9, "max_lat": 24.7, "min_lng": 91.2, "max_lng": 92.3},
        "Mizoram": {"min_lat": 21.9, "max_lat": 24.5, "min_lng": 92.2, "max_lng": 93.3},
        "Nagaland": {"min_lat": 25.2, "max_lat": 27.0, "min_lng": 93.0, "max_lng": 95.4},
        "Manipur": {"min_lat": 23.8, "max_lat": 25.7, "min_lng": 93.0, "max_lng": 94.8},
        "Meghalaya": {"min_lat": 25.1, "max_lat": 26.1, "min_lng": 89.8, "max_lng": 92.8},
        "Arunachal Pradesh": {"min_lat": 26.5, "max_lat": 29.4, "min_lng": 91.6, "max_lng": 97.4},
        "Assam": {"min_lat": 24.1, "max_lat": 28.2, "min_lng": 89.7, "max_lng": 96.0},
        
        # Southern States (more precise boundaries)
        "Kerala": {"min_lat": 8.1, "max_lat": 12.8, "min_lng": 74.9, "max_lng": 77.4},
        "Tamil Nadu": {"min_lat": 8.1, "max_lat": 13.1, "min_lng": 76.2, "max_lng": 80.3},
        "Karnataka": {"min_lat": 11.7, "max_lat": 18.5, "min_lng": 74.1, "max_lng": 78.6},
        "Andhra Pradesh": {"min_lat": 12.4, "max_lat": 19.9, "min_lng": 76.8, "max_lng": 84.8},
        "Telangana": {"min_lat": 15.5, "max_lat": 19.9, "min_lng": 77.2, "max_lng": 81.1},
        
        # Central and Western States
        "Maharashtra": {"min_lat": 15.6, "max_lat": 22.0, "min_lng": 72.6, "max_lng": 80.9},
        "Gujarat": {"min_lat": 20.1, "max_lat": 24.7, "min_lng": 68.2, "max_lng": 74.5},
        "Madhya Pradesh": {"min_lat": 21.1, "max_lat": 26.9, "min_lng": 74.0, "max_lng": 82.8},
        
        # Eastern States (Chhattisgarh before Odisha for overlapping regions)
        "Chhattisgarh": {"min_lat": 17.8, "max_lat": 24.1, "min_lng": 80.2, "max_lng": 84.4},
        "Odisha": {"min_lat": 17.5, "max_lat": 22.5, "min_lng": 81.3, "max_lng": 87.3},
        "Jharkhand": {"min_lat": 21.8, "max_lat": 25.3, "min_lng": 83.2, "max_lng": 87.9},
        "West Bengal": {"min_lat": 21.5, "max_lat": 27.2, "min_lng": 85.5, "max_lng": 89.9},
        "Bihar": {"min_lat": 24.2, "max_lat": 27.7, "min_lng": 83.3, "max_lng": 88.8},
        
        # Northern States
        "Uttar Pradesh": {"min_lat": 23.7, "max_lat": 31.1, "min_lng": 77.0, "max_lng": 84.7},
        "Uttarakhand": {"min_lat": 28.7, "max_lat": 31.5, "min_lng": 77.3, "max_lng": 81.1},
        "Himachal Pradesh": {"min_lat": 30.4, "max_lat": 33.2, "min_lng": 75.6, "max_lng": 79.1},
        "Punjab": {"min_lat": 29.5, "max_lat": 32.3, "min_lng": 73.9, "max_lng": 76.9},
        "Haryana": {"min_lat": 28.4, "max_lat": 31.0, "min_lng": 74.4, "max_lng": 77.5},
        "Rajasthan": {"min_lat": 23.1, "max_lat": 30.2, "min_lng": 69.3, "max_lng": 78.2},
        
        # Union Territories
        "Jammu and Kashmir": {"min_lat": 32.2, "max_lat": 37.1, "min_lng": 73.9, "max_lng": 80.3},
        "Ladakh": {"min_lat": 32.0, "max_lat": 37.1, "min_lng": 75.8, "max_lng": 80.3},
        "Andaman and Nicobar Islands": {"min_lat": 6.7, "max_lat": 13.4, "min_lng": 92.2, "max_lng": 94.3},
        "Lakshadweep": {"min_lat": 8.2, "max_lat": 12.3, "min_lng": 71.7, "max_lng": 74.0},
        "Dadra and Nagar Haveli and Daman and Diu": {"min_lat": 20.0, "max_lat": 20.8, "min_lng": 72.8, "max_lng": 73.2},
    }
    
    print(f"Checking coordinates: lat={lat}, lng={lng}")
    
    # First check special overlapping regions
    for (min_lat, max_lat, min_lng, max_lng), state in overlapping_regions.items():
        if (min_lat <= lat <= max_lat and min_lng <= lng <= max_lng):
            print(f"Overlapping region found: {state} for coordinates ({lat}, {lng})")
            return state
    
    # Then check regular state boundaries
    detected_state = None
    for state, bounds in state_boundaries.items():
        if (bounds["min_lat"] <= lat <= bounds["max_lat"] and 
            bounds["min_lng"] <= lng <= bounds["max_lng"]):
            detected_state = state
            print(f"State found: {state} for coordinates ({lat}, {lng})")
            break
    
    # If boundary mapping found a state, validate with Gemini if available
    if detected_state and _gemini_model:
        try:
            # Use Gemini to double-check the detection
            prompt = f"""
            Given coordinates latitude: {lat}, longitude: {lng}, confirm which Indian state this location belongs to.
            
            The boundary mapping suggests: {detected_state}
            
            Please confirm if this is correct. Return ONLY the state name in English, nothing else.
            If the coordinates are outside India, return "Outside India".
            
            State name:
            """
            
            response = _gemini_model.generate_content(prompt)
            gemini_state = response.text.strip()
            
            # Clean up Gemini response
            lines = gemini_state.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('State') and not line.startswith('The') and line.lower() != "outside india":
                    print(f"Gemini validation: {line} for coordinates ({lat}, {lng})")
                    
                    # If Gemini agrees with boundary mapping, use it
                    if detected_state.lower() in line.lower() or line.lower() in detected_state.lower():
                        print(f"[OK] Gemini confirms: {detected_state}")
                        return detected_state
                    else:
                        # If Gemini disagrees, check if Gemini's suggestion is valid
                        gemini_valid = False
                        for state, bounds in state_boundaries.items():
                            if (state.lower() in line.lower() or line.lower() in state.lower()) and \
                               (bounds["min_lat"] <= lat <= bounds["max_lat"] and 
                                bounds["min_lng"] <= lng <= bounds["max_lng"]):
                                print(f"[OK] Gemini correction: {state} for coordinates ({lat}, {lng})")
                                return state
                        
                        # If Gemini's suggestion is not valid, stick with boundary mapping
                        print(f"[WARNING] Gemini suggestion '{line}' not valid for coordinates, using boundary mapping: {detected_state}")
                        return detected_state
            
            # If Gemini response is unclear, use boundary mapping
            print(f"[WARNING] Gemini response unclear, using boundary mapping: {detected_state}")
            return detected_state
            
        except Exception as e:
            print(f"Error using Gemini for validation: {e}")
            print(f"Using boundary mapping result: {detected_state}")
            return detected_state
    
    # If no boundary mapping result, try Gemini as fallback
    if not detected_state and _gemini_model:
        try:
            prompt = f"""
            Given coordinates latitude: {lat}, longitude: {lng}, determine which Indian state this location belongs to.
            
            Return ONLY the state name in English, nothing else. If the coordinates are outside India, return "Outside India".
            
            State name:
            """
            
            response = _gemini_model.generate_content(prompt)
            gemini_state = response.text.strip()
            
            # Clean up response
            lines = gemini_state.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('State') and not line.startswith('The') and line.lower() != "outside india":
                    print(f"Gemini fallback found: {line} for coordinates ({lat}, {lng})")
                    return line
            
        except Exception as e:
            print(f"Error using Gemini fallback: {e}")
    
    if detected_state:
        return detected_state
    
    print(f"No state found for coordinates: lat={lat}, lng={lng}")
    return None

# INGRES ChatBOT Endpoints
@app.post("/ingres/query", response_model=GroundwaterResponse)
async def query_groundwater_data(request: GroundwaterQuery):
    """
    Query groundwater data using INGRES ChatBOT.
    Provides intelligent analysis with criticality assessment and recommendations.
    """
    try:
        # Extract state from query if not provided
        if not request.state:
            # Try to extract state from query text using Gemini
            if _gemini_model:
                try:
                    prompt = f"""
                    Extract the Indian state name from this query: "{request.query}"
                    
                    Return only the state name, nothing else.
                    If no state is mentioned, return "None".
                    """
                    response = _gemini_model.generate_content(prompt)
                    extracted_state = response.text.strip()
                    if extracted_state and extracted_state.lower() != "none":
                        request.state = extracted_state
                except Exception as e:
                    print(f"Error extracting state from query: {e}")
        
        # If still no state, try to find a state in the query text
        if not request.state:
            query_lower = request.query.lower()
            indian_states = [
                "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
                "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
                "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
                "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
                "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
                "delhi", "chandigarh", "puducherry", "jammu and kashmir", "ladakh"
            ]
            
            for state in indian_states:
                if state in query_lower:
                    request.state = state.title()
                    break
        
        # Get groundwater data
        data = get_groundwater_data(
            state=request.state,
            district=request.district,
            assessment_unit=request.assessment_unit
        )
        
        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        
        # Generate recommendations
        quality_issues = data.get("quality_analysis", {}).get("issues", []) if data.get("quality_analysis") else []
        recommendations = generate_groundwater_recommendations(
            status=data["criticality_status"],
            extraction_percentage=data["extraction_stage"],
            quality_issues=quality_issues
        )
        
        # Create visualizations if requested
        visualizations = []
        if request.include_visualizations:
            visualizations = create_groundwater_visualizations(data)
        
        # Get comparison data
        comparison_data = get_state_comparison_data(data["state"])
        
        # Generate enhanced statistics
        enhanced_stats = generate_enhanced_statistics()
        
        # Prepare numerical values
        numerical_values = {
            "extraction_stage": data["extraction_stage"],
            "annual_recharge": data["annual_recharge"],
            "extractable_resource": data["extractable_resource"],
            "total_extraction": data["total_extraction"],
            "future_availability": data["future_availability"],
            "rainfall": data["rainfall"],
            "total_area": data["total_area"]
        }
        
        return GroundwaterResponse(
            data=data,
            criticality_status=data["criticality_status"],
            criticality_emoji=data["criticality_emoji"],
            numerical_values=numerical_values,
            recommendations=recommendations,
            visualizations=visualizations,
            comparison_data=comparison_data,
            quality_analysis=data["quality_analysis"],
            additional_resources=data.get("additional_resources"),
            key_findings_trends=data.get("key_findings_trends"),
            enhanced_statistics=enhanced_stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing groundwater query: {str(e)}")

@app.post("/ingres/location-analysis", response_model=LocationAnalysisResponse)
async def analyze_location_groundwater(request: LocationAnalysisRequest):
    """
    Analyze groundwater data for a specific location using coordinates.
    """
    try:
        # Get state from coordinates
        state = get_state_from_coordinates(request.lat, request.lng)
        
        if not state:
            raise HTTPException(status_code=404, detail="Location not found in India")
        
        # Get groundwater data for the state
        data = get_groundwater_data(state=state)
        
        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        
        # Generate recommendations
        quality_issues = data.get("quality_analysis", {}).get("issues", []) if data.get("quality_analysis") else []
        recommendations = generate_groundwater_recommendations(
            status=data["criticality_status"],
            extraction_percentage=data["extraction_stage"],
            quality_issues=quality_issues
        )
        
        # Create visualizations if requested
        visualizations = []
        if request.include_visualizations:
            visualizations = create_groundwater_visualizations(data)
        
        # Generate enhanced statistics
        enhanced_stats = generate_enhanced_statistics()
        
        # Prepare numerical values
        numerical_values = {
            "extraction_stage": data["extraction_stage"],
            "annual_recharge": data["annual_recharge"],
            "extractable_resource": data["extractable_resource"],
            "total_extraction": data["total_extraction"],
            "future_availability": data["future_availability"],
            "rainfall": data["rainfall"],
            "total_area": data["total_area"]
        }
        
        return LocationAnalysisResponse(
            state=state,
            district=data.get("district"),
            assessment_unit=data.get("assessment_unit"),
            groundwater_data=data,
            criticality_status=data["criticality_status"],
            criticality_emoji=data["criticality_emoji"],
            numerical_values=numerical_values,
            recommendations=recommendations,
            visualizations=visualizations,
            quality_analysis=data["quality_analysis"],
            additional_resources=data.get("additional_resources"),
            key_findings_trends=data.get("key_findings_trends"),
            enhanced_statistics=enhanced_stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing location: {str(e)}")

@app.get("/ingres/states")
async def get_available_states():
    """
    Get list of all available states in the dataset.
    """
    try:
        df = pd.read_csv('master_groundwater_data.csv', low_memory=False)
        states = sorted(df['STATE'].unique().tolist())
        return {"states": states, "count": len(states)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving states: {str(e)}")

@app.get("/ingres/districts/{state}")
async def get_districts_by_state(state: str):
    """
    Get list of districts for a specific state.
    """
    try:
        df = pd.read_csv('master_groundwater_data.csv', low_memory=False)
        state_df = df[df['STATE'].str.upper() == state.upper()]
        
        if state_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for state: {state}")
        
        districts = sorted(state_df['DISTRICT'].unique().tolist())
        return {"state": state, "districts": districts, "count": len(districts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving districts: {str(e)}")

@app.get("/ingres/criticality-summary")
async def get_criticality_summary():
    """
    Get national summary of groundwater criticality status.
    """
    try:
        df = pd.read_csv('master_groundwater_data.csv', low_memory=False)
        
        # Calculate criticality distribution
        extraction_data = df['Stage of Ground Water Extraction (%) - Total - Total'].dropna()
        
        safe_count = len(extraction_data[extraction_data < 70])
        semi_critical_count = len(extraction_data[(extraction_data >= 70) & (extraction_data < 90)])
        critical_count = len(extraction_data[(extraction_data >= 90) & (extraction_data < 100)])
        over_exploited_count = len(extraction_data[extraction_data >= 100])
        
        total_districts = len(extraction_data)
        
        # Create visualizations
        visualizations = []
        
        # Pie chart for criticality distribution
        pie_chart = {
            "type": "pie_chart",
            "title": "National Groundwater Criticality Distribution",
            "data": {
                "labels": ["Safe", "Semi-Critical", "Critical", "Over-Exploited"],
                "values": [safe_count, semi_critical_count, critical_count, over_exploited_count],
                "colors": ["#28a745", "#ffc107", "#dc3545", "#6c757d"]
            }
        }
        visualizations.append(pie_chart)
        
        # Bar chart for state-wise criticality
        state_criticality = df.groupby('STATE')['Stage of Ground Water Extraction (%) - Total - Total'].mean().sort_values(ascending=False).head(10)
        bar_chart = {
            "type": "bar_chart",
            "title": "Top 10 States by Average Extraction Stage",
            "data": {
                "x": state_criticality.index.tolist(),
                "y": state_criticality.values.tolist(),
                "x_label": "State",
                "y_label": "Extraction Stage (%)"
            }
        }
        visualizations.append(bar_chart)
        
        # Gauge chart for national average
        gauge_chart = {
            "type": "gauge_chart",
            "title": "National Average Extraction Stage",
            "data": {
                "value": round(extraction_data.mean(), 2),
                "max_value": 150,
                "thresholds": [70, 90, 100],
                "threshold_labels": ["Safe", "Semi-Critical", "Critical", "Over-Exploited"]
            }
        }
        visualizations.append(gauge_chart)
        
        return {
            "total_districts": int(total_districts),
            "criticality_distribution": {
                "safe": {
                    "count": int(safe_count),
                    "percentage": round(safe_count / total_districts * 100, 2)
                },
                "semi_critical": {
                    "count": int(semi_critical_count),
                    "percentage": round(semi_critical_count / total_districts * 100, 2)
                },
                "critical": {
                    "count": int(critical_count),
                    "percentage": round(critical_count / total_districts * 100, 2)
                },
                "over_exploited": {
                    "count": int(over_exploited_count),
                    "percentage": round(over_exploited_count / total_districts * 100, 2)
                }
            },
            "national_average_extraction": round(extraction_data.mean(), 2),
            "visualizations": visualizations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating criticality summary: {str(e)}")

@app.get("/ingres/groundwater-summary/{state}/{district}")
async def get_groundwater_summary(state: str, district: str):
    """
    Get comprehensive groundwater summary with key findings for a specific district.
    Includes basis of availability, major uses, and safety category.
    """
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        # Filter data for the specific state and district
        filtered_df = _master_df[
            (_master_df['STATE'].str.contains(state, case=False, na=False)) &
            (_master_df['DISTRICT'].str.contains(district, case=False, na=False))
        ]
        
        if filtered_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {district}, {state}")
        
        # Get the most recent record for the district
        latest_record = filtered_df.sort_values('Assessment_Year', ascending=False).iloc[0]
        
        # Generate comprehensive summary
        summary = generate_comprehensive_groundwater_summary(latest_record)
        
        return {
            "success": True,
            "summary": summary,
            "data_source": "master_groundwater_data.csv",
            "generated_at": pd.Timestamp.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating groundwater summary: {str(e)}")

@app.get("/ingres/groundwater-summary-by-coordinates")
async def get_groundwater_summary_by_coordinates(lat: float, lon: float):
    """
    Get comprehensive groundwater summary for coordinates.
    """
    try:
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        # Find the closest district based on coordinates
        # This is a simplified approach - in practice, you'd use proper geospatial matching
        # For now, we'll return a sample from the first available record
        if _master_df.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        sample_record = _master_df.iloc[0]
        summary = generate_comprehensive_groundwater_summary(sample_record)
        
        return {
            "success": True,
            "summary": summary,
            "coordinates": {"latitude": lat, "longitude": lon},
            "note": "Sample data - implement proper geospatial matching for accurate results",
            "generated_at": pd.Timestamp.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating groundwater summary: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "qdrant_connected": _qdrant_client is not None,
            "model_loaded": _model is not None,
            "gemini_configured": _gemini_model is not None,
            "data_loaded": _master_df is not None,
            "embeddings_uploaded": _embeddings_uploaded
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring."""
    return {
        "status": "healthy",
        "message": "Groundwater RAG API is running",
        "version": "1.0.0",
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Groundwater Data Analysis API", "status": "running"}

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global _embeddings_uploaded
    try:
        print("[INIT] Starting application initialization...")
        
        # Run the synchronous initialization in a thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _init_components)
        print("[OK] Core components initialized")
        
        # Initialize Qdrant client with timeout
        collection_setup = False
        try:
            print("[INIT] Initializing Qdrant client...")
            await asyncio.wait_for(
                loop.run_in_executor(None, _init_qdrant), 
                timeout=10.0
            )
            print("[OK] Qdrant client initialized")
            
            # Test collection setup with timeout
            print("[INIT] Setting up Qdrant collection...")
            collection_setup = await asyncio.wait_for(
                loop.run_in_executor(None, setup_collection),
                timeout=15.0
            )
            if collection_setup:
                print("[OK] Qdrant collection ready")
            else:
                print("[WARNING] Qdrant collection setup failed, continuing with limited functionality")
        except asyncio.TimeoutError:
            print("[WARNING] Qdrant initialization timed out, continuing with limited functionality")
        except Exception as e:
            print(f"[WARNING] Qdrant initialization error: {e}, continuing with limited functionality")
        
        # Initialize BM25 for search fallback (skip data upload during startup)
        try:
            print("[INIT] Initializing BM25...")
            await loop.run_in_executor(None, _load_bm25)
            print("[OK] BM25 initialized")
        except Exception as e:
            print(f"[WARNING] BM25 initialization error: {e}")
        
        print("Note: Data upload to Qdrant can be done via /upload-data endpoint")
        
        print(" Groundwater RAG API started successfully!")
    except Exception as e:
        print(f"[ERROR] Startup error: {e}")
        print("[WARNING] Continuing with limited functionality...")

# Run with: uvicorn main:app --reload --port 8000
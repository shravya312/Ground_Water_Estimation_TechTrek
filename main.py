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
from typing import List, Dict, Optional
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
COLLECTION_NAME = "groundwater_excel_collection"
VECTOR_SIZE = 384
MIN_SIMILARITY_SCORE = 0.5

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
    
    try:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        
        torch.set_default_dtype(torch.float32)
        
        _model = SentenceTransformer(
            "all-MiniLM-L6-v2",
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
                "all-MiniLM-L6-v2",
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
                _model = SentenceTransformer("all-MiniLM-L6-v2")
                _model.eval()
                return True
                
            except Exception as e3:
                print(f"All initialization methods failed. Last error: {str(e3)}")
                _model = None
                return False

def _init_components():
    global _qdrant_client, _model, _nlp, _gemini_model, _master_df, _translator_model, _translator_tokenizer, _indic_processor
    if _qdrant_client is None:
        try:
            print(f"🔄 Connecting to Qdrant at {QDRANT_URL}...")
            _qdrant_client = QdrantClient(
                url=QDRANT_URL, 
                api_key=QDRANT_API_KEY if QDRANT_API_KEY else None, 
                timeout=30,
                prefer_grpc=False  # Use HTTP instead of gRPC for better compatibility
            )
            # Test the connection
            _qdrant_client.get_collections()
            print("✅ Qdrant connection established")
        except Exception as e:
            print(f"❌ Failed to initialize Qdrant client: {str(e)}")
            print("⚠️ Continuing without Qdrant - some features will be limited")
            _qdrant_client = None
    if _model is None:
        if not initialize_sentence_transformer():
            print("Warning: Dense embeddings are disabled (SentenceTransformer failed). Falling back to BM25-only search.")
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"Warning: Failed to initialize spaCy NLP model: {str(e)}. Some features may be limited.")
            _nlp = None
    if _gemini_model is None and GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini API: {str(e)}")
    # Skip translator model loading for now to speed up startup
    # This can be loaded later when needed
    if False:  # Disabled for faster startup
        try:
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
            print(f"Warning: Failed to initialize IndicTrans2 translator: {e}")
    if _master_df is None:
        try:
            _master_df = pd.read_csv("master_groundwater_data.csv", low_memory=False)
            _master_df['STATE'] = _master_df['STATE'].fillna('').astype(str)
            _master_df['DISTRICT'] = _master_df['DISTRICT'].fillna('').astype(str)
            _master_df['ASSESSMENT UNIT'] = _master_df['ASSESSMENT UNIT'].fillna('').astype(str)
            _master_df['combined_text'] = _master_df.apply(create_detailed_combined_text, axis=1)
        except FileNotFoundError:
            raise Exception("Error: master_groundwater_data.csv not found. Please run excel_ingestor.py first.")
        except Exception as e:
            raise Exception(f"Error loading master groundwater data: {str(e)}")

def create_detailed_combined_text(row):
    """Generates a detailed combined text string for a DataFrame row."""
    parts = []
    for col, value in row.items():
        if pd.notna(value) and value != '' and col not in ['S.No']:
            parts.append(f"{col}: {value}")
    return " | ".join(parts)

def tokenize_text(text):
    """Tokenize text for BM25 processing."""
    return text.lower().split()

def get_embeddings(texts):
    """Convert text into embeddings"""
    try:
        if _model is None:
            return None
        
        try:
            test_text = texts[0] if texts else "test"
            test_embedding = _model.encode([test_text], show_progress_bar=False)
            return _model.encode(texts, show_progress_bar=False)
            
        except Exception as meta_error:
            if "meta tensor" in str(meta_error).lower():
                print("Meta tensor issue detected. Attempting to reinitialize model...")
                
                try:
                    _model = None
                    _model = SentenceTransformer(
                        "all-MiniLM-L6-v2",
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
            print("⚠️ Qdrant client not available, skipping collection setup")
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
            print(f"Created new collection: {COLLECTION_NAME}")
        else:
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
        
        print(f"✅ All {total_uploaded} Excel rows uploaded to Qdrant.")
        return True
    except Exception as e:
        print(f"Error uploading Excel data to Qdrant: {str(e)}")
        return False

def _load_bm25():
    global _bm25_model, _all_chunks, _bm25_df
    if _bm25_model is not None:
        return
    try:
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
        else:
            _all_chunks = _master_df['combined_text'].tolist()
            _bm25_df = _master_df.copy()
        if _all_chunks:
            tokenized_chunks = [tokenize_text(chunk) for chunk in _all_chunks]
            _bm25_model = BM25Okapi(tokenized_chunks)
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

def search_excel_chunks(query_text, year=None, target_state=None, target_district=None, extracted_parameters=None):
    """Retrieve most relevant Excel data rows using hybrid search, with optional year and location filtering."""
    _init_components()
    _load_bm25()
    qdrant_filter_conditions = []

    if year:
        qdrant_filter_conditions.append(
            FieldCondition(
                key="Assessment_Year",
                match=MatchValue(value=year)
            )
        )
    
    if target_state:
        qdrant_filter_conditions.append(
            FieldCondition(
                key="STATE",
                match=MatchValue(value=target_state)
            )
        )
    
    if target_district:
        qdrant_filter_conditions.append(
            FieldCondition(
                key="DISTRICT",
                match=MatchValue(value=target_district)
            )
        )

    if extracted_parameters:
        for param_type, value in extracted_parameters.items():
            qdrant_filter_conditions.append(
                FieldCondition(
                    key="text",
                    match=MatchValue(value=str(param_type).lower())
                )
            )
    
    qdrant_filter = Filter(must=qdrant_filter_conditions) if qdrant_filter_conditions else None

    try:
        # Dense Retrieval (Qdrant)
        dense_hits = {}
        dense_payloads = {}
        if _model is not None:
            query_vector = _model.encode([query_text])[0]
            qdrant_results = _qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=20,
                with_payload=True
            )
            dense_hits = {hit.payload.get("text", ""): hit.score for hit in qdrant_results}
            dense_payloads = {hit.payload.get("text", ""): hit.payload for hit in qdrant_results}

        # Sparse Retrieval (BM25)
        sparse_hits = {}
        if _bm25_model and _all_chunks and _bm25_df is not None:
            tokenized_query = tokenize_text(query_text)
            bm25_scores = _bm25_model.get_scores(tokenized_query)
            
            for i, score in enumerate(bm25_scores):
                if score > 0:
                    chunk_text_bm25 = _all_chunks[i]
                    if (year and _bm25_df.iloc[i]['Assessment_Year'] != year) or \
                       (target_state and _bm25_df.iloc[i]['STATE'] != target_state) or \
                       (target_district and _bm25_df.iloc[i]['DISTRICT'] != target_district):
                            continue
                    sparse_hits[chunk_text_bm25] = score
        
        # Hybrid Scoring
        combined_scores = {}
        alpha = 0.5  # Can be made configurable

        all_candidate_texts = set(dense_hits.keys()).union(set(sparse_hits.keys()))

        if not all_candidate_texts:
            return []
        
        max_dense_score = max(dense_hits.values()) if dense_hits else 1.0
        max_sparse_score = max(sparse_hits.values()) if sparse_hits else 1.0

        for text_chunk in all_candidate_texts:
            dense_score = dense_hits.get(text_chunk, 0.0) / max_dense_score
            sparse_score = sparse_hits.get(text_chunk, 0.0) / max_sparse_score
            combined_scores[text_chunk] = (alpha * dense_score) + ((1 - alpha) * sparse_score)
        
        sorted_chunks_with_scores = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Retrieve original payloads for the top re-ranked chunks
        results_with_payloads = []
        for chunk_text, score in sorted_chunks_with_scores[:20]:
            if chunk_text in dense_payloads:
                payload = dense_payloads[chunk_text]
            else:
                matching_rows = _bm25_df[_bm25_df['combined_text'] == chunk_text]
                if not matching_rows.empty:
                    payload = matching_rows.iloc[0].to_dict()
                else:
                    payload = {"text": chunk_text}
            results_with_payloads.append({"score": score, "data": payload})

        return results_with_payloads

    except Exception as e:
        print(f"Error performing hybrid search: {str(e)}")
        return []

def re_rank_chunks(query_text, candidate_results, top_k=5):
    """Re-ranks candidate results based on semantic similarity to the query."""
    if not candidate_results:
        return []

    if _model is None:
        sorted_candidates = sorted(candidate_results, key=lambda r: r.get('score', 0), reverse=True)
        return sorted_candidates[:top_k]

    candidate_texts = [res['data'].get('text', '') for res in candidate_results]
    if not candidate_texts:
        return []

    query_embedding = _model.encode([query_text])[0]
    chunk_embeddings = _model.encode(candidate_texts)

    query_embedding_norm = np.linalg.norm(query_embedding)
    chunk_embeddings_norm = np.linalg.norm(chunk_embeddings, axis=1)

    if query_embedding_norm == 0:
        return []
    chunk_embeddings_norm[chunk_embeddings_norm == 0] = 1e-12

    similarities = np.dot(chunk_embeddings, query_embedding) / (chunk_embeddings_norm * query_embedding_norm)

    re_ranked_scores = []
    for i, res in enumerate(candidate_results):
        re_ranked_scores.append((res['data'], similarities[i]))

    re_ranked_scores.sort(key=lambda x: x[1], reverse=True)

    final_results = []
    for data, score in re_ranked_scores[:top_k]:
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

def generate_answer_from_gemini(query, context_data, year=None, target_state=None, target_district=None, chat_history=None, extracted_parameters=None, user_language='en'):
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
        f"""Here are the rules for data presentation:
- If a specific year is provided, give data for that year.
- If no specific year is provided, summarize the data including averages across all available years for the specified location (state or district).
- ALWAYS include ALL relevant column data in your response - don't just mention a few key metrics.
- Present data in a structured, formatted manner with clear headings, subheadings, and organized sections.
- Use PROPER MARKDOWN formatting for better readability (headers, bullet points, tables, etc.).
- Include both numerical values and their units (ham, ha, mm, %).
- For each data point, explain what it represents and its significance.
- Organize data into logical categories: Rainfall Data, Geographical Area, Groundwater Recharge, Extraction Data, etc.
- Use PROPER MARKDOWN TABLES for numerical data - format like this:
  | Parameter | Cultivated (C) (ham) | Non-Cultivated (NC) (ham) | Perennial (PQ) (ham) | Total (ham) |
  |-----------|---------------------|---------------------------|---------------------|-------------|
  | Rainfall Recharge | 15000.50 | 12000.25 | 0.0 | 27000.75 |
- Highlight key findings and trends.
- Do NOT ask follow-up questions about what aspect of estimation the user is interested in. Provide a comprehensive summary of ALL available relevant metrics.
- If data is missing for certain columns, mention that explicitly.
- Format the output like a professional groundwater assessment report with proper markdown tables.
- NEVER use pipe characters (|) or hyphens (-) as text - only use them for markdown table formatting.
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
        return f"❌ Error from Gemini: {str(e)}"

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

def answer_query(query: str, user_language: str = 'en', user_id: str = None) -> str:
    query = (query or '').strip()
    if not query:
        return "Please provide a question."
    try:
        _init_components()
    except Exception as e:
        return f"Initialization error: {str(e)}"
    
    # Detect and translate query to English for processing
    original_query = query
    translated_query, detected_lang = translate_query_to_english(query)
    
    year = None
    year_match = re.search(r'\b(19|20)\d{2}\b', translated_query)
    if year_match:
        year = int(year_match.group(0))
    
    target_state = None
    target_district = None
    if _master_df is not None:
        unique_states = _master_df['STATE'].unique().tolist()
        unique_districts = _master_df['DISTRICT'].unique().tolist()
        
        # Try to find state with fuzzy matching
        for state in unique_states:
            if pd.notna(state):
                # Exact match
                if re.search(r'\b' + re.escape(str(state)) + r'\b', translated_query, re.IGNORECASE):
                    target_state = state
                    break
                # Partial match
                elif str(state).lower() in translated_query.lower():
                    target_state = state
                    break

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
    
    expanded_terms = expand_query(translated_query)
    expanded_query_text = f"{translated_query} {expanded_terms}".strip()
    
    candidate_results = search_excel_chunks(expanded_query_text, year=year, target_state=target_state, target_district=target_district, extracted_parameters=extracted_parameters)
    
    # If no results found with location filters, try without location filters
    if not candidate_results:
        candidate_results = search_excel_chunks(expanded_query_text, year=year, target_state=None, target_district=None, extracted_parameters=extracted_parameters)
    
    # If still no results, try with just the basic query without expansion
    if not candidate_results:
        candidate_results = search_excel_chunks(translated_query, year=year, target_state=None, target_district=None, extracted_parameters=extracted_parameters)
    
    # If still no results, try with original query (before translation)
    if not candidate_results:
        candidate_results = search_excel_chunks(original_query, year=year, target_state=None, target_district=None, extracted_parameters=extracted_parameters)
    
    # If still no results, try with common groundwater keywords
    if not candidate_results:
        groundwater_query = "groundwater estimation data analysis"
        candidate_results = search_excel_chunks(groundwater_query, year=year, target_state=None, target_district=None, extracted_parameters=extracted_parameters)
    
    if not candidate_results:
        return "I couldn't find enough relevant information in the groundwater data to answer your question."
    
    re_ranked_results = re_rank_chunks(expanded_query_text, candidate_results, top_k=5)
    if not re_ranked_results:
        return "I couldn't find enough relevant information in the groundwater data to answer your question."
    
    context_data = [res['data'] for res in re_ranked_results]
    
    # Load chat history if user_id provided
    chat_history = None
    if user_id:
        chat_history = load_chat_history(user_id)
    
    answer = generate_answer_from_gemini(
        translated_query, 
        context_data, 
        year=year, 
        target_state=target_state, 
        target_district=target_district, 
        chat_history=chat_history, 
        extracted_parameters=extracted_parameters,
        user_language=user_language
    )
    
    # Translate answer back to user's language if needed
    if user_language != 'en':
        answer = translate_answer_to_language(answer, user_language)
    
    return answer

# --- Visualization Functions ---
def create_groundwater_overview_dashboard(df):
    """Create a comprehensive overview dashboard of groundwater data."""
    if df is None or df.empty:
        return None
    
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

def create_state_analysis_plots(df, selected_state=None):
    """Create detailed analysis plots for a specific state or all states."""
    if df is None or df.empty:
        return None
    
    # Filter data for selected state if provided
    if selected_state:
        state_df = df[df['STATE'] == selected_state]
        title_suffix = f" - {selected_state}"
    else:
        state_df = df
        title_suffix = " - All States"
    
    if state_df.empty:
        return None
    
    # Create subplots
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
        ]
    )
    
    # 1. Groundwater Extraction by District
    if 'Ground Water Extraction for all uses (ha.m) - Total - Total' in state_df.columns:
        extraction_col = 'Ground Water Extraction for all uses (ha.m) - Total - Total'
        district_extraction = state_df.groupby('DISTRICT')[extraction_col].sum().sort_values(ascending=False).head(10)
        
        fig.add_trace(
            go.Bar(
                x=district_extraction.index,
                y=district_extraction.values,
                name="Extraction by District",
                marker_color='skyblue'
            ),
            row=1, col=1
        )
    
    # 2. Rainfall Distribution
    if 'Rainfall (mm) - Total - Total' in state_df.columns:
        rainfall_col = 'Rainfall (mm) - Total - Total'
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
    if ('Annual Ground water Recharge (ham) - Total - Total' in state_df.columns and 
        'Ground Water Extraction for all uses (ha.m) - Total - Total' in state_df.columns):
        recharge_col = 'Annual Ground water Recharge (ham) - Total - Total'
        extraction_col = 'Ground Water Extraction for all uses (ha.m) - Total - Total'
        
        scatter_data = state_df[[recharge_col, extraction_col]].dropna()
        
        fig.add_trace(
            go.Scatter(
                x=scatter_data[recharge_col],
                y=scatter_data[extraction_col],
                mode='markers',
                name="Recharge vs Extraction",
                marker=dict(
                    color=scatter_data[recharge_col],
                    size=8,
                    opacity=0.7,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=scatter_data.index
            ),
            row=2, col=1
        )
    
    # 4. Quality Tagging Analysis
    if 'Quality Tagging' in state_df.columns:
        quality_data = state_df['Quality Tagging'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=quality_data.index,
                values=quality_data.values,
                name="Quality Distribution"
            ),
            row=2, col=2
        )
    
    # Update layout with white text for dark theme
    fig.update_layout(
        height=800,
        title_text=f"Detailed State Analysis{title_suffix}",
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

def create_temporal_analysis_plots(df):
    """Create temporal analysis plots showing trends over time."""
    if df is None or df.empty:
        return None
    
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
    """Create a geographical heatmap of groundwater metrics by state."""
    if df is None or df.empty or metric not in df.columns:
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
    """Create a correlation matrix heatmap of numerical groundwater parameters."""
    if df is None or df.empty:
        return None
    
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
    """Create statistical summary plots including distribution and box plots."""
    if df is None or df.empty:
        return None
    
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
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        fig = create_state_analysis_plots(_master_df, state)
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
        _init_components()
        if _master_df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        available_states = sorted([s for s in _master_df['STATE'].unique() if pd.notna(s)])
        return {"states": available_states, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask(request: AskRequest):
    """Main endpoint for asking questions about groundwater data."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    try:
        user_lang = request.language or detect_language(query)
        answer = answer_query(query, user_lang, request.user_id)
        return {
            "answer": answer, 
            "detected_lang": detect_language(query),
            "selected_lang": user_lang,
            "query": query
        }
    except Exception as exc:
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
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    try:
        user_lang = request.language or detect_language(query)
        answer = answer_query(query, user_lang, request.user_id)
        
        # Add additional formatting instructions for better structure
        formatted_answer = f"""
# 💧 Groundwater Data Analysis Report

## Query
**Question:** {query}

## Analysis
{answer}

---
*Report generated by Groundwater RAG API - Multilingual Support*  
*Language: {SUPPORTED_LANGUAGES.get(user_lang, user_lang)}*
        """
        
        return {
            "answer": formatted_answer.strip(), 
            "detected_lang": detect_language(query),
            "selected_lang": user_lang,
            "query": query,
            "formatted": True
        }
    except Exception as exc:
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

def get_state_from_coordinates(lat: float, lng: float) -> str:
    """Convert coordinates to state name using Gemini API with fallback to boundary mapping."""
    # Try Gemini first if available
    if _gemini_model:
        try:
            prompt = f"""
            Given the coordinates latitude: {lat}, longitude: {lng}, determine which Indian state this location belongs to.
            
            Return ONLY the state name in English, nothing else. If the coordinates are outside India, return "Outside India".
            
            Common Indian states include: Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, Chhattisgarh, Goa, Gujarat, Haryana, Himachal Pradesh, Jharkhand, Karnataka, Kerala, Madhya Pradesh, Maharashtra, Manipur, Meghalaya, Mizoram, Nagaland, Odisha, Punjab, Rajasthan, Sikkim, Tamil Nadu, Telangana, Tripura, Uttar Pradesh, Uttarakhand, West Bengal, Delhi, Jammu and Kashmir, Ladakh, Andaman and Nicobar Islands, Chandigarh, Dadra and Nagar Haveli and Daman and Diu, Lakshadweep, Puducherry.
            
            State name:
            """
            
            response = _gemini_model.generate_content(prompt)
            state_name = response.text.strip()
            
            # Clean up the response
            if "Outside India" in state_name or "outside" in state_name.lower():
                return None
            
            # Remove any extra text and return just the state name
            lines = state_name.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('State') and not line.startswith('The'):
                    print(f"Gemini found state: {line} for coordinates ({lat}, {lng})")
                    return line
            
            return state_name
        except Exception as e:
            print(f"Error using Gemini for state detection: {e}")
    
    # Fallback to boundary mapping
    state_boundaries = {
        # Major states first to avoid conflicts
        "Maharashtra": {"min_lat": 15.6, "max_lat": 22.0, "min_lng": 72.6, "max_lng": 80.9},
        "Karnataka": {"min_lat": 11.7, "max_lat": 18.5, "min_lng": 74.1, "max_lng": 78.6},
        "Gujarat": {"min_lat": 20.1, "max_lat": 24.7, "min_lng": 68.2, "max_lng": 74.5},
        "Rajasthan": {"min_lat": 23.1, "max_lat": 30.2, "min_lng": 69.3, "max_lng": 78.2},
        "Madhya Pradesh": {"min_lat": 21.1, "max_lat": 26.9, "min_lng": 74.0, "max_lng": 82.8},
        "Uttar Pradesh": {"min_lat": 23.7, "max_lat": 31.1, "min_lng": 77.0, "max_lng": 84.7},
        "Bihar": {"min_lat": 24.2, "max_lat": 27.7, "min_lng": 83.3, "max_lng": 88.8},
        "West Bengal": {"min_lat": 21.5, "max_lat": 27.2, "min_lng": 85.5, "max_lng": 89.9},
        "Odisha": {"min_lat": 17.5, "max_lat": 22.5, "min_lng": 81.3, "max_lng": 87.3},
        "Chhattisgarh": {"min_lat": 17.8, "max_lat": 24.1, "min_lng": 80.2, "max_lng": 84.4},
        "Jharkhand": {"min_lat": 21.8, "max_lat": 25.3, "min_lng": 83.2, "max_lng": 87.9},
        "Andhra Pradesh": {"min_lat": 12.4, "max_lat": 19.9, "min_lng": 76.8, "max_lng": 84.8},
        "Telangana": {"min_lat": 15.5, "max_lat": 19.9, "min_lng": 77.2, "max_lng": 81.1},
        "Tamil Nadu": {"min_lat": 8.1, "max_lat": 13.1, "min_lng": 76.2, "max_lng": 80.3},
        "Kerala": {"min_lat": 8.1, "max_lat": 12.8, "min_lng": 74.9, "max_lng": 77.4},
        "Goa": {"min_lat": 14.8, "max_lat": 15.8, "min_lng": 73.7, "max_lng": 74.2},
        "Haryana": {"min_lat": 28.4, "max_lat": 31.0, "min_lng": 74.4, "max_lng": 77.5},
        "Punjab": {"min_lat": 29.5, "max_lat": 32.3, "min_lng": 73.9, "max_lng": 76.9},
        "Himachal Pradesh": {"min_lat": 30.4, "max_lat": 33.2, "min_lng": 75.6, "max_lng": 79.1},
        "Uttarakhand": {"min_lat": 28.7, "max_lat": 31.5, "min_lng": 77.3, "max_lng": 81.1},
        "Delhi": {"min_lat": 28.4, "max_lat": 28.9, "min_lng": 76.8, "max_lng": 77.3},
        # Northeastern states
        "Assam": {"min_lat": 24.1, "max_lat": 28.2, "min_lng": 89.7, "max_lng": 96.0},
        "Arunachal Pradesh": {"min_lat": 26.5, "max_lat": 29.4, "min_lng": 91.6, "max_lng": 97.4},
        "Manipur": {"min_lat": 23.8, "max_lat": 25.7, "min_lng": 93.0, "max_lng": 94.8},
        "Meghalaya": {"min_lat": 25.1, "max_lat": 26.1, "min_lng": 89.8, "max_lng": 92.8},
        "Mizoram": {"min_lat": 21.9, "max_lat": 24.5, "min_lng": 92.2, "max_lng": 93.3},
        "Nagaland": {"min_lat": 25.2, "max_lat": 27.0, "min_lng": 93.0, "max_lng": 95.4},
        "Tripura": {"min_lat": 22.9, "max_lat": 24.7, "min_lng": 91.2, "max_lng": 92.3},
        "Sikkim": {"min_lat": 27.0, "max_lat": 28.2, "min_lng": 88.0, "max_lng": 88.9}
    }
    
    # Debug logging
    print(f"Checking coordinates: lat={lat}, lng={lng}")
    
    for state, bounds in state_boundaries.items():
        if (bounds["min_lat"] <= lat <= bounds["max_lat"] and 
            bounds["min_lng"] <= lng <= bounds["max_lng"]):
            print(f"Found state: {state} for coordinates ({lat}, {lng})")
            return state
    
    print(f"No state found for coordinates: lat={lat}, lng={lng}")
    return None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        _init_components()
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

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global _embeddings_uploaded
    try:
        print("🔄 Starting application initialization...")
        
        # Run the synchronous initialization in a thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _init_components)
        print("✅ Core components initialized")
        
        # Initialize collection with timeout
        try:
            collection_setup = await loop.run_in_executor(None, setup_collection)
            if collection_setup:
                print("✅ Qdrant collection ready")
            else:
                print("⚠️ Qdrant collection setup failed, continuing with limited functionality")
        except Exception as e:
            print(f"⚠️ Qdrant collection error: {e}, continuing with limited functionality")
            collection_setup = False
        
        # Initialize BM25 and embeddings
        if collection_setup and not _embeddings_uploaded:
            try:
                if check_excel_embeddings_exist():
                    await loop.run_in_executor(None, _load_bm25)
                    _embeddings_uploaded = True
                    print("✅ Excel data embeddings loaded and BM25 initialized.")
                else:
                    if _master_df is not None:
                        print("⏳ Uploading Excel data to Qdrant...")
                        upload_success = await loop.run_in_executor(None, upload_excel_to_qdrant, _master_df)
                        if upload_success:
                            await loop.run_in_executor(None, _load_bm25)
                            _embeddings_uploaded = True
                            print("✅ Excel data processed and indexed.")
                        else:
                            print("❌ Failed to upload Excel data embeddings to Qdrant.")
            except Exception as e:
                print(f"⚠️ Embedding initialization error: {e}")
        elif _bm25_model is None:
            try:
                print("📚 Embeddings previously uploaded. Initializing BM25 from existing data...")
                await loop.run_in_executor(None, _load_bm25)
                print("✅ BM25 initialized from existing data")
            except Exception as e:
                print(f"⚠️ BM25 initialization error: {e}")
        
        print("🚀 Groundwater RAG API started successfully!")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        print("⚠️ Continuing with limited functionality...")

# Run with: uvicorn main:app --reload --port 8000
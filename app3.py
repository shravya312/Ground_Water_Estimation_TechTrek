import os
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PayloadSchemaType, CreateFieldIndex
from qdrant_client.models import PointStruct 
from sentence_transformers import SentenceTransformer 
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import re
import hashlib
import uuid
from rank_bm25 import BM25Okapi
import pandas as pd
import spacy
import torch
import json
from typing import List, Dict, Optional
from langdetect import detect_langs, LangDetectException
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
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

# --- Initialize Clients and Models ---
# Qdrant Client
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )
except Exception as e:
    st.error(f"Failed to initialize Qdrant client: {str(e)}")
    st.stop()

def _fix_meta_tensors(model):
    """Fix meta tensors by converting them to real tensors."""
    try:
        # Method 1: Fix parameters
        for name, param in model.named_parameters():
            if hasattr(param, 'is_meta') and param.is_meta:
                # Create a new parameter with the same shape and dtype
                new_param = torch.nn.Parameter(torch.zeros_like(param, device='cpu'))
                # Replace the meta parameter
                param.data = new_param.data
        
        # Method 2: Fix buffers
        for name, buffer in model.named_buffers():
            if hasattr(buffer, 'is_meta') and buffer.is_meta:
                # Create a new buffer with the same shape and dtype
                new_buffer = torch.zeros_like(buffer, device='cpu')
                # Replace the meta buffer
                buffer.data = new_buffer.data
        
        # Method 3: Force model to CPU and reinitialize if needed
        try:
            model = model.to('cpu')
        except Exception:
            # If moving to CPU fails, try to reinitialize the model
            pass
            
        return True
    except Exception as e:
        print(f"Meta tensor fix failed: {e}")
        return False

def initialize_sentence_transformer():
    """Robust initialization of SentenceTransformer with meta tensor handling."""
    global model
    
    # Method 1: Standard initialization
    try:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        
        torch.set_default_dtype(torch.float32)
        
        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu",
            model_kwargs={
                "low_cpu_mem_usage": False,
                "torch_dtype": torch.float32,
                "use_safetensors": False,
            },
        )
        model.eval()
        return True
        
    except Exception as e1:
        st.warning(f"Method 1 failed: {str(e1)}")
        
        # Method 2: Alternative initialization
        try:
            model = None
            model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu",
                model_kwargs={
                    "torch_dtype": torch.float32,
                    "trust_remote_code": False,
                },
            )
            model = model.to("cpu")
            model.eval()
            return True
            
        except Exception as e2:
            st.warning(f"Method 2 failed: {str(e2)}")
            
            # Method 3: Minimal initialization
            try:
                model = None
                model = SentenceTransformer("all-MiniLM-L6-v2")
                model.eval()
                return True
                
            except Exception as e3:
                st.error(f"All initialization methods failed. Last error: {str(e3)}")
                model = None
                return False

# Sentence Transformer for Embeddings (with graceful fallback)
model = None
if not initialize_sentence_transformer():
    st.warning(
        "Dense embeddings are disabled (SentenceTransformer failed). Falling back to BM25-only search.\n" 
        "The system will still work with sparse (BM25) search only."
    )

# spaCy NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    st.error(f"Failed to initialize spaCy NLP model: {str(e)}")
    st.stop()

# Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Failed to initialize Gemini API: {str(e)}")
    st.stop()

# --- Constants ---
COLLECTION_NAME = "groundwater_excel_collection"
VECTOR_SIZE = 384
MIN_SIMILARITY_SCORE = 0.5

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

# --- Global DataFrames ---
master_df = None

# --- Translation Functions ---
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
        # Confidence threshold to avoid misclassification on short texts
        if getattr(top, 'prob', 1.0) < 0.50:
            return "en"
        # Normalize some composite codes if needed
        return top.lang
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
            st.warning(f"googletrans translation error ({source_lang}->{target_lang}): {e}")
    
    # Fallback to deep-translator if available
    if DeepTranslator is not None:
        try:
            src = source_lang if source_lang != 'auto' else 'auto'
            translated = DeepTranslator(source=src, target=target_lang).translate(text)
            if translated:
                return translated
        except Exception as e:
            st.warning(f"deep-translator error ({source_lang}->{target_lang}): {e}")
    
    # Last resort: use Gemini for translation
    try:
        prompt = f"Translate the following text from {SUPPORTED_LANGUAGES.get(source_lang, source_lang)} to {SUPPORTED_LANGUAGES.get(target_lang, target_lang)}. Only return the translated text, no additional explanations:\n\n{text}"
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.warning(f"Gemini translation failed: {str(e)}")
        return text

def translate_query_to_english(query: str) -> tuple[str, str]:
    """Translate user query to English for processing, return (translated_query, original_lang)."""
    detected_lang = detect_language(query)
    if detected_lang == 'en':
        return query, 'en'
    
    translated_query = translate_text(query, 'en', detected_lang)
    return translated_query, detected_lang

def translate_answer_to_language(answer: str, target_lang: str) -> str:
    """Translate the answer back to the user's language."""
    if target_lang == 'en':
        return answer
    return translate_text(answer, target_lang, 'en')


# --- Core Functions (Adapted for Excel Data) ---
@st.cache_data(show_spinner=False)
def load_master_dataframe():
    global master_df
    if master_df is None:
        try:
            master_df = pd.read_csv("master_groundwater_data.csv", low_memory=False)
            master_df['STATE'] = master_df['STATE'].fillna('').astype(str)
            master_df['DISTRICT'] = master_df['DISTRICT'].fillna('').astype(str)
            master_df['ASSESSMENT UNIT'] = master_df['ASSESSMENT UNIT'].fillna('').astype(str)
            master_df['combined_text'] = master_df.apply(create_detailed_combined_text, axis=1)
            st.success("Master groundwater data loaded.")
        except FileNotFoundError:
            st.error("Error: master_groundwater_data.csv not found. Please run excel_ingestor.py first.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading master groundwater data: {str(e)}")
            st.stop()
    return master_df

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

@st.cache_data(show_spinner=False)
def get_embeddings(texts):
    """Convert text into embeddings"""
    try:
        if model is None:
            return None
        
        # Additional safety check for meta tensor issues
        try:
            # Test with a small sample first
            test_text = texts[0] if texts else "test"
            test_embedding = model.encode([test_text], show_progress_bar=False)
            
            # If test works, proceed with full batch
            return model.encode(texts, show_progress_bar=False)
            
        except Exception as meta_error:
            if "meta tensor" in str(meta_error).lower():
                st.warning("Meta tensor issue detected. Attempting to reinitialize model...")
                
                # Try to reinitialize the model
                try:
                    # Clear the model
                    model = None
                    
                    # Reinitialize with different approach
                    model = SentenceTransformer(
                        "all-MiniLM-L6-v2",
                        device="cpu",
                        model_kwargs={
                            "torch_dtype": torch.float32,
                            "trust_remote_code": False,
                        },
                    )
                    
                    # Force proper initialization
                    model.eval()
                    
                    # Test again
                    test_embedding = model.encode([test_text], show_progress_bar=False)
                    return model.encode(texts, show_progress_bar=False)
                    
                except Exception as reinit_error:
                    st.error(f"Model reinitialization failed: {str(reinit_error)}")
                    return None
            else:
                raise meta_error
                
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
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
        collections = qdrant_client.get_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)
        
        if not collection_exists:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            st.info(f"Created new collection: {COLLECTION_NAME}")
        else:
            st.info(f"Using existing collection: {COLLECTION_NAME}")
        
        # Create indexes
        for field_name, field_type in [("Assessment_Year", PayloadSchemaType.INTEGER), 
                                      ("STATE", PayloadSchemaType.KEYWORD),
                                      ("DISTRICT", PayloadSchemaType.KEYWORD)]:
            try:
                qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=field_type
                )
            except Exception:
                pass  # Index might already exist
        
        return True
    except Exception as e:
        st.error(f"Error setting up collection: {str(e)}")
        return False

def check_excel_embeddings_exist():
    """Check if embeddings for master_groundwater_data.csv exist in Qdrant by counting points"""
    try:
        collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        return collection_info.points_count > 0
    except Exception as e:
        st.exception(f"Error checking existing embeddings: {str(e)}")
        return False

def clear_all_embeddings():
    """Deletes all points in the collection"""
    try:
        qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=models.FilterSelector(filter=models.Filter(must=[])))
        st.success("🗑️ Cleared all embeddings and payloads from collection.")
        st.session_state.bm25_model = None
        st.session_state.all_chunks = None
        st.session_state.embeddings_uploaded = False
    except Exception as e:
        st.error(f"Error clearing embeddings: {e}")

def upload_excel_to_qdrant(df_to_upload, batch_size=1000):
    """Upload Excel DataFrame rows to Qdrant in batches"""
    if df_to_upload.empty:
        return False
    
    try:
        texts = df_to_upload['combined_text'].tolist()
        embeddings = get_embeddings(texts)
        if not is_valid_embeddings(embeddings):
            st.error("Failed to generate embeddings for Excel data.")
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
            
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True
            )
            total_uploaded += len(points)
            st.write(f"Uploaded {total_uploaded}/{len(df_to_upload)} records...")
        
        st.success(f"✅ All {total_uploaded} Excel rows uploaded to Qdrant.")
        return True
    except Exception as e:
        st.error(f"Error uploading Excel data to Qdrant: {str(e)}")
        return False

def load_all_excel_chunks_for_bm25(df_for_bm25=None):
    """Loads all combined_text chunks from Qdrant or DataFrame to build a global BM25 model."""
    all_chunks = []
    if df_for_bm25 is not None and not df_for_bm25.empty:
        all_chunks = df_for_bm25['combined_text'].tolist()
    elif check_excel_embeddings_exist():
        try:
            scroll_result, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100000,
                with_payload=True,
                with_vectors=False
            )
            all_chunks = [point.payload.get("text", "") for point in scroll_result if point.payload.get("text")]
            if all_chunks:
                st.session_state.bm25_df = pd.DataFrame([point.payload for point in scroll_result])
                st.session_state.bm25_df['combined_text'] = all_chunks
        except Exception as e:
            st.error(f"Error retrieving chunks from Qdrant for BM25: {str(e)}")
            return

    if all_chunks:
        tokenized_all_chunks = [tokenize_text(chunk) for chunk in all_chunks]
        st.session_state.bm25_model = BM25Okapi(tokenized_all_chunks)
        st.session_state.all_chunks = all_chunks
        st.info("BM25 model initialized with all Excel chunks.")
    else:
        st.session_state.bm25_model = None
        st.session_state.all_chunks = []
        st.session_state.bm25_df = pd.DataFrame(columns=master_df.columns if master_df is not None else [])
        st.info("No Excel data found in Qdrant or DataFrame for BM25 initialization. Initialized with empty data.")

@st.cache_data(show_spinner=False)
def search_excel_chunks(query_text, year=None, target_state=None, target_district=None, extracted_parameters=None):
    """Retrieve most relevant Excel data rows using hybrid search, with optional year and location filtering."""
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
        if model is not None:
            query_vector = model.encode([query_text])[0]
            qdrant_results = qdrant_client.search(
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
        if 'bm25_model' in st.session_state and 'all_chunks' in st.session_state and 'bm25_df' in st.session_state:
            bm25_model = st.session_state.bm25_model
            all_bm25_chunks = st.session_state.all_chunks
            bm25_df_full = st.session_state.bm25_df
            
            tokenized_query = tokenize_text(query_text)
            bm25_scores = bm25_model.get_scores(tokenized_query)
            
            for i, score in enumerate(bm25_scores):
                if score > 0:
                    chunk_text_bm25 = all_bm25_chunks[i]
                    if (year and bm25_df_full.iloc[i]['Assessment_Year'] != year) or \
                       (target_state and bm25_df_full.iloc[i]['STATE'] != target_state) or \
                       (target_district and bm25_df_full.iloc[i]['DISTRICT'] != target_district):
                        continue
                    sparse_hits[chunk_text_bm25] = score
        
        # Hybrid Scoring
        combined_scores = {}
        alpha = st.session_state.alpha if model is not None else 0.0

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
                matching_rows = bm25_df_full[bm25_df_full['combined_text'] == chunk_text]
                if not matching_rows.empty:
                    payload = matching_rows.iloc[0].to_dict()
                else:
                    payload = {"text": chunk_text}
            results_with_payloads.append({"score": score, "data": payload})

        return results_with_payloads

    except Exception as e:
        st.error(f"Error performing hybrid search: {str(e)}")
        return []

def re_rank_chunks(query_text, candidate_results, top_k=5):
    """Re-ranks candidate results based on semantic similarity to the query."""
    if not candidate_results:
        return []

    if model is None:
        sorted_candidates = sorted(candidate_results, key=lambda r: r.get('score', 0), reverse=True)
        return sorted_candidates[:top_k]

    candidate_texts = [res['data'].get('text', '') for res in candidate_results]
    if not candidate_texts:
        return []

    query_embedding = model.encode([query_text])[0]
    chunk_embeddings = model.encode(candidate_texts)

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
    prompt = (
        f"Generate {num_terms} related terms or short phrases for the following query. "
        f"Output only the terms, separated by commas, no other text.\n"
        f"Query: {query}\n"
        f"Related terms:"
    )
    try:
        response = gemini_model.generate_content(prompt)
        expanded_terms = [term.strip() for term in response.text.strip().split(',') if term.strip()]
        return " ".join(expanded_terms) 
    except Exception as e:
        st.warning(f"Error expanding query: {e}")
        return ""

def generate_answer_from_gemini(query, context_data, year=None, target_state=None, target_district=None, chat_history=None, extracted_parameters=None, user_language='en'):
    """Use Gemini to answer the question based on structured Excel data with multilingual support."""
    if not query or not context_data:
        return "Please provide both a question and relevant data context."
    
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

    # Column descriptions for better understanding
    column_descriptions = """
COLUMN DESCRIPTIONS:
- S.No: Serial number
- STATE: State name
- DISTRICT: District name  
- ASSESSMENT UNIT: Assessment unit name
- Assessment_Year: Year of assessment
- Rainfall (mm): Precipitation data (C=Consolidated, NC=Non-Consolidated, PQ=Partially Consolidated, Total=All)
- Total Geographical Area (ha): Total area in hectares
- Ground Water Recharge (ham): Groundwater recharge from various sources (rainfall, canals, irrigation, tanks, etc.)
- Inflows and Outflows (ham): Water movement (base flow, stream recharges, lateral flows, vertical flows, evaporation, transpiration)
- Annual Ground water Recharge (ham): Total annual groundwater recharge
- Environmental Flows (ham): Water reserved for environmental needs
- Annual Extractable Ground water Resource (ham): Total extractable groundwater resource
- Ground Water Extraction for all uses (ha.m): Water extraction for domestic, industrial, and irrigation use
- Stage of Ground Water Extraction (%): Percentage of groundwater extraction relative to availability
- Allocation of Ground Water Resource for Domestic Utilisation for projected year 2025 (ham): Future domestic water allocation
- Net Annual Ground Water Availability for Future Use (ham): Available groundwater for future use
- Quality Tagging: Water quality parameters (major and other parameters)
- Additional Potential Resources Under Specific Conditions(ham): Additional resources under specific conditions
- Coastal Areas: Coastal area groundwater data
- In-Storage Unconfined Ground Water Resources(ham): Stored unconfined aquifer resources
- Dynamic Confined Ground Water Resources(ham): Dynamic confined aquifer resources
- In-Storage Confined Ground Water Resources(ham): Stored confined aquifer resources
- Dynamic Semi Confined Ground Water Resources (ham): Dynamic semi-confined aquifer resources
- In-Storage Semi Confined Ground Water Resources (ham): Stored semi-confined aquifer resources
- Total Ground Water Availability in the area (ham): Total groundwater availability (fresh and saline)
"""

    prompt = (
        f"You are an expert groundwater data analyst. Provide a comprehensive summary of the groundwater data.{language_instruction}\n"
        f"""Here are the rules for data presentation:
- If a specific year is provided, give data for that year.
- If no specific year is provided, summarize the data including averages across all available years for the specified location (state or district).
- ALWAYS include ALL relevant column data in your response - don't just mention a few key metrics.
- Present data in a structured format with clear headings and values.
- Include both numerical values and their units (ham, ha, mm, %).
- For each data point, explain what it represents and its significance.
- Do NOT ask follow-up questions about what aspect of estimation the user is interested in. Provide a comprehensive summary of ALL available relevant metrics.
- If data is missing for certain columns, mention that explicitly.
"""
        f"{column_descriptions}\n"
        f"{conversation_history_str}"
        f"{extracted_params_str}"
        f"Base your answer ONLY on the following groundwater data{location_info}{year_info}:\n{context_str}\n\n"
        f"If the data doesn't contain the answer, state that. Do NOT make up information.\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    try:
        response = gemini_model.generate_content(prompt)
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

def _user_archive_dir(username: str) -> str:
    safe_username = re.sub(r"[^a-zA-Z0-9_-]", "_", username)
    p = os.path.join(CHAT_HISTORY_DIR, safe_username)
    os.makedirs(p, exist_ok=True)
    return p

def _slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"[^a-zA-Z0-9 _-]", "", text)
    text = text.replace(" ", "-")
    return text[:max_len] or "chat"

def archive_current_chat(username: str, messages: List[Dict[str, str]]) -> Optional[str]:
    if not username or not messages:
        return None
    try:
        first_user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user" and m.get("content")), "")
        title_slug = _slugify(first_user_msg)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{title_slug}.json"
        archive_dir = _user_archive_dir(username)
        archive_path = os.path.join(archive_dir, filename)
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
        return archive_path
    except Exception:
        return None

def list_archived_chats(username: str) -> List[Dict[str, str]]:
    if not username:
        return []
    archive_dir = _user_archive_dir(username)
    try:
        files = [os.path.join(archive_dir, f) for f in os.listdir(archive_dir) if f.endswith('.json')]
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        chats = []
        for p in files:
            name = os.path.basename(p)
            try:
                ts_part, slug_part = name.split('_', 1)
                ts_label = pd.to_datetime(ts_part, format="%Y%m%d").strftime("%Y-%m-%d") if len(ts_part) >= 8 else ts_part
                label = f"{ts_label} — {os.path.splitext(slug_part)[0]}"
            except Exception:
                label = os.path.splitext(name)[0]
            chats.append({"path": p, "label": label})
        return chats
    except Exception:
        return []

# ===== Streamlit UI =====
st.set_page_config(layout="wide")
app_title_suffix = ""
if "user" in st.session_state and st.session_state.get("user"):
    app_title_suffix = f" — Signed in as {st.session_state['user']}"
st.title(f"💧 Groundwater Chatbot (Multilingual){app_title_suffix}")

# Language selection in sidebar
with st.sidebar:
    st.header("🌐 Language Settings")
    selected_language = st.selectbox(
        "Select your preferred language:",
        options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda x: SUPPORTED_LANGUAGES[x],
        index=0,  # Default to English
        key="language_selector"
    )
    
    # Store selected language in session state
    if "selected_language" not in st.session_state:
        st.session_state.selected_language = selected_language
    else:
        st.session_state.selected_language = selected_language

    st.header("🔧 Options")
    if st.button("🗑 Clear All Embeddings", help="This will delete all stored Excel data embeddings from Qdrant"):
        clear_all_embeddings()
        st.session_state.embeddings_uploaded = False
        st.rerun()
    if st.button("🔄 Clear Conversation", help="Start a new chat session by clearing the current conversation history"):
        st.session_state.messages = []
        if st.session_state.get("user"):
            save_chat_history(st.session_state["user"], st.session_state.messages)
        st.rerun()
    
    # Add retry button for model initialization if it failed
    if model is None:
        if st.button("🔄 Retry Model Initialization", help="Try to reinitialize the SentenceTransformer model"):
            st.info("Attempting to reinitialize SentenceTransformer...")
            if initialize_sentence_transformer():
                st.success("✅ Model initialization successful! Dense embeddings are now available.")
                st.rerun()
            else:
                st.error("❌ Model initialization failed. Please check your PyTorch installation.")

    st.header("Hyperparameters")
    st.session_state.alpha = st.slider(
        "Alpha for Hybrid Search (Dense vs Sparse)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust the weighting between dense (vector) and sparse (BM25) retrieval."
    )

    st.header("💬 Chats")
    if "user" not in st.session_state:
        st.session_state.user = None
    
    if st.button("🆕 New chat", use_container_width=True):
        if st.session_state.messages and st.session_state.user:
            archive_current_chat(st.session_state.user, st.session_state.messages)
            save_chat_history(st.session_state.user, [])
        st.session_state.messages = []
        st.rerun()

    # List archived chats for the user
    if st.session_state.user:
        archived = list_archived_chats(st.session_state.user)
        if archived:
            chat_list_container = st.container()
            max_show = 30
            for i, chat in enumerate(archived[:max_show]):
                if chat_list_container.button(f"{chat['label']}", key=f"chat_item_{i}", use_container_width=True):
                    try:
                        with open(chat["path"], "r", encoding="utf-8") as f:
                            st.session_state.messages = json.load(f)
                        save_chat_history(st.session_state.user, st.session_state.messages)
                        st.rerun()
                    except Exception:
                        st.warning("Failed to open the selected chat.")
            if len(archived) > max_show:
                st.caption(f"Showing latest {max_show} chats out of {len(archived)}")
        else:
            st.caption("No archived chats yet. Start a new chat to create one.")

    st.header("ℹ Information")
    st.info("""
    *How it works:*
    - Processes `master_groundwater_data.csv` (generated by `excel_ingestor.py`)
    - Creates embeddings for each row and stores them in Qdrant.
    - Uses hybrid search (dense + sparse) and reranking for relevant data retrieval.
    - Gemini LLM generates answers based on the retrieved structured data.
    - **Supports multiple languages** - ask questions in your preferred language!
    - Supports year-specific queries or aggregates data if no year is mentioned.
    """)

    st.header("👤 Account")
    if "user" not in st.session_state:
        st.session_state.user = None
    if st.session_state.user:
        st.success(f"Logged in as {st.session_state.user}")
        if st.button("🚪 Log out"):
            st.session_state.user = None
            st.rerun()
    else:
        auth_tabs = st.tabs(["Login", "Register"]) 
        with auth_tabs[0]:
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            if st.button("Sign in", key="login_button"):
                if authenticate_user(login_username, login_password):
                    st.session_state.user = login_username.strip()
                    st.session_state.messages = load_chat_history(st.session_state.user)
                    st.success("Logged in successfully.")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        with auth_tabs[1]:
            reg_username = st.text_input("Username", key="reg_username")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_password2 = st.text_input("Confirm Password", type="password", key="reg_password2")
            if st.button("Create account", key="register_button"):
                if not reg_username.strip():
                    st.error("Username is required.")
                elif not reg_password:
                    st.error("Password is required.")
                elif reg_password != reg_password2:
                    st.error("Passwords do not match.")
                elif register_user(reg_username, reg_password):
                    st.success("Account created. Please log in.")
                else:
                    st.error("Username already exists or invalid input.")

# --- Session State Initialization ---
if 'embeddings_uploaded' not in st.session_state:
    st.session_state.embeddings_uploaded = False
if 'bm25_model' not in st.session_state:
    st.session_state.bm25_model = None
if 'all_chunks' not in st.session_state:
    st.session_state.all_chunks = None
if 'bm25_df' not in st.session_state:
    st.session_state.bm25_df = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user' not in st.session_state:
    st.session_state.user = None
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = 'en'

# --- Main App Logic ---
master_df = load_master_dataframe()

# Setup Qdrant collection and upload embeddings if not already done
if setup_collection():
    if not st.session_state.embeddings_uploaded:
        st.info("⏳ Checking for existing Excel data embeddings...")
        if check_excel_embeddings_exist():
            st.info("📚 Found existing Excel data embeddings. Initializing BM25...")
            load_all_excel_chunks_for_bm25()
            st.session_state.embeddings_uploaded = True
            st.success("✅ Excel data embeddings loaded and BM25 initialized.")
        else:
            st.info("⏳ Uploading Excel data to Qdrant...")
            if upload_excel_to_qdrant(master_df):
                load_all_excel_chunks_for_bm25(master_df)
                st.session_state.embeddings_uploaded = True
                st.success("✅ Excel data processed and indexed.")
            else:
                st.error("Failed to upload Excel data embeddings to Qdrant.")
    elif st.session_state.embeddings_uploaded and st.session_state.bm25_model is None:
        st.info("📚 Embeddings previously uploaded. Initializing BM25 from existing data...")
        load_all_excel_chunks_for_bm25()

# --- Chat Interface ---
st.header("Ask a Question about Groundwater Data")
if not st.session_state.user:
    st.warning("Please log in to chat and save your conversation history.")

# Display current language and model status
current_lang_name = SUPPORTED_LANGUAGES.get(st.session_state.selected_language, 'English')
st.info(f"🌐 Current language: {current_lang_name}")

# Display model status
if model is not None:
    st.success("✅ Dense embeddings available (hybrid search enabled)")
else:
    st.warning("⚠️ Dense embeddings disabled (BM25-only search)")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input with language-specific placeholder
placeholder_text = "You:"
if st.session_state.selected_language != 'en':
    placeholder_text = f"You (in {current_lang_name}):"

user_query = st.chat_input(placeholder_text, key="user_input")

if user_query and st.session_state.embeddings_uploaded:
    # Detect and translate query to English for processing
    original_query = user_query
    translated_query, detected_lang = translate_query_to_english(user_query)
    
    # Show language detection info
    if detected_lang != 'en':
        st.info(f"🌐 Detected language: {SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)}")
        st.info(f"🔄 Translated query: {translated_query}")
    
    # Add user message to chat history (store original query)
    st.session_state.messages.append({"role": "user", "content": original_query})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(original_query)
        if detected_lang != 'en':
            st.caption(f"Translated: {translated_query}")

    # Process the translated query
    year = None
    year_match = re.search(r'\b(19|20)\d{2}\b', translated_query)
    if year_match:
        year = int(year_match.group(0))

    # Extract location from translated query
    target_state = None
    target_district = None

    if master_df is not None:
        unique_states = master_df['STATE'].unique().tolist()
        unique_districts = master_df['DISTRICT'].unique().tolist()

        for state in unique_states:
            if re.search(r'\b' + re.escape(state) + r'\b', translated_query, re.IGNORECASE):
                target_state = state
                break

        if target_state:
            districts_in_state = master_df[master_df['STATE'] == target_state]['DISTRICT'].unique().tolist()
            for district in districts_in_state:
                if re.search(r'\b' + re.escape(district) + r'\b', translated_query, re.IGNORECASE):
                    target_district = district
                    break
    
    # Enhanced NLP for extracting specific groundwater parameters
    extracted_parameters = {}
    doc = nlp(translated_query)

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
    
    st.info("✨ Expanding query...")
    expanded_terms = expand_query(translated_query)
    expanded_query_text = f"{translated_query} {expanded_terms}".strip()
    
    st.info("🔍 Searching for relevant information...")
    candidate_results = search_excel_chunks(expanded_query_text, year=year, target_state=target_state, target_district=target_district, extracted_parameters=extracted_parameters)
    
    re_ranked_results = re_rank_chunks(expanded_query_text, candidate_results, top_k=5)
    
    if re_ranked_results:
        context_data = [res['data'] for res in re_ranked_results]
        
        # Show data completeness information
        if context_data:
            total_columns = len(context_data[0].keys()) if context_data else 0
            data_entries = len(context_data)
            st.info(f"📊 Found {data_entries} data entries with {total_columns} columns each")
            
            # Show data completeness for first entry
            if context_data:
                first_entry = context_data[0]
                available_data = sum(1 for v in first_entry.values() if pd.notna(v) and str(v).strip() != '')
                completeness = (available_data / len(first_entry)) * 100
                st.info(f"📈 Data completeness: {completeness:.1f}% ({available_data}/{len(first_entry)} columns have data)")
        
        st.info("🤖 Generating comprehensive answer from Gemini...")
        
        # Generate answer in the user's language
        answer = generate_answer_from_gemini(
            translated_query, 
            context_data, 
            year=year, 
            target_state=target_state, 
            target_district=target_district, 
            chat_history=st.session_state.messages, 
            extracted_parameters=extracted_parameters,
            user_language=detected_lang
        )
        
        st.subheader("🧠 Answer:")
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        if st.session_state.get("user"):
            save_chat_history(st.session_state["user"], st.session_state.messages)
    else:
        warning_message = "I couldn't find enough relevant information in the groundwater data to answer your question."
        if detected_lang != 'en':
            warning_message = translate_answer_to_language(warning_message, detected_lang)
        
        with st.chat_message("assistant"):
            st.warning(warning_message)
        st.session_state.messages.append({"role": "assistant", "content": warning_message})
        if st.session_state.get("user"):
            save_chat_history(st.session_state["user"], st.session_state.messages)
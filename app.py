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

load_dotenv()

# --- Environment Variables ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Initialize Clients and Models ---
# Qdrant Client
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60 # Increased timeout to 60 seconds
    )
except Exception as e:
    st.error(f"Failed to initialize Qdrant client: {str(e)}")
    st.stop()

# Sentence Transformer for Embeddings
try:
    model = SentenceTransformer("all-MiniLM-L6-v2", device=torch.device('cpu'))
except Exception as e:
    st.error(f"Failed to initialize SentenceTransformer: {str(e)}")
    st.stop()

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
VECTOR_SIZE = 384  # Based on all-MiniLM-L6-v2 model
MIN_SIMILARITY_SCORE = 0.5  # Adjusted for potentially more diverse Excel data

# --- Global DataFrames ---
master_df = None # To be loaded from master_groundwater_data.csv

# --- Core Functions (Adapted for Excel Data) ---
@st.cache_data(show_spinner=False)
def load_master_dataframe():
    global master_df
    if master_df is None:
        try:
            master_df = pd.read_csv("master_groundwater_data.csv", low_memory=False)
            # Ensure relevant columns are strings for consistent embedding
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
    """
    Generates a detailed combined text string for a DataFrame row,
    including all column names and their non-null values.
    """
    parts = []
    for col, value in row.items():
        if pd.notna(value) and value != '' and col not in ['S.No']: # Exclude S.No and empty values
            parts.append(f"{col}: {value}")
    return " | ".join(parts)

def tokenize_text(text):
    """Tokenize text for BM25 processing."""
    return text.lower().split()

@st.cache_data(show_spinner=False)
def get_embeddings(texts):
    """Convert text into embeddings"""
    try:
        return model.encode(texts, show_progress_bar=False)
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
        
        # Create index for Assessment_Year field if it doesn't exist
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="Assessment_Year",
                field_schema=PayloadSchemaType.INTEGER
            )
        except Exception as e:
            st.exception(f"Error creating Assessment_Year payload index: {e}")
            pass # Index might already exist
        
        # Create index for STATE field if it doesn't exist
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="STATE",
                field_schema=PayloadSchemaType.KEYWORD # Use KEYWORD for string matching
            )
        except Exception as e:
            st.exception(f"Error creating STATE payload index: {e}")
            pass # Index might already exist

        # Create index for DISTRICT field if it doesn't exist
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="DISTRICT",
                field_schema=PayloadSchemaType.KEYWORD # Use KEYWORD for string matching
            )
        except Exception as e:
            st.exception(f"Error creating DISTRICT payload index: {e}")
            pass # Index might already exist
        
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
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, row['combined_text'])) # Using hash of combined_text for unique ID
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
                wait=True # Wait for operation to complete before proceeding
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
        # If embeddings exist in Qdrant, retrieve them to build BM25
        try:
            scroll_result, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100000,  # Adjust limit as needed for total number of chunks
                with_payload=True,
                with_vectors=False
            )
            all_chunks = [point.payload.get("text", "") for point in scroll_result if point.payload.get("text")]
            # We also need to recreate a DataFrame for bm25_df if loading from Qdrant
            # This is a bit inefficient, but necessary if BM25 needs the original DataFrame structure.
            # A better approach for very large datasets would be to store BM25-specific info in Qdrant payloads
            if all_chunks:
                # Reconstruct a simplified df for BM25 with just the text and relevant metadata
                # For now, we'll just use the text for BM25, and rely on Qdrant payload for full data
                st.session_state.bm25_df = pd.DataFrame([point.payload for point in scroll_result])
                st.session_state.bm25_df['combined_text'] = all_chunks # Ensure combined_text is consistent
        except Exception as e:
            st.error(f"Error retrieving chunks from Qdrant for BM25: {str(e)}")
            return

    if all_chunks:
        tokenized_all_chunks = [tokenize_text(chunk) for chunk in all_chunks]
        st.session_state.bm25_model = BM25Okapi(tokenized_all_chunks)
        st.session_state.all_chunks = all_chunks # Store original chunks for retrieval
        st.info("BM25 model initialized with all Excel chunks.")
    else:
        st.session_state.bm25_model = None
        st.session_state.all_chunks = []
        st.session_state.bm25_df = pd.DataFrame(columns=master_df.columns if master_df is not None else []) # Initialize as empty DataFrame
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

    # Add filters for extracted parameters
    if extracted_parameters:
        for param_type, value in extracted_parameters.items():
            # For now, a simple text match for the parameter type. 
            # More sophisticated filtering might be needed based on exact column names and numerical ranges.
            qdrant_filter_conditions.append(
                FieldCondition(
                    key="text", # Searching within the combined_text field
                    match=MatchValue(value=str(param_type).lower()) # Match the parameter type as a keyword
                )
            )
            # If a numerical value is associated, consider adding range filters for specific columns
            # This part would require mapping param_type to actual numerical columns in your DataFrame
            # For instance, if 'rainfall' maps to 'Rainfall (mm) - Total'
            # if param_type == "rainfall" and isinstance(value, (int, float)):
            #     qdrant_filter_conditions.append(
            #         FieldCondition(
            #             key="Rainfall (mm) - Total",
            #             range=models.Range(gte=value * 0.9, lte=value * 1.1) # Example: 10% range
            #         )
            #     )
    
    qdrant_filter = Filter(must=qdrant_filter_conditions) if qdrant_filter_conditions else None

    try:
        # --- Dense Retrieval (Qdrant) ---
        query_vector = model.encode([query_text])[0]
        qdrant_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=qdrant_filter, # Apply year and location filters here
            limit=20, # Get more candidates for re-ranking
            with_payload=True
        )
        dense_hits = {hit.payload.get("text", ""): hit.score for hit in qdrant_results}
        dense_payloads = {hit.payload.get("text", ""): hit.payload for hit in qdrant_results}

        # --- Sparse Retrieval (BM25) ---
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
                    # Apply year and location filters to BM25 results if specified
                    if (year and bm25_df_full.iloc[i]['Assessment_Year'] != year) or \
                       (target_state and bm25_df_full.iloc[i]['STATE'] != target_state) or \
                       (target_district and bm25_df_full.iloc[i]['DISTRICT'] != target_district):
                        continue
                    sparse_hits[chunk_text_bm25] = score
        
        # --- Hybrid Scoring ---
        combined_scores = {}
        alpha = st.session_state.alpha # Weight for dense vs sparse

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
        for chunk_text, score in sorted_chunks_with_scores[:20]: # Use top 20 candidates for re-ranking
            # Prioritize dense_payloads, as they come directly from Qdrant with full payload
            if chunk_text in dense_payloads:
                payload = dense_payloads[chunk_text]
            else: # Fallback to finding payload from bm25_df_full if only sparse hit
                # This is less efficient, but necessary if BM25 needs the original DataFrame structure.
                # A better approach for very large datasets would be to store BM25-specific info in Qdrant payloads
                matching_rows = bm25_df_full[bm25_df_full['combined_text'] == chunk_text]
                if not matching_rows.empty:
                    payload = matching_rows.iloc[0].to_dict()
                else:
                    payload = {"text": chunk_text} # Minimal payload if not found elsewhere
            results_with_payloads.append({"score": score, "data": payload})

        return results_with_payloads

    except Exception as e:
        st.error(f"Error performing hybrid search: {str(e)}")
        return []

def re_rank_chunks(query_text, candidate_results, top_k=5):
    """Re-ranks candidate results (each containing 'data' and 'score') based on semantic similarity to the query."""
    if not candidate_results:
        return []

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

    # Combine original score (from hybrid search) with re-ranking score
    # A simple weighted average or multiplication can be used
    re_ranked_scores = []
    for i, res in enumerate(candidate_results):
        # You can adjust the weighting (e.g., 0.5 * res['score'] + 0.5 * similarities[i])
        # For now, we will mostly rely on the re-ranking similarity as the primary re-ranking factor
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

def generate_answer_from_gemini(query, context_data, year=None, target_state=None, target_district=None, chat_history=None, extracted_parameters=None):
    """Use Gemini to answer the question based on structured Excel data."""
    if not query or not context_data:
        return "Please provide both a question and relevant data context."
    
    # Format structured data into a readable string for the LLM
    data_summary = []
    for item in context_data:
        data_summary.append(f"State: {item.get('STATE')}, District: {item.get('DISTRICT')}, Assessment Unit: {item.get('ASSESSMENT UNIT')}, Year: {item.get('Assessment_Year')}\n")
        for key, value in item.items():
            # Include specific relevant metrics, excluding internal keys or combined_text
            if key not in ['STATE', 'DISTRICT', 'ASSESSMENT UNIT', 'Assessment_Year', 'combined_text', 'text'] and pd.notna(value):
                data_summary.append(f"   - {key}: {value}")
        data_summary.append("---")
    
    # If no specific year, calculate and add averages to the context (for district or state-level data)
    if year is None:
        numerical_columns = [
            'ANNUAL EXTRACTABLE GROUNDWATER RESOURCE (in MCM/year)',
            'ANNUAL GROUNDWATER EXTRACTION FOR IRRIGATION (in MCM/year)',
            'ANNUAL GROUNDWATER EXTRACTION FOR DOMESTIC & INDUSTRIAL SUPPLY (in MCM/year)',
            'TOTAL ANNUAL GROUNDWATER EXTRACTION (in MCM/year)',
            'ANNUAL GROUNDWATER RECHARGE (in MCM/year)',
            'GROSS DRAFT (in MCM/year)',
            'NET ANNUAL GROUNDWATER AVAILABILITY (in MCM/year)',
            'STAGE OF GROUNDWATER EXTRACTION (%)'
        ]

        context_df = pd.DataFrame(context_data)
        avg_data = {}
        for col in numerical_columns:
            if col in context_df.columns:
                numeric_values = pd.to_numeric(context_df[col], errors='coerce').dropna()
                if not numeric_values.empty:
                    avg_data[col] = numeric_values.mean()
        
        if avg_data:
            data_summary.append(f"\n--- Averages for the retrieved data ---")
            for key, value in avg_data.items():
                data_summary.append(f"   - Average {key}: {value:.2f}")
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

    prompt = (
        f"You are an expert groundwater data analyst. Provide a concise summary of the groundwater data.\n"
        f"""Here are the rules for data presentation:
- If a specific year is provided, give data for that year.
- If no specific year is provided, summarize the data including averages across all available years for the specified location (state or district).
- Do NOT ask follow-up questions about what aspect of estimation the user is interested in if the data contains multiple metrics. Just provide a summary of the available relevant metrics.
|"""
        f"{conversation_history_str}" # Include conversation history here
        f"{extracted_params_str}" # Include extracted parameters here
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

# ===== Streamlit UI =====
st.set_page_config(layout="wide")
st.title("💧 Groundwater Chatbot (Excel Data)")

# Sidebar for options
with st.sidebar:
    st.header("🔧 Options")
    if st.button("🗑 Clear All Embeddings", help="This will delete all stored Excel data embeddings from Qdrant"):
        clear_all_embeddings()
        st.session_state.embeddings_uploaded = False
        st.rerun()
    if st.button("🔄 Clear Conversation", help="Start a new chat session by clearing the current conversation history"):
        st.session_state.messages = []
        st.experimental_rerun()

    st.header("Hyperparameters")
    # Slider to adjust the alpha for hybrid search
    st.session_state.alpha = st.slider(
        "Alpha for Hybrid Search (Dense vs Sparse)",
        min_value=0.0,
        max_value=1.0,
        value=0.5, # Default value
        step=0.05,
        help="Adjust the weighting between dense (vector) and sparse (BM25) retrieval."
    )

    st.header("ℹ Information")
    st.info("""
    *How it works:*
    - Processes `master_groundwater_data.csv` (generated by `excel_ingestor.py`)
    - Creates embeddings for each row and stores them in Qdrant.
    - Uses hybrid search (dense + sparse) and reranking for relevant data retrieval.
    - Gemini LLM generates answers based on the retrieved structured data.
    - Supports year-specific queries or aggregates data if no year is mentioned.
    """)

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

# --- Main App Logic ---
master_df = load_master_dataframe()

# Setup Qdrant collection and upload embeddings if not already done
if setup_collection():
    if not st.session_state.embeddings_uploaded:
        st.info("⏳ Checking for existing Excel data embeddings...")
        if check_excel_embeddings_exist():
            st.info("📚 Found existing Excel data embeddings. Initializing BM25...")
            load_all_excel_chunks_for_bm25() # Load from Qdrant directly
            st.session_state.embeddings_uploaded = True
            st.success("✅ Excel data embeddings loaded and BM25 initialized.")
        else:
            st.info("⏳ Uploading Excel data to Qdrant...")
            if upload_excel_to_qdrant(master_df):
                load_all_excel_chunks_for_bm25(master_df) # Load from the newly uploaded master_df
                st.session_state.embeddings_uploaded = True
                st.success("✅ Excel data processed and indexed.")
            else:
                st.error("Failed to upload Excel data embeddings to Qdrant.")
    elif st.session_state.embeddings_uploaded and st.session_state.bm25_model is None:
        # If embeddings were uploaded in a previous run, but BM25 not initialized in this session
        st.info("📚 Embeddings previously uploaded. Initializing BM25 from existing data...")
        load_all_excel_chunks_for_bm25()

# --- Chat Interface ---
st.header("Ask a Question about Groundwater Data")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("You:", key="user_input")

if user_query and st.session_state.embeddings_uploaded:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_query)

    # --- NLP for year extraction ---
    year = None
    year_match = re.search(r'\b(19|20)\d{2}\b', user_query) # Basic regex for 4-digit year
    if year_match:
        year = int(year_match.group(0))

    # --- NLP for location extraction (state and district) ---
    target_state = None
    target_district = None

    if master_df is not None:
        # Extract unique states and districts from the master DataFrame
        unique_states = master_df['STATE'].unique().tolist()
        unique_districts = master_df['DISTRICT'].unique().tolist()

        # Attempt to match a state from the query (case-insensitive)
        for state in unique_states:
            if re.search(r'\b' + re.escape(state) + r'\b', user_query, re.IGNORECASE):
                target_state = state
                break

        # If a state is found, attempt to match a district within that state (case-insensitive)
        if target_state:
            # Filter districts to only those belonging to the target_state if master_df allows
            districts_in_state = master_df[master_df['STATE'] == target_state]['DISTRICT'].unique().tolist()
            for district in districts_in_state:
                if re.search(r'\b' + re.escape(district) + r'\b', user_query, re.IGNORECASE):
                    target_district = district
                    break
    
    # --- Enhanced NLP for extracting specific groundwater parameters and values ---
    extracted_parameters = {}
    doc = nlp(user_query) # Process query with spaCy for better entity recognition

    # Keywords for groundwater metrics
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

    # Look for numerical values near keywords
    for keyword_type, synonyms in groundwater_keywords.items():
        for token in doc:
            if token.text.lower() in synonyms or any(synonym in token.text.lower() for synonym in synonyms): # Check for both exact match and substring
                # Look for numbers in the vicinity of the keyword
                for i in range(max(0, token.i - 3), min(len(doc), token.i + 4)): # Look 3 tokens before and 3 tokens after
                    if doc[i].is_digit and doc[i].text.isdigit(): # Ensure it's a digit and not part of a word
                        try:
                            value = float(doc[i].text)
                            # Store the first numerical value found for this keyword type
                            if keyword_type not in extracted_parameters:
                                extracted_parameters[keyword_type] = value
                                break # Move to the next keyword type once a value is found
                        except ValueError:
                            continue
    
    st.info("✨ Expanding query...")
    expanded_terms = expand_query(user_query)
    expanded_query_text = f"{user_query} {expanded_terms}".strip()
    
    st.info("🔍 Searching for relevant information...")
    candidate_results = search_excel_chunks(expanded_query_text, year=year, target_state=target_state, target_district=target_district, extracted_parameters=extracted_parameters)
    
    re_ranked_results = re_rank_chunks(expanded_query_text, candidate_results, top_k=5)
    
    if re_ranked_results:
        context_data = [res['data'] for res in re_ranked_results]
        st.info("🤖 Generating answer from Gemini...")
        answer = generate_answer_from_gemini(user_query, context_data, year=year, target_state=target_state, target_district=target_district, chat_history=st.session_state.messages, extracted_parameters=extracted_parameters)
        st.subheader("🧠 Answer:")
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        warning_message = "I couldn't find enough relevant information in the groundwater data to answer your question."
        with st.chat_message("assistant"):
            st.warning(warning_message)
        st.session_state.messages.append({"role": "assistant", "content": warning_message})
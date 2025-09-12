import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from rank_bm25 import BM25Okapi
import spacy
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "groundwater_excel_collection"

_qdrant_client = None
_model = None
_nlp = None
_gemini_model = None
_master_df = None
_bm25_model = None
_all_chunks = None
_bm25_df = None

app = FastAPI(title="Groundwater RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    query: str

def _init_components():
    global _qdrant_client, _model, _nlp, _gemini_model, _master_df
    if _qdrant_client is None:
        try:
            _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        except Exception as e:
            raise Exception(f"Failed to initialize Qdrant client: {str(e)}")
    if _model is None:
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2", device=torch.device('cpu'))
        except Exception as e:
            raise Exception(f"Failed to initialize SentenceTransformer: {str(e)}")
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            raise Exception(f"Failed to initialize spaCy NLP model: {str(e)}")
    if _gemini_model is None and GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini API: {str(e)}")
    if _master_df is None:
        try:
            _master_df = pd.read_csv("master_groundwater_data.csv", low_memory=False)
            _master_df['STATE'] = _master_df['STATE'].fillna('').astype(str)
            _master_df['DISTRICT'] = _master_df['DISTRICT'].fillna('').astype(str)
            _master_df['ASSESSMENT UNIT'] = _master_df['ASSESSMENT UNIT'].fillna('').astype(str)
            _master_df['combined_text'] = _master_df['STATE'] + " " + _master_df['DISTRICT'] + " " + _master_df['ASSESSMENT UNIT'] + " " + _master_df['Assessment_Year'].astype(str)
        except FileNotFoundError:
            raise Exception("Error: master_groundwater_data.csv not found. Please run excel_ingestor.py first.")
        except Exception as e:
            raise Exception(f"Error loading master groundwater data: {str(e)}")

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
            tokenized_chunks = [text.lower().split() for text in _all_chunks]
            _bm25_model = BM25Okapi(tokenized_chunks)
    except Exception as e:
        print(f"Warning: Could not initialize BM25: {e}")
        _bm25_model = None
        _all_chunks = []
        _bm25_df = pd.DataFrame()

def _search_excel_chunks(query_text, year=None, target_state=None, target_district=None):
    _init_components()
    _load_bm25()
    qdrant_filter_conditions = []
    if year:
        qdrant_filter_conditions.append(FieldCondition(key="Assessment_Year", match=MatchValue(value=year)))
    if target_state:
        qdrant_filter_conditions.append(FieldCondition(key="STATE", match=MatchValue(value=target_state)))
    if target_district:
        qdrant_filter_conditions.append(FieldCondition(key="DISTRICT", match=MatchValue(value=target_district)))
    qdrant_filter = Filter(must=qdrant_filter_conditions) if qdrant_filter_conditions else None
    try:
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
        sparse_hits = {}
        if _bm25_model and _all_chunks and _bm25_df is not None:
            tokenized_query = query_text.lower().split()
            bm25_scores = _bm25_model.get_scores(tokenized_query)
            for i, score in enumerate(bm25_scores):
                if score > 0 and i < len(_all_chunks):
                    chunk_text_bm25 = _all_chunks[i]
                    if isinstance(_bm25_df, pd.DataFrame) and i < len(_bm25_df):
                        row_i = _bm25_df.iloc[i]
                        if ((year and row_i.get('Assessment_Year') != year) or
                            (target_state and row_i.get('STATE') != target_state) or
                            (target_district and row_i.get('DISTRICT') != target_district)):
                            continue
                    sparse_hits[chunk_text_bm25] = score
        combined_scores = {}
        alpha = 0.5
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
        results_with_payloads = []
        for chunk_text, score in sorted_chunks_with_scores[:20]:
            if chunk_text in dense_payloads:
                payload = dense_payloads[chunk_text]
            else:
                if isinstance(_bm25_df, pd.DataFrame):
                    matching_rows = _bm25_df[_bm25_df.get('combined_text', pd.Series(index=_bm25_df.index, dtype=str)) == chunk_text]
                    if not matching_rows.empty:
                        payload = matching_rows.iloc[0].to_dict()
                    else:
                        payload = {"text": chunk_text}
                else:
                    payload = {"text": chunk_text}
            results_with_payloads.append({"score": score, "data": payload})
        return results_with_payloads
    except Exception as e:
        print(f"Error performing hybrid search: {str(e)}")
        return []

def _re_rank_chunks(query_text, candidate_results, top_k=5):
    if not candidate_results:
        return []
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

def _expand_query(query, num_terms=3):
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

def _generate_answer_from_gemini(query, context_data, year=None, target_state=None, target_district=None):
    if not query or not context_data:
        return "Please provide both a question and relevant data context."
    if not _gemini_model:
        lines = []
        for item in context_data[:3]:
            lines.append(f"State: {item.get('STATE')}, District: {item.get('DISTRICT')}, Unit: {item.get('ASSESSMENT UNIT')}")
        return f"No LLM configured. Top matches:\n" + "\n".join(lines)
    data_summary = []
    for item in context_data:
        data_summary.append(f"State: {item.get('STATE')}, District: {item.get('DISTRICT')}, Assessment Unit: {item.get('ASSESSMENT UNIT')}, Year: {item.get('Assessment_Year')}")
        for key, value in item.items():
            if key not in ['STATE', 'DISTRICT', 'ASSESSMENT UNIT', 'Assessment_Year', 'combined_text', 'text'] and pd.notna(value):
                data_summary.append(f"  - {key}: {value}")
        data_summary.append("---")
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
                data_summary.append(f"  - Average {key}: {value:.2f}")
            data_summary.append("---")
    context_str = "\n".join(data_summary)
    year_info = f" for the year {year}" if year else " (averaged across all available years)"
    location_info = f" for {target_district} District, {target_state}" if target_district else (f" for {target_state}" if target_state else "")
    prompt = (
        f"You are an expert groundwater data analyst. Provide a concise summary of the groundwater data.\n"
        f"""Here are the rules for data presentation:
- If a specific year is provided, give data for that year.
- If no specific year is provided, summarize the data including averages across all available years for the specified location (state or district).
- Do NOT ask follow-up questions about what aspect of estimation the user is interested in if the data contains multiple metrics. Just provide a summary of the available relevant metrics.
"""
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

def answer_query(query: str) -> str:
    query = (query or '').strip()
    if not query:
        return "Please provide a question."
    try:
        _init_components()
    except Exception as e:
        return f"Initialization error: {str(e)}"
    year = None
    year_match = re.search(r'\b(19|20)\d{2}\b', query)
    if year_match:
        year = int(year_match.group(0))
    target_state = None
    target_district = None
    if _master_df is not None:
        unique_states = _master_df['STATE'].unique().tolist()
        unique_districts = _master_df['DISTRICT'].unique().tolist()
        for state in unique_states:
            if re.search(r'\b' + re.escape(state) + r'\b', query, re.IGNORECASE):
                target_state = state
                break
        if target_state:
            districts_in_state = _master_df[_master_df['STATE'] == target_state]['DISTRICT'].unique().tolist()
            for district in districts_in_state:
                if re.search(r'\b' + re.escape(district) + r'\b', query, re.IGNORECASE):
                    target_district = district
                    break
    expanded_terms = _expand_query(query)
    expanded_query_text = f"{query} {expanded_terms}".strip()
    candidate_results = _search_excel_chunks(expanded_query_text, year=year, target_state=target_state, target_district=target_district)
    if not candidate_results:
        return "I couldn't find enough relevant information in the groundwater data to answer your question."
    re_ranked_results = _re_rank_chunks(expanded_query_text, candidate_results, top_k=5)
    if not re_ranked_results:
        return "I couldn't find enough relevant information in the groundwater data to answer your question."
    context_data = [res['data'] for res in re_ranked_results]
    answer = _generate_answer_from_gemini(query, context_data, year=year, target_state=target_state, target_district=target_district)
    return answer

@app.post("/ask")
async def ask(request: AskRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    try:
        answer = answer_query(query)
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.on_event("startup")
def startup_event():
    _init_components()
    _load_bm25()

# Run with: uvicorn main:app --reload --port 8000
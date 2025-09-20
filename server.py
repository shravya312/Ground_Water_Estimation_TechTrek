from flask import Flask, request, jsonify
from flask_cors import CORS
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

load_dotenv()

app = Flask(__name__)
CORS(app)

# Environment Variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "groundwater_excel_collection"

# Global singletons
_qdrant_client = None
_model = None
_nlp = None
_gemini_model = None
_master_df = None
_bm25_model = None
_all_chunks = None
_bm25_df = None

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
        # Check if embeddings exist in Qdrant
        collection_info = _qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        if collection_info.points_count > 0:
            # Load from Qdrant
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
            # Load from master_df
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
    """Retrieve most relevant Excel data rows using hybrid search."""
    _init_components()
    _load_bm25()
    
    # Build Qdrant filter
    qdrant_filter_conditions = []
    if year:
        qdrant_filter_conditions.append(FieldCondition(key="Assessment_Year", match=MatchValue(value=year)))
    if target_state:
        qdrant_filter_conditions.append(FieldCondition(key="STATE", match=MatchValue(value=target_state)))
    if target_district:
        qdrant_filter_conditions.append(FieldCondition(key="DISTRICT", match=MatchValue(value=target_district)))
    
    qdrant_filter = Filter(must=qdrant_filter_conditions) if qdrant_filter_conditions else None
    
    try:
        # Dense Retrieval (Qdrant)
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
            tokenized_query = query_text.lower().split()
            bm25_scores = _bm25_model.get_scores(tokenized_query)
            
            for i, score in enumerate(bm25_scores):
                if score > 0 and i < len(_all_chunks):
                    chunk_text_bm25 = _all_chunks[i]
                    # Apply filters
                    if isinstance(_bm25_df, pd.DataFrame) and i < len(_bm25_df):
                        row_i = _bm25_df.iloc[i]
                        if ((year and row_i.get('Assessment_Year') != year) or
                            (target_state and row_i.get('STATE') != target_state) or
                            (target_district and row_i.get('DISTRICT') != target_district)):
                            continue
                    sparse_hits[chunk_text_bm25] = score
        
        # Hybrid Scoring
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
        
        # Retrieve payloads
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
    """Re-ranks candidate results based on semantic similarity."""
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

def _generate_answer_from_gemini(query, context_data, year=None, target_state=None, target_district=None):
    """Use Gemini to answer the question based on structured Excel data."""
    if not query or not context_data:
        return "Please provide both a question and relevant data context."
    
    if not _gemini_model:
        # Fallback without Gemini
        lines = []
        for item in context_data[:3]:
            lines.append(f"State: {item.get('STATE')}, District: {item.get('DISTRICT')}, Unit: {item.get('ASSESSMENT UNIT')}")
        return f"No LLM configured. Top matches:\n" + "\n".join(lines)
    
    # Format structured data
    data_summary = []
    for item in context_data:
        data_summary.append(f"State: {item.get('STATE')}, District: {item.get('DISTRICT')}, Assessment Unit: {item.get('ASSESSMENT UNIT')}, Year: {item.get('Assessment_Year')}")
        for key, value in item.items():
            if key not in ['STATE', 'DISTRICT', 'ASSESSMENT UNIT', 'Assessment_Year', 'combined_text', 'text'] and pd.notna(value):
                data_summary.append(f"  - {key}: {value}")
        data_summary.append("---")
    
    # Add averages if no specific year
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
    """Main function to answer groundwater queries using Streamlit app logic."""
    query = (query or '').strip()
    if not query:
        return "Please provide a question."
    
    try:
        _init_components()
    except Exception as e:
        return f"Initialization error: {str(e)}"
    
    # Extract year from query
    year = None
    year_match = re.search(r'\b(19|20)\d{2}\b', query)
    if year_match:
        year = int(year_match.group(0))
    
    # Extract location from query
    target_state = None
    target_district = None
    
    if _master_df is not None:
        unique_states = _master_df['STATE'].unique().tolist()
        unique_districts = _master_df['DISTRICT'].unique().tolist()
        
        # Match state
        for state in unique_states:
            if re.search(r'\b' + re.escape(state) + r'\b', query, re.IGNORECASE):
                target_state = state
                break
        
        # Match district
        if target_state:
            districts_in_state = _master_df[_master_df['STATE'] == target_state]['DISTRICT'].unique().tolist()
            for district in districts_in_state:
                if re.search(r'\b' + re.escape(district) + r'\b', query, re.IGNORECASE):
                    target_district = district
                    break
    
    # Expand query
    expanded_terms = _expand_query(query)
    expanded_query_text = f"{query} {expanded_terms}".strip()
    
    # Search for relevant data
    candidate_results = _search_excel_chunks(expanded_query_text, year=year, target_state=target_state, target_district=target_district)
    
    if not candidate_results:
        return "I couldn't find enough relevant information in the groundwater data to answer your question."
    
    # Re-rank results
    re_ranked_results = _re_rank_chunks(expanded_query_text, candidate_results, top_k=5)
    
    if not re_ranked_results:
        return "I couldn't find enough relevant information in the groundwater data to answer your question."
    
    # Generate answer
    context_data = [res['data'] for res in re_ranked_results]
    answer = _generate_answer_from_gemini(query, context_data, year=year, target_state=target_state, target_district=target_district)
    
    return answer


def get_state_from_coordinates(lat, lng):
    """Get state name from coordinates using Gemini API."""
    if not _gemini_model:
        # Fallback to client-side state detection using the same logic
        # Reordered to prioritize states with overlapping boundaries
        state_boundaries = {
            "Maharashtra": {"min_lat": 15.6, "max_lat": 22.0, "min_lng": 72.6, "max_lng": 80.9},
            "Karnataka": {"min_lat": 11.7, "max_lat": 18.5, "min_lng": 74.1, "max_lng": 78.6},
            "Gujarat": {"min_lat": 20.1, "max_lat": 24.7, "min_lng": 68.2, "max_lng": 74.5},
            "Rajasthan": {"min_lat": 23.1, "max_lat": 30.2, "min_lng": 69.3, "max_lng": 78.2},
            "Madhya Pradesh": {"min_lat": 21.1, "max_lat": 26.9, "min_lng": 74.0, "max_lng": 82.8},
            "Uttar Pradesh": {"min_lat": 23.7, "max_lat": 31.1, "min_lng": 77.0, "max_lng": 84.7},
            "Bihar": {"min_lat": 24.2, "max_lat": 27.7, "min_lng": 83.3, "max_lng": 88.8},
            "West Bengal": {"min_lat": 21.5, "max_lat": 27.2, "min_lng": 85.5, "max_lng": 89.9},
            # Chhattisgarh before Odisha to handle overlapping regions
            "Chhattisgarh": {"min_lat": 17.8, "max_lat": 24.1, "min_lng": 80.2, "max_lng": 84.4},
            "Odisha": {"min_lat": 17.5, "max_lat": 22.5, "min_lng": 81.3, "max_lng": 87.3},
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
            "Assam": {"min_lat": 24.1, "max_lat": 28.2, "min_lng": 89.7, "max_lng": 96.0},
            "Arunachal Pradesh": {"min_lat": 26.5, "max_lat": 29.4, "min_lng": 91.6, "max_lng": 97.4},
            "Manipur": {"min_lat": 23.8, "max_lat": 25.7, "min_lng": 93.0, "max_lng": 94.8},
            "Meghalaya": {"min_lat": 25.1, "max_lat": 26.1, "min_lng": 89.8, "max_lng": 92.8},
            "Mizoram": {"min_lat": 21.9, "max_lat": 24.5, "min_lng": 92.2, "max_lng": 93.3},
            "Nagaland": {"min_lat": 25.2, "max_lat": 27.0, "min_lng": 93.0, "max_lng": 95.4},
            "Tripura": {"min_lat": 22.9, "max_lat": 24.7, "min_lng": 91.2, "max_lng": 92.3},
            "Sikkim": {"min_lat": 27.0, "max_lat": 28.2, "min_lng": 88.0, "max_lng": 88.9}
        }
        
        for state, bounds in state_boundaries.items():
            if bounds["min_lat"] <= lat <= bounds["max_lat"] and bounds["min_lng"] <= lng <= bounds["max_lng"]:
                return state
        
        return None
    
    prompt = f"""
    Given the coordinates latitude: {lat}, longitude: {lng}, determine which Indian state this location belongs to.
    
    Return ONLY the state name in English, nothing else. If the coordinates are outside India, return "Outside India".
    
    Common Indian states include: Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, Chhattisgarh, Goa, Gujarat, Haryana, Himachal Pradesh, Jharkhand, Karnataka, Kerala, Madhya Pradesh, Maharashtra, Manipur, Meghalaya, Mizoram, Nagaland, Odisha, Punjab, Rajasthan, Sikkim, Tamil Nadu, Telangana, Tripura, Uttar Pradesh, Uttarakhand, West Bengal, Delhi, Jammu and Kashmir, Ladakh, Andaman and Nicobar Islands, Chandigarh, Dadra and Nagar Haveli and Daman and Diu, Lakshadweep, Puducherry.
    
    State name:
    """
    
    try:
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
                return line
        
        return state_name
    except Exception as e:
        print(f"Error getting state from coordinates: {e}")
        return None

def analyze_location_data(state_name):
    """Analyze groundwater data for a specific state."""
    if not state_name or not _master_df is not None:
        return None
    
    # Filter data for the specific state
    state_data = _master_df[_master_df['STATE'].str.contains(state_name, case=False, na=False)]
    
    if state_data.empty:
        return None
    
    # Calculate summary statistics
    summary = {
        'districts_covered': state_data['DISTRICT'].nunique(),
        'years_covered': sorted(state_data['Assessment_Year'].unique().tolist()),
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
    
    return {
        'summary': summary,
        'key_metrics': key_metrics,
        'data_points': len(state_data)
    }

def generate_location_analysis(state_name, analysis_data):
    """Generate a comprehensive analysis of groundwater data for the location."""
    if not _gemini_model or not analysis_data:
        return f"Groundwater data analysis for {state_name}:\n\nData points: {analysis_data.get('data_points', 0)}\nDistricts covered: {analysis_data.get('summary', {}).get('districts_covered', 0)}"
    
    # Format the data for Gemini
    data_summary = []
    data_summary.append(f"State: {state_name}")
    data_summary.append(f"Total Assessment Units: {analysis_data.get('data_points', 0)}")
    data_summary.append(f"Districts Covered: {analysis_data.get('summary', {}).get('districts_covered', 0)}")
    data_summary.append(f"Years Covered: {', '.join(map(str, analysis_data.get('summary', {}).get('years_covered', [])))}")
    data_summary.append("")
    
    # Add key metrics
    key_metrics = analysis_data.get('key_metrics', {})
    if key_metrics:
        data_summary.append("Key Groundwater Metrics (averages):")
        for metric, stats in key_metrics.items():
            if stats['count'] > 0:
                data_summary.append(f"- {metric}: {stats['mean']:.2f} (range: {stats['min']:.2f} - {stats['max']:.2f})")
        data_summary.append("")
    
    context_str = "\n".join(data_summary)
    
    prompt = f"""
    You are an expert groundwater analyst. Analyze the following groundwater data for {state_name} and provide a comprehensive summary.
    
    Data Summary:
    {context_str}
    
    Please provide:
    1. A brief overview of the groundwater situation in {state_name}
    2. Key findings about groundwater availability, extraction, and recharge
    3. Any notable patterns or concerns
    4. Recommendations for groundwater management
    
    Keep the response concise but informative, focusing on the most important insights.
    """
    
    try:
        response = _gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

@app.post("/analyze-location")
def analyze_location():
    try:
        data = request.get_json(silent=True) or {}
        lat = data.get("lat")
        lng = data.get("lng")
        
        if lat is None or lng is None:
            return jsonify({"error": "Missing 'lat' or 'lng' parameters"}), 400
        
        # Initialize components
        _init_components()
        
        # Get state from coordinates
        state_name = get_state_from_coordinates(lat, lng)
        
        if not state_name:
            return jsonify({
                "error": "Could not determine state from coordinates",
                "state": None,
                "analysis": "The provided coordinates are outside India or could not be mapped to a specific state."
            })
        
        # Analyze groundwater data for the state
        analysis_data = analyze_location_data(state_name)
        
        if not analysis_data:
            return jsonify({
                "error": f"No groundwater data found for {state_name}",
                "state": state_name,
                "analysis": f"No groundwater assessment data is available for {state_name} in our database."
            })
        
        # Generate comprehensive analysis
        analysis_text = generate_location_analysis(state_name, analysis_data)
        
        return jsonify({
            "state": state_name,
            "data_points": analysis_data.get('data_points', 0),
            "summary": analysis_data.get('summary', {}),
            "analysis": analysis_text
        })
        
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.post("/ask")
def ask():
    try:
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"error": "Missing 'query'"}), 400
        answer = answer_query(query)
        return jsonify({"answer": answer})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)



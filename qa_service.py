import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', '')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
COLLECTION_NAME = 'groundwater_excel_collection'

model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

gemini = None
if GEMINI_API_KEY:
  genai.configure(api_key=GEMINI_API_KEY)
  gemini = genai.GenerativeModel('gemini-1.5-flash')

def answer_query(query: str) -> str:
  query = (query or '').strip()
  if not query:
    return 'Please provide a question.'

  try:
    vec = model.encode([query])[0]
    results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=vec, limit=20, with_payload=True)
    payloads = [hit.payload for hit in results if hit.payload]
  except Exception:
    payloads = []

  if not payloads:
    return "I couldn't find enough relevant information in the groundwater data to answer your question."

  if not gemini:
    first = payloads[0]
    return f"No LLM configured. Top match: State: {first.get('STATE')}, District: {first.get('DISTRICT')}, Unit: {first.get('ASSESSMENT UNIT')}"

  lines = []
  for item in payloads[:5]:
    lines.append(f"State: {item.get('STATE')}, District: {item.get('DISTRICT')}, Assessment Unit: {item.get('ASSESSMENT UNIT')}, Year: {item.get('Assessment_Year')}")
  context = "\n".join(lines)
  prompt = (
    "You are an expert groundwater data analyst. Provide a concise summary based only on the data below.\n"
    f"Data:\n{context}\n\nQuestion: {query}\nAnswer:"
  )
  resp = gemini.generate_content(prompt)
  return resp.text.strip()



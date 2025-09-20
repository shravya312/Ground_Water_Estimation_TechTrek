#!/usr/bin/env python3
"""
Check Qdrant collection structure and sample data
"""

from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', '')
COLLECTION_NAME = 'ingris_groundwater_collection'

try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    
    # Get collection info
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f'Collection: {COLLECTION_NAME}')
    print(f'Points count: {collection_info.points_count}')
    print(f'Vectors count: {collection_info.vectors_count}')
    print(f'Status: {collection_info.status}')
    print(f'Vector size: {collection_info.config.params.vectors.size}')
    print(f'Distance metric: {collection_info.config.params.vectors.distance}')
    
    # Get sample points
    print('\n=== Sample Points ===')
    points = client.scroll(collection_name=COLLECTION_NAME, limit=3)
    for i, point in enumerate(points[0]):
        print(f'\nPoint {i+1}:')
        print(f'  ID: {point.id}')
        if point.vector:
            print(f'  Vector size: {len(point.vector)}')
        else:
            print(f'  Vector: None')
        print(f'  Payload keys: {list(point.payload.keys())}')
        if 'combined_text' in point.payload:
            text_preview = point.payload['combined_text'][:200]
            print(f'  Combined text preview: {text_preview}...')
        if 'state' in point.payload:
            print(f'  State: {point.payload["state"]}')
        if 'district' in point.payload:
            print(f'  District: {point.payload["district"]}')
        if 'year' in point.payload:
            print(f'  Year: {point.payload["year"]}')
            
except Exception as e:
    print(f'Error: {e}')

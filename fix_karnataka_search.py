#!/usr/bin/env python3
"""
Fix for Karnataka search issue - ensure proper state filtering
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def fixed_csv_search(query_text, year=None, target_state=None, target_district=None, limit=20):
    """Fixed CSV search that properly filters by state"""
    print(f"[SEARCH] Fixed CSV search for: {query_text}")
    print(f"   Target state: {target_state}")
    print(f"   Target district: {target_district}")
    
    try:
        # Load CSV data
        df = pd.read_csv('ingris_rag_ready_complete.csv', low_memory=False)
        print(f"[DATA] Loaded {len(df)} records from CSV")
        
        # Apply filters
        filtered_df = df.copy()
        
        if target_state:
            print(f"[FILTER] Filtering by state: {target_state}")
            # Use exact match for state
            filtered_df = filtered_df[filtered_df['state'].str.upper() == target_state.upper()]
            print(f"[DATA] After state filter: {len(filtered_df)} records")
        
        if target_district:
            print(f"[FILTER] Filtering by district: {target_district}")
            filtered_df = filtered_df[filtered_df['district'].str.contains(target_district, case=False, na=False)]
            print(f"[DATA] After district filter: {len(filtered_df)} records")
        
        if year:
            print(f"[FILTER] Filtering by year: {year}")
            filtered_df = filtered_df[filtered_df['year'] == year]
            print(f"[DATA] After year filter: {len(filtered_df)} records")
        
        if len(filtered_df) == 0:
            print("[ERROR] No records found after filtering")
            return []
        
        # Show sample of filtered data
        print(f"[SAMPLE] Sample filtered records:")
        sample = filtered_df[['state', 'district', 'taluk']].head(5)
        for idx, row in sample.iterrows():
            print(f"   {row['state']} - {row['district']} - {row['taluk']}")
        
        # Perform semantic search on filtered data
        print(f"[SEARCH] Performing semantic search on {len(filtered_df)} records...")
        
        # Initialize model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get combined texts
        combined_texts = filtered_df['combined_text'].fillna('').tolist()
        
        # Generate embeddings
        data_embeddings = model.encode(combined_texts)
        query_embedding = model.encode([query_text])[0]
        
        # Calculate similarities
        similarities = []
        for i, (_, row) in enumerate(filtered_df.iterrows()):
            similarity = float(query_embedding @ data_embeddings[i])
            similarities.append((i, similarity, row))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top results
        top_results = similarities[:limit]
        
        # Format results
        results = []
        for i, (original_idx, similarity, row) in enumerate(top_results):
            result = {
                'score': similarity,
                'data': row.to_dict(),
                'state': row.get('state', 'N/A'),
                'district': row.get('district', 'N/A'),
                'year': row.get('year', 'N/A'),
                'taluk': row.get('taluk', 'N/A')
            }
            results.append(result)
        
        print(f"[SUCCESS] Found {len(results)} results")
        print(f"[RESULTS] Top 5 results:")
        for i, result in enumerate(results[:5]):
            print(f"   {i+1}. {result['state']} - {result['district']} - {result['taluk']} (score: {result['score']:.3f})")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Error in fixed CSV search: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_fixed_search():
    """Test the fixed search"""
    print("Testing Fixed Karnataka Search")
    print("=" * 50)
    
    # Test query
    query = "ground water estimation in karnataka"
    target_state = "KARNATAKA"
    
    results = fixed_csv_search(query, target_state=target_state, limit=10)
    
    if results:
        print(f"\n[SUCCESS] Search successful! Found {len(results)} results")
        
        # Verify all results are from Karnataka
        karnataka_count = sum(1 for r in results if 'KARNATAKA' in str(r['state']).upper())
        print(f"[VERIFY] Karnataka results: {karnataka_count}/{len(results)}")
        
        if karnataka_count == len(results):
            print("[SUCCESS] All results are from Karnataka!")
        else:
            print("[ERROR] Some results are not from Karnataka!")
    else:
        print("[ERROR] No results found")

if __name__ == "__main__":
    test_fixed_search()

#!/usr/bin/env python3
"""
Direct Karnataka Groundwater Search - No API needed!
"""

import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()

def search_karnataka_groundwater(query="ground water estimation in karnataka"):
    """Direct search for Karnataka groundwater data"""
    print("🔍 Karnataka Groundwater Search")
    print("=" * 50)
    print(f"Query: {query}")
    print()
    
    try:
        # Connect to Qdrant
        print("🔄 Connecting to Qdrant...")
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, check_compatibility=False)
        
        # Load embedding model
        print("🔄 Loading embedding model...")
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Create query vector
        query_vector = model.encode([query])[0].tolist()
        
        # Search for Karnataka data specifically
        print("🔍 Searching Karnataka data...")
        karnataka_filter = Filter(
            must=[
                FieldCondition(
                    key="STATE",
                    match=MatchValue(value="KARNATAKA")
                )
            ]
        )
        
        results = client.search(
            collection_name="ingris_groundwater_collection",
            query_vector=query_vector,
            query_filter=karnataka_filter,
            limit=15,
            with_payload=True,
            score_threshold=0.0  # Very low threshold to get results
        )
        
        print(f"✅ Found {len(results)} Karnataka records")
        print()
        
        if results:
            print("📊 KARNATAKA GROUNDWATER DATA:")
            print("=" * 50)
            
            for i, result in enumerate(results, 1):
                payload = result.payload
                print(f"\n{i}. {payload.get('DISTRICT', 'N/A')} District")
                print(f"   Year: {payload.get('Assessment_Year', 'N/A')}")
                print(f"   Rainfall: {payload.get('rainfall_mm', 'N/A')} mm")
                print(f"   Ground Water Recharge: {payload.get('ground_water_recharge_ham', 'N/A')} ham")
                print(f"   Stage of Extraction: {payload.get('stage_of_ground_water_extraction_', 'N/A')}%")
                print(f"   Categorization: {payload.get('categorization_of_assessment_unit', 'N/A')}")
                print(f"   Score: {result.score:.4f}")
                
                if i >= 10:  # Limit to top 10 for readability
                    print(f"\n... and {len(results) - 10} more districts")
                    break
            
            # Summary statistics
            print("\n📈 SUMMARY STATISTICS:")
            print("=" * 30)
            
            districts = [r.payload.get('DISTRICT', 'N/A') for r in results]
            years = [r.payload.get('Assessment_Year', 'N/A') for r in results]
            rainfalls = [float(r.payload.get('rainfall_mm', 0)) for r in results if r.payload.get('rainfall_mm', 'N/A') != 'N/A']
            extractions = [float(r.payload.get('stage_of_ground_water_extraction_', 0)) for r in results if r.payload.get('stage_of_ground_water_extraction_', 'N/A') != 'N/A']
            
            print(f"Total Districts: {len(set(districts))}")
            print(f"Years Covered: {sorted(set(years))}")
            if rainfalls:
                print(f"Average Rainfall: {sum(rainfalls)/len(rainfalls):.1f} mm")
                print(f"Rainfall Range: {min(rainfalls):.1f} - {max(rainfalls):.1f} mm")
            if extractions:
                print(f"Average Extraction Stage: {sum(extractions)/len(extractions):.1f}%")
                print(f"Extraction Range: {min(extractions):.1f}% - {max(extractions):.1f}%")
            
            # Categorization summary
            categories = [r.payload.get('categorization_of_assessment_unit', 'N/A') for r in results]
            from collections import Counter
            cat_counts = Counter(categories)
            print(f"\nGroundwater Status:")
            for cat, count in cat_counts.items():
                print(f"  {cat}: {count} districts")
        
        else:
            print("❌ No Karnataka data found")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def main():
    """Main function"""
    print("🚀 Starting Karnataka Groundwater Search...")
    print()
    
    # Test with the specific query
    results = search_karnataka_groundwater("ground water estimation in karnataka")
    
    if results:
        print(f"\n✅ Search completed successfully!")
        print(f"Found {len(results)} relevant Karnataka groundwater records")
    else:
        print("\n❌ Search failed - no results found")

if __name__ == "__main__":
    main()

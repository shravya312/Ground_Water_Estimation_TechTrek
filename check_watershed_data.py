#!/usr/bin/env python3
"""
Check for watershed and administrative data in Qdrant
"""

import os
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "ingris_groundwater_collection"

def check_qdrant_data():
    """Check what watershed and administrative data is available in Qdrant"""
    print("🔍 Checking Watershed & Administrative Data in Qdrant")
    print("=" * 60)
    
    try:
        # Connect to Qdrant
        print("🔄 Connecting to Qdrant...")
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            check_compatibility=False
        )
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            print(f"❌ Collection '{COLLECTION_NAME}' not found")
            print(f"Available collections: {collection_names}")
            return
        
        print(f"✅ Connected to Qdrant")
        print(f"✅ Collection '{COLLECTION_NAME}' exists")
        
        # Get collection info
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"📊 Collection has {collection_info.points_count} points")
        
        # Search for watershed-related data
        print("\n🔍 Searching for watershed and administrative data...")
        
        # Initialize embedding model
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Search queries for different administrative levels
        search_queries = [
            "watershed district category",
            "tehsil taluk administrative division",
            "block mandal village administrative",
            "watershed management administrative boundaries",
            "district tehsil block village hierarchy"
        ]
        
        for query in search_queries:
            print(f"\n🔍 Searching for: '{query}'")
            
            # Generate query embedding
            query_vector = model.encode([query])[0].tolist()
            
            # Search in Qdrant
            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=5,
                with_payload=True,
                score_threshold=0.1
            )
            
            if results:
                print(f"✅ Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"\n--- Result {i} (Score: {result.score:.3f}) ---")
                    payload = result.payload
                    
                    # Check for administrative fields
                    admin_fields = [
                        'Watershed District', 'Watershed Category', 'Tehsil', 'Taluk', 
                        'Block', 'Mandal', 'Village', 'STATE', 'DISTRICT', 'ASSESSMENT UNIT'
                    ]
                    
                    found_fields = []
                    for field in admin_fields:
                        if field in payload and payload[field] and str(payload[field]).strip() != '':
                            found_fields.append(f"{field}: {payload[field]}")
                    
                    if found_fields:
                        print("📋 Administrative Data Found:")
                        for field in found_fields:
                            print(f"  • {field}")
                    else:
                        print("❌ No administrative data found in this result")
            else:
                print("❌ No results found")
        
        # Also check the CSV file directly
        print("\n🔍 Checking CSV file for administrative data...")
        try:
            df = pd.read_csv("ingris_rag_ready_complete.csv", low_memory=False)
            print(f"✅ CSV loaded with {len(df)} rows and {len(df.columns)} columns")
            
            # Check for administrative columns
            admin_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['watershed', 'tehsil', 'taluk', 'block', 'mandal', 'village']):
                    admin_columns.append(col)
            
            if admin_columns:
                print(f"📋 Found administrative columns: {admin_columns}")
                
                # Show sample data for these columns
                for col in admin_columns[:5]:  # Show first 5 columns
                    unique_values = df[col].dropna().unique()
                    print(f"\n📊 {col}:")
                    print(f"  • Total values: {len(unique_values)}")
                    print(f"  • Sample values: {list(unique_values[:5])}")
            else:
                print("❌ No administrative columns found in CSV")
                
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def search_specific_administrative_data():
    """Search for specific administrative data patterns"""
    print("\n🔍 Searching for Specific Administrative Data Patterns")
    print("=" * 60)
    
    try:
        # Connect to Qdrant
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            check_compatibility=False
        )
        
        # Initialize embedding model
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Search for specific administrative terms
        specific_terms = [
            "Hanuru taluk",
            "Bangalore-East",
            "Bangalore North",
            "watershed category safe semi-critical critical over-exploited",
            "administrative division tehsil block"
        ]
        
        for term in specific_terms:
            print(f"\n🔍 Searching for: '{term}'")
            
            query_vector = model.encode([term])[0].tolist()
            
            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=3,
                with_payload=True,
                score_threshold=0.1
            )
            
            if results:
                print(f"✅ Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"\n--- Result {i} (Score: {result.score:.3f}) ---")
                    payload = result.payload
                    
                    # Show relevant fields
                    relevant_fields = ['STATE', 'DISTRICT', 'ASSESSMENT UNIT', 'Taluk', 'Tehsil', 'Block', 'Mandal', 'Village']
                    for field in relevant_fields:
                        if field in payload and payload[field]:
                            print(f"  • {field}: {payload[field]}")
            else:
                print("❌ No results found")
                
    except Exception as e:
        print(f"❌ Error in specific search: {e}")

def main():
    """Main function"""
    print("🧪 Watershed & Administrative Data Checker")
    print("=" * 50)
    
    check_qdrant_data()
    search_specific_administrative_data()
    
    print("\n💡 Summary:")
    print("This script checks if watershed and administrative data is available in:")
    print("1. Qdrant vector database")
    print("2. CSV source file")
    print("3. Specific administrative terms like 'Hanuru', 'Bangalore-East', etc.")

if __name__ == "__main__":
    main()

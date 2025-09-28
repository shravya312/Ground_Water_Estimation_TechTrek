#!/usr/bin/env python3
"""
Check ChromaDB collections and fix dimension issues
"""

import chromadb
from chromadb.config import Settings

def check_collections():
    """Check available ChromaDB collections"""
    print("🔍 Checking ChromaDB Collections")
    print("=" * 40)
    
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
         
        # List all collections
        collections = client.list_collections()
        print(f"📊 Found {len(collections)} collections:")
        
        for i, collection in enumerate(collections, 1):
            print(f"\n{i}. Collection: {collection.name}")
            print(f"   ID: {collection.id}")
            
            # Get collection details
            try:
                coll = client.get_collection(collection.name)
                count = coll.count()
                print(f"   Records: {count:,}")
                
                # Get a sample to check dimensions
                sample = coll.get(limit=1, include=['embeddings'])
                if sample['embeddings'] and sample['embeddings'][0]:
                    dim = len(sample['embeddings'][0])
                    print(f"   Embedding Dimension: {dim}")
                else:
                    print(f"   Embedding Dimension: Unknown")
                    
            except Exception as e:
                print(f"   Error getting details: {e}")
        
        return collections
        
    except Exception as e:
        print(f"❌ Error checking collections: {e}")
        return []

def fix_dimension_issue():
    """Fix the dimension mismatch by recreating with correct model"""
    print("\n🔧 Fixing Dimension Issue")
    print("=" * 30)
    
    try:
        from sentence_transformers import SentenceTransformer
        import pandas as pd
        
        # Load the correct embedding model
        print("🔄 Loading all-mpnet-base-v2 model (768D)...")
        model = SentenceTransformer("all-mpnet-base-v2")
        print("✅ Model loaded")
        
        # Load CSV data
        print("🔄 Loading CSV data...")
        df = pd.read_csv("ingris_rag_ready_complete.csv", skiprows=1)
        print(f"✅ Loaded {len(df)} records")
        
        # Clean column names
        df.columns = [
            'serial_number', 'state', 'district', 'island', 'watershed_district',
            'rainfall_mm', 'total_geographical_area_ha', 'ground_water_recharge_ham',
            'inflows_and_outflows_ham', 'annual_ground_water_recharge_ham',
            'environmental_flows_ham', 'annual_extractable_ground_water_resource_ham',
            'ground_water_extraction_for_all_uses_ham', 'stage_of_ground_water_extraction_',
            'categorization_of_assessment_unit', 'pre_monsoon_of_gw_trend',
            'post_monsoon_of_gw_trend', 'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham',
            'net_annual_ground_water_availability_for_future_use_ham', 'quality_tagging',
            'additional_potential_resources_under_specific_conditionsham', 'coastal_areas',
            'instorage_unconfined_ground_water_resourcesham', 'total_ground_water_availability_in_unconfined_aquifier_ham',
            'dynamic_confined_ground_water_resourcesham', 'instorage_confined_ground_water_resourcesham',
            'total_confined_ground_water_resources_ham', 'dynamic_semi_confined_ground_water_resources_ham',
            'instorage_semi_confined_ground_water_resources_ham', 'total_semiconfined_ground_water_resources_ham',
            'total_ground_water_availability_in_the_area_ham', 'source_file', 'year',
            'tehsil', 'taluk', 'block', 'valley', 'assessment_unit', 'mandal',
            'village', 'watershed_category', 'firka', 'combined_text'
        ]
        
        # Create combined text for search
        df['search_text'] = df.apply(lambda row: f"""
        State: {row['state']} | District: {row['district']} | 
        Rainfall: {row['rainfall_mm']} mm | Ground Water Recharge: {row['ground_water_recharge_ham']} ham |
        Annual Ground Water Recharge: {row['annual_ground_water_recharge_ham']} ham |
        Ground Water Extraction: {row['ground_water_extraction_for_all_uses_ham']} ham |
        Stage of Extraction: {row['stage_of_ground_water_extraction_']}% |
        Categorization: {row['categorization_of_assessment_unit']} |
        Pre-monsoon Trend: {row['pre_monsoon_of_gw_trend']} |
        Post-monsoon Trend: {row['post_monsoon_of_gw_trend']} |
        Future Availability: {row['net_annual_ground_water_availability_for_future_use_ham']} ham |
        Quality Tagging: {row['quality_tagging']} |
        Watershed Category: {row['watershed_category']} |
        Year: {row['year']} | Block: {row['block']} | Village: {row['village']}
        """, axis=1)
        
        # Connect to ChromaDB
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Delete existing collection if it exists
        try:
            client.delete_collection("ingris_groundwater_collection")
            print("🗑️ Deleted existing collection")
        except:
            pass
        
        # Create new collection with correct dimensions
        print("🔄 Creating new collection with 768D embeddings...")
        collection = client.create_collection(
            name="ingris_groundwater_collection",
            metadata={"hnsw:space": "cosine"}
        )
        print("✅ Collection created")
        
        # Upload data in batches
        batch_size = 1000
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        print(f"🔄 Uploading {len(df)} records in {total_batches} batches...")
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_df)} records)...")
            
            # Prepare documents
            documents = batch_df['search_text'].tolist()
            metadatas = []
            ids = []
            
            for idx, row in batch_df.iterrows():
                metadata = {
                    'serial_number': str(row['serial_number']),
                    'state': str(row['state']),
                    'district': str(row['district']),
                    'year': str(row['year']),
                    'rainfall_mm': str(row['rainfall_mm']),
                    'ground_water_recharge_ham': str(row['ground_water_recharge_ham']),
                    'annual_ground_water_recharge_ham': str(row['annual_ground_water_recharge_ham']),
                    'ground_water_extraction_for_all_uses_ham': str(row['ground_water_extraction_for_all_uses_ham']),
                    'stage_of_ground_water_extraction_': str(row['stage_of_ground_water_extraction_']),
                    'categorization_of_assessment_unit': str(row['categorization_of_assessment_unit']),
                    'pre_monsoon_of_gw_trend': str(row['pre_monsoon_of_gw_trend']),
                    'post_monsoon_of_gw_trend': str(row['post_monsoon_of_gw_trend']),
                    'net_annual_ground_water_availability_for_future_use_ham': str(row['net_annual_ground_water_availability_for_future_use_ham']),
                    'quality_tagging': str(row['quality_tagging']),
                    'watershed_category': str(row['watershed_category']),
                    'block': str(row['block']),
                    'village': str(row['village'])
                }
                metadatas.append(metadata)
                # Use unique ID based on index to avoid duplicates
                ids.append(f"record_{i + idx}")
            
            # Generate embeddings
            print(f"🔄 Generating embeddings for batch {batch_num}...")
            embeddings = model.encode(documents).tolist()
            
            # Add to collection
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            print(f"✅ Batch {batch_num} uploaded")
        
        # Verify upload
        count = collection.count()
        print(f"\n🎉 Upload completed! Collection now has {count:,} records")
        
        # Test search
        print("\n🔍 Testing search...")
        test_query = "groundwater estimation in Karnataka"
        query_embedding = model.encode([test_query])[0].tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        
        if results['documents'] and results['documents'][0]:
            print(f"✅ Search test successful - found {len(results['documents'][0])} results")
            print(f"   First result: {results['documents'][0][0][:100]}...")
        else:
            print("❌ Search test failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing dimension issue: {e}")
        return False

def main():
    """Main function"""
    print("🔧 ChromaDB Collection Checker and Fixer")
    print("=" * 50)
    
    # Check existing collections
    collections = check_collections()
    
    # Check if we need to fix dimensions
    needs_fix = True
    for collection in collections:
        if collection.name == "ingris_groundwater_collection":
            try:
                client = chromadb.PersistentClient(path="./chroma_db")
                coll = client.get_collection(collection.name)
                sample = coll.get(limit=1, include=['embeddings'])
                if sample['embeddings'] and sample['embeddings'][0]:
                    dim = len(sample['embeddings'][0])
                    if dim == 768:
                        print(f"\n✅ Collection has correct 768D embeddings")
                        needs_fix = False
                    else:
                        print(f"\n⚠️ Collection has {dim}D embeddings, need 768D")
            except:
                pass
    
    if needs_fix:
        print(f"\n🔧 Fixing dimension issue...")
        if fix_dimension_issue():
            print("\n🎉 Dimension issue fixed!")
        else:
            print("\n❌ Failed to fix dimension issue")
    else:
        print("\n✅ No fixes needed")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple script to upload master CSV data using existing main.py functions
"""

import main2
import pandas as pd

def upload_master_csv_simple():
    """Upload master CSV data using existing functions."""
    print("🚀 Uploading Master CSV Data (Simple Approach)")
    print("=" * 60)
    
    # Initialize components
    main2._init_components()
    
    # Clear existing collection
    print("🗑️ Clearing existing collection...")
    try:
        main2._qdrant_client.delete_collection("groundwater_excel_collection")
        print("✅ Collection deleted")
    except:
        print("ℹ️ Collection doesn't exist or already deleted")
    
    # Create new collection
    print("🆕 Creating new collection...")
    from qdrant_client.http.models import Distance, VectorParams
    main2._qdrant_client.create_collection(
        collection_name="groundwater_excel_collection",
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )
    print("✅ Collection created")
    
    # Read master CSV
    print("📄 Reading master CSV...")
    df = pd.read_csv('master_groundwater_data.csv', low_memory=False)
    print(f"📊 Found {len(df)} records in master CSV")
    
    # Show sample of data
    print("\n📋 Sample Data:")
    print(f"  Columns: {list(df.columns)[:10]}...")
    print(f"  Sample STATE: {df['STATE'].iloc[0] if 'STATE' in df.columns else 'N/A'}")
    print(f"  Sample DISTRICT: {df['DISTRICT'].iloc[0] if 'DISTRICT' in df.columns else 'N/A'}")
    print(f"  Sample Year: {df['Assessment_Year'].iloc[0] if 'Assessment_Year' in df.columns else 'N/A'}")
    
    # Check for Karnataka
    karnataka_data = df[df['STATE'] == 'KARNATAKA'] if 'STATE' in df.columns else pd.DataFrame()
    print(f"\n🔍 Karnataka Records: {len(karnataka_data)}")
    
    if len(karnataka_data) > 0:
        print(f"📋 Karnataka Districts: {sorted(karnataka_data['DISTRICT'].unique())}")
    
    # Use existing upload function from main.py
    print("\n🔄 Uploading data using existing functions...")
    
    # Create combined text for each row
    def create_combined_text(row):
        parts = []
        for col, value in row.items():
            if pd.notna(value) and value != '' and col not in ['S.No']:
                parts.append(f"{col}: {value}")
        return " | ".join(parts)
    
    df['combined_text'] = df.apply(create_combined_text, axis=1)
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['combined_text'])
    final_count = len(df)
    print(f"📊 Deduplication: {initial_count} → {final_count} records ({initial_count - final_count} duplicates removed)")
    
    # Upload in batches using existing function
    batch_size = 50
    uploaded = 0
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        
        try:
            # Use the existing upload function from main.py
            main2.upload_dataframe_to_qdrant(batch_df, main2._model, main2._qdrant_client)
            uploaded += len(batch_df)
            print(f"   📤 Batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}: {len(batch_df)} records uploaded")
        except Exception as e:
            print(f"   ❌ Error uploading batch {i//batch_size + 1}: {e}")
            continue
    
    # Verify upload
    print("\n🔄 Verifying upload...")
    collection_info = main2._qdrant_client.get_collection("groundwater_excel_collection")
    print(f"✅ Total points in collection: {collection_info.points_count}")
    
    print("\n🎉 Master CSV upload completed!")
    print(f"📈 Uploaded {uploaded} records from master CSV")
    print("🔍 Your RAG system now has comprehensive groundwater data for all states!")

if __name__ == "__main__":
    upload_master_csv_simple()

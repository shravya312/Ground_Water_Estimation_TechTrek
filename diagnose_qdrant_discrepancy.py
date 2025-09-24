#!/usr/bin/env python3
"""
Diagnostic script to identify and fix the Qdrant collection discrepancy.
This script will:
1. Check the current collection status
2. Identify duplicate combined_text values in the source data
3. Fix the missing records by re-uploading with proper deduplication
"""

import os
import pandas as pd
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "groundwater_excel_collection"
VECTOR_SIZE = 384

def initialize_components():
    """Initialize Qdrant client and model."""
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        return qdrant_client, model
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return None, None

def get_collection_info(qdrant_client):
    """Get detailed collection information."""
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        return {
            'points_count': collection_info.points_count,
            'vectors_count': collection_info.vectors_count,
            'status': collection_info.status,
            'config': collection_info.config
        }
    except Exception as e:
        print(f"❌ Error getting collection info: {e}")
        return None

def create_detailed_combined_text(row):
    """Generates a detailed combined text string for a DataFrame row."""
    parts = []
    for col, value in row.items():
        if pd.notna(value) and value != '' and col not in ['S.No']:
            parts.append(f"{col}: {value}")
    return " | ".join(parts)

def analyze_duplicates_in_data(df):
    """Analyze duplicate combined_text values in the source data."""
    print("🔍 Analyzing duplicate combined_text values...")
    
    # Create combined_text column if it doesn't exist
    if 'combined_text' not in df.columns:
        print("📝 Creating combined_text column...")
        df['combined_text'] = df.apply(create_detailed_combined_text, axis=1)
    
    # Check for duplicates
    duplicate_mask = df['combined_text'].duplicated(keep=False)
    duplicates = df[duplicate_mask]
    
    if len(duplicates) > 0:
        print(f"⚠️ Found {len(duplicates)} rows with duplicate combined_text values")
        
        # Group by combined_text to see how many duplicates per text
        duplicate_groups = duplicates.groupby('combined_text').size()
        print(f"📊 Duplicate groups: {len(duplicate_groups)}")
        print(f"📊 Max duplicates per text: {duplicate_groups.max()}")
        print(f"📊 Average duplicates per text: {duplicate_groups.mean():.2f}")
        
        # Show some examples
        print("\n📋 Sample duplicate texts:")
        for i, (text, count) in enumerate(duplicate_groups.head(5).items()):
            print(f"  {i+1}. Count: {count}, Text: {text[:100]}...")
        
        return len(duplicates), duplicate_groups
    else:
        print("✅ No duplicate combined_text values found")
        return 0, None

def check_missing_records(qdrant_client, df):
    """Check which records are missing from Qdrant."""
    print("🔍 Checking for missing records...")
    
    try:
        # Get all points from Qdrant
        all_points = []
        offset = None
        limit = 1000
        
        while True:
            if offset is None:
                points = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=limit,
                    with_payload=True
                )
            else:
                points = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=limit,
                    offset=offset,
                    with_payload=True
                )
            
            if not points[0]:  # No more points
                break
                
            all_points.extend(points[0])
            offset = points[1]
            
            if offset is None:  # No more points
                break
        
        print(f"📊 Found {len(all_points)} points in Qdrant collection")
        
        # Extract combined_text values from Qdrant
        qdrant_texts = set()
        for point in all_points:
            if 'combined_text' in point.payload:
                qdrant_texts.add(point.payload['combined_text'])
        
        # Check which source texts are missing
        source_texts = set(df['combined_text'].tolist())
        missing_texts = source_texts - qdrant_texts
        
        print(f"📊 Source data has {len(source_texts)} unique combined_text values")
        print(f"📊 Qdrant has {len(qdrant_texts)} unique combined_text values")
        print(f"❌ Missing {len(missing_texts)} unique combined_text values")
        
        if missing_texts:
            print("\n📋 Sample missing texts:")
            for i, text in enumerate(list(missing_texts)[:5]):
                print(f"  {i+1}. {text[:100]}...")
        
        return missing_texts, len(all_points)
        
    except Exception as e:
        print(f"❌ Error checking missing records: {e}")
        return None, 0

def fix_missing_records(qdrant_client, model, df, missing_texts):
    """Fix missing records by uploading them with unique IDs."""
    if not missing_texts:
        print("✅ No missing records to fix")
        return True
    
    print(f"🔧 Fixing {len(missing_texts)} missing records...")
    
    try:
        # Filter dataframe to only missing records
        missing_df = df[df['combined_text'].isin(missing_texts)]
        print(f"📊 Found {len(missing_df)} rows to upload")
        
        # Generate embeddings
        texts = missing_df['combined_text'].tolist()
        embeddings = model.encode(texts)
        
        # Create points with unique IDs (using row index + timestamp)
        points = []
        for i, (index, row) in enumerate(missing_df.iterrows()):
            # Use a combination of index and text hash for unique ID
            unique_id = f"fix_{index}_{uuid.uuid4().hex[:8]}"
            
            point = PointStruct(
                id=unique_id,
                vector=embeddings[i].tolist(),
                payload=row.to_dict()
            )
            points.append(point)
        
        # Upload in batches
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True
            )
            total_uploaded += len(batch)
            print(f"📤 Uploaded batch {i//batch_size + 1}: {total_uploaded}/{len(points)} records")
        
        print(f"✅ Successfully uploaded {total_uploaded} missing records")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing missing records: {e}")
        return False

def main():
    """Main diagnostic and fix function."""
    print("🔍 Qdrant Collection Discrepancy Diagnostic Tool")
    print("=" * 50)
    
    # Initialize components
    qdrant_client, model = initialize_components()
    if not qdrant_client or not model:
        return
    
    # Get collection info
    print("\n📊 Collection Information:")
    collection_info = get_collection_info(qdrant_client)
    if collection_info:
        print(f"  Points count: {collection_info['points_count']}")
        print(f"  Vectors count: {collection_info['vectors_count']}")
        print(f"  Status: {collection_info['status']}")
    
    # Load source data
    print("\n📁 Loading source data...")
    try:
        # Try to load the master CSV file
        csv_file = "master_groundwater_data.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, low_memory=False)
            print(f"✅ Loaded {len(df)} rows from {csv_file}")
        else:
            print(f"❌ Source file not found: {csv_file}")
            return
    except Exception as e:
        print(f"❌ Error loading source data: {e}")
        return
    
    # Analyze duplicates
    print("\n🔍 Step 1: Analyzing duplicates in source data")
    duplicate_count, duplicate_groups = analyze_duplicates_in_data(df)
    
    # Check missing records
    print("\n🔍 Step 2: Checking for missing records")
    missing_texts, qdrant_count = check_missing_records(qdrant_client, df)
    
    # Summary
    print("\n📊 Summary:")
    print(f"  Source data rows: {len(df)}")
    print(f"  Source unique texts: {len(set(df['combined_text'].tolist()))}")
    print(f"  Qdrant points: {qdrant_count}")
    print(f"  Duplicate rows in source: {duplicate_count}")
    print(f"  Missing unique texts: {len(missing_texts) if missing_texts else 0}")
    
    # Calculate expected vs actual
    expected_unique = len(set(df['combined_text'].tolist()))
    actual_unique = qdrant_count
    discrepancy = expected_unique - actual_unique
    
    print(f"\n🎯 Discrepancy Analysis:")
    print(f"  Expected unique records: {expected_unique}")
    print(f"  Actual records in Qdrant: {actual_unique}")
    print(f"  Discrepancy: {discrepancy}")
    
    if discrepancy > 0:
        print(f"\n🔧 Fixing {discrepancy} missing records...")
        if fix_missing_records(qdrant_client, model, df, missing_texts):
            # Verify fix
            print("\n✅ Verification:")
            new_info = get_collection_info(qdrant_client)
            if new_info:
                print(f"  New points count: {new_info['points_count']}")
                print(f"  Records added: {new_info['points_count'] - qdrant_count}")
        else:
            print("❌ Failed to fix missing records")
    else:
        print("✅ No discrepancy found - collection is up to date!")

if __name__ == "__main__":
    main()

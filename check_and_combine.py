#!/usr/bin/env python3
"""
Check current data and prepare for upload
"""

import pandas as pd
import os

def check_current_data():
    """Check the current state of our data."""
    print("🔍 Checking Current Data Status")
    print("=" * 50)
    
    # Check INGRIS data
    if os.path.exists('ingris_rag_ready.csv'):
        df_ingris = pd.read_csv('ingris_rag_ready.csv', low_memory=False)
        print(f"📊 INGRIS CSV:")
        print(f"   Total records: {len(df_ingris)}")
        print(f"   States: {df_ingris['state'].nunique()}")
        print(f"   Top 5 states:")
        for state, count in df_ingris['state'].value_counts().head().items():
            print(f"     {state}: {count}")
        
        # Check if we have the required columns
        required_cols = ['state', 'district', 'year', 'source_file']
        missing_cols = [col for col in required_cols if col not in df_ingris.columns]
        if missing_cols:
            print(f"   ⚠️ Missing columns: {missing_cols}")
        else:
            print(f"   ✅ All required columns present")
        
        return df_ingris
    else:
        print("❌ INGRIS CSV not found!")
        return None

def prepare_for_upload(df):
    """Prepare the data for upload to Qdrant."""
    print(f"\n🔄 Preparing Data for Upload")
    print("=" * 50)
    
    if df is None:
        print("❌ No data to prepare")
        return None
    
    # Create combined text for RAG
    def create_combined_text(row):
        parts = []
        for col, value in row.items():
            if pd.notna(value) and value != '' and col not in ['source_file', 'combined_text']:
                parts.append(f"{col}: {value}")
        return " | ".join(parts)
    
    print("🔄 Creating combined text...")
    df['combined_text'] = df.apply(create_combined_text, axis=1)
    
    # Remove duplicates based on combined_text
    initial_count = len(df)
    df = df.drop_duplicates(subset=['combined_text'])
    final_count = len(df)
    print(f"📊 Deduplication: {initial_count} → {final_count} records ({initial_count - final_count} duplicates removed)")
    
    # Save the final upload-ready CSV
    output_file = "ingris_upload_ready.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Upload-ready CSV saved: {output_file}")
    
    # Also update the original file
    df.to_csv("ingris_rag_ready.csv", index=False)
    print("✅ Updated original ingris_rag_ready.csv")
    
    print(f"\n📊 Final Data Summary:")
    print(f"   Total records: {len(df)}")
    print(f"   States: {df['state'].nunique()}")
    print(f"   Years: {df['year'].nunique()}")
    print(f"   Columns: {len(df.columns)}")
    
    return df

def check_upload_requirements():
    """Check if we're ready for upload."""
    print(f"\n🔍 Upload Requirements Check")
    print("=" * 50)
    
    # Check if smart_upload_tracker_structured.py exists
    if os.path.exists('smart_upload_tracker_structured.py'):
        print("✅ smart_upload_tracker_structured.py found")
    else:
        print("❌ smart_upload_tracker_structured.py not found")
    
    # Check if ingris_rag_ready.csv exists and has data
    if os.path.exists('ingris_rag_ready.csv'):
        df = pd.read_csv('ingris_rag_ready.csv', low_memory=False)
        if len(df) > 0:
            print(f"✅ ingris_rag_ready.csv ready with {len(df)} records")
            return True
        else:
            print("❌ ingris_rag_ready.csv is empty")
            return False
    else:
        print("❌ ingris_rag_ready.csv not found")
        return False

if __name__ == "__main__":
    # Check current data
    df = check_current_data()
    
    if df is not None:
        # Prepare for upload
        df_final = prepare_for_upload(df)
        
        # Check upload requirements
        ready = check_upload_requirements()
        
        if ready:
            print(f"\n🎉 READY FOR UPLOAD!")
            print("=" * 50)
            print("✅ All data is prepared and ready")
            print("✅ smart_upload_tracker_structured.py is available")
            print("✅ ingris_rag_ready.csv contains all state data")
            print("\n🚀 You can now start the upload process!")
        else:
            print(f"\n❌ NOT READY FOR UPLOAD")
            print("Please fix the issues above before proceeding.")

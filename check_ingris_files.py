#!/usr/bin/env python3
"""
Check all INGRIS files to find the missing data
"""

import pandas as pd
import os

def check_ingris_files():
    """Check all INGRIS files to find where the 1 lakh records went."""
    print("🔍 Checking All INGRIS Files")
    print("=" * 50)
    
    files_to_check = [
        'ingris_rag_ready.csv',
        'ingris_rag_ready_final.csv', 
        'ingris_rag_ready_final_fixed.csv',
        'ingris_upload_ready.csv'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file, low_memory=False)
                print(f"📄 {file}: {len(df):,} records")
                
                # Check if it has the original 1 lakh records
                if len(df) > 90000:  # Close to 1 lakh
                    print(f"   🎯 This might be the file with ~1 lakh records!")
                    print(f"   States: {df['state'].nunique() if 'state' in df.columns else 'N/A'}")
                    
            except Exception as e:
                print(f"❌ {file}: Error reading - {e}")
        else:
            print(f"❌ {file}: Not found")
    
    # Check if there are any other CSV files that might contain the data
    print(f"\n🔍 Looking for other CSV files...")
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            if len(df) > 50000:  # Large files
                print(f"📄 {csv_file}: {len(df):,} records")
        except:
            pass

if __name__ == "__main__":
    check_ingris_files()

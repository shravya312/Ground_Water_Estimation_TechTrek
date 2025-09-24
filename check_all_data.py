#!/usr/bin/env python3
"""
Check all available data files
"""

import pandas as pd
import os

def check_all_data():
    """Check all available data files."""
    print("🔍 Checking All Available Data")
    print("=" * 50)
    
    files_to_check = [
        'ingris_rag_ready.csv',
        'master_groundwater_data.csv', 
        'ingris_upload_ready.csv',
        'ingris_rag_ready_final.csv',
        'ingris_rag_ready_final_fixed.csv'
    ]
    
    total_records = 0
    
    for file in files_to_check:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file, low_memory=False)
                print(f"📄 {file}: {len(df):,} records")
                total_records += len(df)
                
                # Show state info if available
                if 'state' in df.columns:
                    states = df['state'].nunique()
                    print(f"   States: {states}")
                    if states > 0:
                        top_states = df['state'].value_counts().head(3)
                        print(f"   Top states: {dict(top_states)}")
                
            except Exception as e:
                print(f"❌ {file}: Error reading - {e}")
        else:
            print(f"❌ {file}: Not found")
    
    print(f"\n📊 Total Records Across All Files: {total_records:,}")
    
    # Check if we have the master CSV with 1 lakh records
    if os.path.exists('master_groundwater_data.csv'):
        df_master = pd.read_csv('master_groundwater_data.csv', low_memory=False)
        print(f"\n🎯 Master CSV Analysis:")
        print(f"   Records: {len(df_master):,}")
        if 'STATE' in df_master.columns:
            states = df_master['STATE'].nunique()
            print(f"   States: {states}")
            print(f"   Top 5 states:")
            for state, count in df_master['STATE'].value_counts().head().items():
                print(f"     {state}: {count:,}")

if __name__ == "__main__":
    check_all_data()

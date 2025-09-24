#!/usr/bin/env python3
"""
Fix states in the full ~1 lakh dataset
"""

import pandas as pd
import os

def fix_full_dataset_states():
    """Fix states in the full dataset using our extraction logic."""
    print("🔧 Fixing States in Full ~1 Lakh Dataset")
    print("=" * 60)
    
    # Load the full dataset
    if os.path.exists('ingris_clean_data.csv'):
        print("📄 Loading full dataset...")
        df = pd.read_csv('ingris_clean_data.csv', low_memory=False)
        print(f"   Total records: {len(df):,}")
        
        # Check current state distribution
        if 'state' in df.columns:
            print(f"   Current states: {df['state'].nunique()}")
            print(f"   Unknown states: {df['state'].value_counts().get('Unknown', 0):,}")
        
        # Apply state extraction logic
        print("🔄 Applying state extraction logic...")
        
        # Group by source_file and extract state from filename
        if 'source_file' in df.columns:
            print("   Extracting states from source files...")
            
            # Create a mapping of source files to states
            file_to_state = {}
            
            # Get unique source files
            unique_files = df['source_file'].unique()
            print(f"   Processing {len(unique_files)} unique source files...")
            
            for file in unique_files:
                if pd.notna(file):
                    # Extract state from filename using our logic
                    state = extract_state_from_filename(file)
                    if state != "Unknown":
                        file_to_state[file] = state
                        print(f"     {file} → {state}")
            
            print(f"   Found states for {len(file_to_state)} files")
            
            # Apply the mapping
            df['state'] = df['source_file'].map(file_to_state).fillna('Unknown')
            
            # Check results
            states = df['state'].nunique()
            unknown_count = df['state'].value_counts().get('Unknown', 0)
            valid_count = len(df) - unknown_count
            
            print(f"\n📊 State Extraction Results:")
            print(f"   Total states: {states}")
            print(f"   Valid states: {valid_count:,} records")
            print(f"   Unknown states: {unknown_count:,} records")
            print(f"   Success rate: {(valid_count/len(df)*100):.1f}%")
            
            if valid_count > 0:
                print(f"\n📋 Top 10 states found:")
                for state, count in df['state'].value_counts().head(10).items():
                    if state != 'Unknown':
                        print(f"     {state}: {count:,}")
        
        # Create combined text for RAG
        print("🔄 Creating combined text...")
        def create_combined_text(row):
            parts = []
            for col, value in row.items():
                if pd.notna(value) and value != '' and col not in ['combined_text', 'source_file']:
                    parts.append(f"{col}: {value}")
            return " | ".join(parts)
        
        df['combined_text'] = df.apply(create_combined_text, axis=1)
        
        # Remove duplicates
        print("🔄 Removing duplicates...")
        initial_count = len(df)
        df = df.drop_duplicates(subset=['combined_text'])
        final_count = len(df)
        print(f"📊 Deduplication: {initial_count:,} → {final_count:,} records ({initial_count - final_count:,} duplicates removed)")
        
        # Save the fixed dataset
        output_file = "ingris_rag_ready.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Fixed dataset saved: {output_file}")
        
        print(f"\n📊 Final Dataset Summary:")
        print(f"   Total records: {len(df):,}")
        if 'state' in df.columns:
            states = df['state'].nunique()
            print(f"   States: {states}")
            print(f"   Top 10 states:")
            for state, count in df['state'].value_counts().head(10).items():
                print(f"     {state}: {count:,}")
        
        return df
    else:
        print("❌ Full dataset not found!")
        return None

def extract_state_from_filename(filename):
    """Extract state name from filename."""
    import re
    
    # Common state mappings based on filename patterns
    state_mappings = {
        'ANDHRA PRADESH': ['andhra', 'ap'],
        'TELANGANA': ['telangana', 'tg'],
        'TAMILNADU': ['tamil', 'tn'],
        'BIHAR': ['bihar', 'br'],
        'MADHYA PRADESH': ['madhya', 'mp'],
        'ODISHA': ['odisha', 'or'],
        'RAJASTHAN': ['rajasthan', 'rj'],
        'UTTAR PRADESH': ['uttar', 'up'],
        'JHARKHAND': ['jharkhand', 'jh'],
        'GUJARAT': ['gujarat', 'gj'],
        'MAHARASHTRA': ['maharashtra', 'mh'],
        'ASSAM': ['assam', 'as'],
        'KARNATAKA': ['karnataka', 'ka'],
        'KERALA': ['kerala', 'kl'],
        'PUNJAB': ['punjab', 'pb']
    }
    
    filename_lower = filename.lower()
    
    for state, patterns in state_mappings.items():
        for pattern in patterns:
            if pattern in filename_lower:
                return state
    
    return "Unknown"

if __name__ == "__main__":
    df = fix_full_dataset_states()
    if df is not None:
        print(f"\n🎉 Full dataset with fixed states ready!")
        print(f"📈 Total records: {len(df):,}")
        print("🚀 You can now upload the complete dataset!")

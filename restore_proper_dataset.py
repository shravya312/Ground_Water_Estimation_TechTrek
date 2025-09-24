#!/usr/bin/env python3
"""
Restore the properly extracted dataset with all states
"""

import pandas as pd
import os

def restore_proper_dataset():
    """Restore the dataset that has proper state extraction."""
    print("🔄 Restoring Properly Extracted Dataset")
    print("=" * 50)
    
    # Use the dataset that has proper state extraction
    if os.path.exists('ingris_rag_ready_final_fixed.csv'):
        print("📄 Loading properly extracted dataset...")
        df = pd.read_csv('ingris_rag_ready_final_fixed.csv', low_memory=False)
        print(f"   Total records: {len(df):,}")
        print(f"   States: {df['state'].nunique()}")
        
        # Check state distribution
        print(f"\n📋 Top 10 states:")
        for state, count in df['state'].value_counts().head(10).items():
            print(f"     {state}: {count:,}")
        
        # Check for unknown states
        unknown_count = df['state'].value_counts().get('Unknown', 0)
        valid_count = len(df) - unknown_count
        print(f"\n📊 State Analysis:")
        print(f"   Valid states: {valid_count:,} records")
        print(f"   Unknown states: {unknown_count:,} records")
        print(f"   Success rate: {(valid_count/len(df)*100):.1f}%")
        
        # Create combined text if not exists
        if 'combined_text' not in df.columns:
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
        
        # Save as the main upload file
        output_file = "ingris_rag_ready.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Dataset saved: {output_file}")
        
        print(f"\n📊 Final Dataset Summary:")
        print(f"   Total records: {len(df):,}")
        print(f"   States: {df['state'].nunique()}")
        print(f"   Years: {df['year'].nunique() if 'year' in df.columns else 'N/A'}")
        
        return df
    else:
        print("❌ Properly extracted dataset not found!")
        return None

if __name__ == "__main__":
    df = restore_proper_dataset()
    if df is not None:
        print(f"\n🎉 Properly extracted dataset restored!")
        print(f"📈 Total records ready for upload: {len(df):,}")
        print("🚀 You can now upload the dataset with proper state names!")

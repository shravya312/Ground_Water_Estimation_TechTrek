#!/usr/bin/env python3
"""
Restore the full ~1 lakh dataset
"""

import pandas as pd
import os

def restore_full_dataset():
    """Restore the full ~1 lakh dataset."""
    print("🔄 Restoring Full ~1 Lakh Dataset")
    print("=" * 50)
    
    # Check the original clean data file
    if os.path.exists('ingris_clean_data.csv'):
        print("📄 Loading original clean data...")
        df_original = pd.read_csv('ingris_clean_data.csv', low_memory=False)
        print(f"   Original records: {len(df_original):,}")
        
        if 'state' in df_original.columns:
            states = df_original['state'].nunique()
            print(f"   States: {states}")
            print(f"   Top 5 states:")
            for state, count in df_original['state'].value_counts().head().items():
                print(f"     {state}: {count:,}")
        
        # Check if this has the proper state names
        unknown_count = df_original['state'].value_counts().get('Unknown', 0)
        print(f"   Unknown states: {unknown_count:,}")
        
        if unknown_count > 0:
            print("   ⚠️ This file still has 'Unknown' states")
            # Use the fixed version if available
            if os.path.exists('ingris_rag_ready_fixed.csv'):
                print("📄 Loading fixed version...")
                df_fixed = pd.read_csv('ingris_rag_ready_fixed.csv', low_memory=False)
                print(f"   Fixed records: {len(df_fixed):,}")
                
                if 'state' in df_fixed.columns:
                    states = df_fixed['state'].nunique()
                    print(f"   States: {states}")
                    unknown_count = df_fixed['state'].value_counts().get('Unknown', 0)
                    print(f"   Unknown states: {unknown_count:,}")
                    
                    if unknown_count == 0:
                        print("   ✅ This file has all proper state names!")
                        # Use this as the main dataset
                        df_final = df_fixed
                    else:
                        print("   ⚠️ This file also has 'Unknown' states")
                        df_final = df_original
                else:
                    df_final = df_original
            else:
                df_final = df_original
        else:
            print("   ✅ This file has all proper state names!")
            df_final = df_original
        
        # Create combined text for RAG
        print("🔄 Creating combined text...")
        def create_combined_text(row):
            parts = []
            for col, value in row.items():
                if pd.notna(value) and value != '' and col not in ['combined_text', 'source_file']:
                    parts.append(f"{col}: {value}")
            return " | ".join(parts)
        
        df_final['combined_text'] = df_final.apply(create_combined_text, axis=1)
        
        # Remove duplicates
        print("🔄 Removing duplicates...")
        initial_count = len(df_final)
        df_final = df_final.drop_duplicates(subset=['combined_text'])
        final_count = len(df_final)
        print(f"📊 Deduplication: {initial_count:,} → {final_count:,} records ({initial_count - final_count:,} duplicates removed)")
        
        # Save as the main upload file
        output_file = "ingris_rag_ready.csv"
        df_final.to_csv(output_file, index=False)
        print(f"✅ Full dataset saved: {output_file}")
        
        print(f"\n📊 Final Dataset Summary:")
        print(f"   Total records: {len(df_final):,}")
        if 'state' in df_final.columns:
            states = df_final['state'].nunique()
            print(f"   States: {states}")
            print(f"   Top 10 states:")
            for state, count in df_final['state'].value_counts().head(10).items():
                print(f"     {state}: {count:,}")
        
        return df_final
    else:
        print("❌ Original clean data file not found!")
        return None

if __name__ == "__main__":
    df = restore_full_dataset()
    if df is not None:
        print(f"\n🎉 Full dataset restored!")
        print(f"📈 Total records ready for upload: {len(df):,}")
        print("🚀 You can now upload the complete dataset!")

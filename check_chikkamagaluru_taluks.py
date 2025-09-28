#!/usr/bin/env python3
"""
Check Chikkamagaluru taluks specifically
"""

import pandas as pd

def check_chikkamagaluru_taluks():
    """Check what taluks are available in Chikkamagaluru district"""
    print("Checking Chikkamagaluru Taluks")
    print("=" * 50)
    
    try:
        # Load the data
        df = pd.read_csv('ingris_rag_ready_complete.csv', low_memory=False)
        print(f"Loaded {len(df)} records from ingris_rag_ready_complete.csv")
        
        # Filter for Chikkamagaluru
        chikkamagaluru_data = df[(df['state'] == 'KARNATAKA') & (df['district'] == 'Chikkamagaluru')]
        print(f"Found {len(chikkamagaluru_data)} records for Chikkamagaluru district")
        
        # Get taluks
        taluks = chikkamagaluru_data['taluk'].dropna().unique()
        taluks = [t for t in taluks if str(t).strip() != '']
        
        print(f"\nChikkamagaluru Taluks ({len(taluks)}):")
        print("-" * 40)
        for i, taluk in enumerate(sorted(taluks), 1):
            print(f"{i:2d}. {taluk}")
        
        # Check for the specific taluks mentioned in the user's report
        mentioned_taluks = ['Narasimharajapura', 'Koppa', 'Ajjampura']
        print(f"\nChecking for mentioned taluks:")
        print("-" * 40)
        for taluk in mentioned_taluks:
            if taluk in taluks:
                print(f"[FOUND] {taluk}")
            else:
                print(f"[NOT FOUND] {taluk}")
        
        # Show sample data for each taluk
        print(f"\nSample data for each taluk:")
        print("-" * 40)
        for taluk in sorted(taluks)[:5]:  # Show first 5 taluks
            taluk_data = chikkamagaluru_data[chikkamagaluru_data['taluk'] == taluk]
            if len(taluk_data) > 0:
                sample = taluk_data.iloc[0]
                print(f"\n{taluk}:")
                print(f"  District: {sample['district']}")
                print(f"  Assessment Unit: {sample['assessment_unit']}")
                print(f"  Stage of Extraction: {sample['stage_of_ground_water_extraction_']}")
                print(f"  Categorization: {sample['categorization_of_assessment_unit']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_chikkamagaluru_taluks()

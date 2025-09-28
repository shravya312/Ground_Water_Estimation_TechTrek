#!/usr/bin/env python3
"""
Search for Ooty in groundwater data
"""

import pandas as pd
import re

def search_ooty_in_data():
    """Search for Ooty/Udhagamandalam in the groundwater data"""
    print("Searching for Ooty in groundwater data...")
    print("=" * 50)
    
    try:
        # Load the main dataset
        print("Loading master_groundwater_data.csv...")
        df = pd.read_csv('master_groundwater_data.csv', low_memory=False)
        print(f"Total records: {len(df)}")
        
        # Search for Ooty variations
        ooty_variations = [
            'ooty', 'ootacamund', 'udhagamandalam', 'nilgiris', 'nilgiri'
        ]
        
        print("\nSearching for Ooty variations...")
        found_records = []
        
        for variation in ooty_variations:
            # Search in STATE column
            state_matches = df[df['STATE'].astype(str).str.contains(variation, case=False, na=False)]
            if not state_matches.empty:
                print(f"\nFound in STATE column for '{variation}':")
                for idx, row in state_matches.iterrows():
                    print(f"  State: {row['STATE']}, District: {row.get('DISTRICT', 'N/A')}")
                    found_records.append(row)
            
            # Search in DISTRICT column
            district_matches = df[df['DISTRICT'].astype(str).str.contains(variation, case=False, na=False)]
            if not district_matches.empty:
                print(f"\nFound in DISTRICT column for '{variation}':")
                for idx, row in district_matches.iterrows():
                    print(f"  State: {row['STATE']}, District: {row.get('DISTRICT', 'N/A')}")
                    found_records.append(row)
            
            # Search in ASSESSMENT UNIT column
            if 'ASSESSMENT UNIT' in df.columns:
                assessment_matches = df[df['ASSESSMENT UNIT'].astype(str).str.contains(variation, case=False, na=False)]
                if not assessment_matches.empty:
                    print(f"\nFound in ASSESSMENT UNIT column for '{variation}':")
                    for idx, row in assessment_matches.iterrows():
                        print(f"  State: {row['STATE']}, District: {row.get('DISTRICT', 'N/A')}, Assessment Unit: {row.get('ASSESSMENT UNIT', 'N/A')}")
                        found_records.append(row)
        
        # Check Tamil Nadu specifically
        print("\nChecking Tamil Nadu districts...")
        tamil_nadu = df[df['STATE'].astype(str).str.contains('tamil', case=False, na=False)]
        if not tamil_nadu.empty:
            print(f"Found {len(tamil_nadu)} Tamil Nadu records")
            districts = tamil_nadu['DISTRICT'].unique()
            print("Tamil Nadu districts:")
            for district in sorted(districts):
                if pd.notna(district):
                    print(f"  - {district}")
        
        # Check Nilgiris specifically
        print("\nChecking for Nilgiris...")
        nilgiris = df[df['DISTRICT'].astype(str).str.contains('nilgiri', case=False, na=False)]
        if not nilgiris.empty:
            print(f"Found {len(nilgiris)} Nilgiris records")
            for idx, row in nilgiris.iterrows():
                print(f"  State: {row['STATE']}, District: {row.get('DISTRICT', 'N/A')}")
        
        if not found_records:
            print("\nNo records found for Ooty/Udhagamandalam")
            print("\nChecking if Tamil Nadu is in the data...")
            tamil_nadu_states = df['STATE'].unique()
            tamil_states = [state for state in tamil_nadu_states if pd.notna(state) and 'tamil' in str(state).lower()]
            if tamil_states:
                print(f"Tamil Nadu states found: {tamil_states}")
            else:
                print("No Tamil Nadu states found in the data")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    search_ooty_in_data()

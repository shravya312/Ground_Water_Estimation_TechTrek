#!/usr/bin/env python3
"""
Analyze data structure for dropdown menus
"""

import pandas as pd
import json

def analyze_dropdown_data():
    """Analyze what data is available for dropdowns"""
    print("Analyzing Data for Dropdown Menus")
    print("=" * 50)
    
    try:
        # Load the master data
        df = pd.read_csv('master_groundwater_data.csv', low_memory=False)
        print(f"Loaded {len(df)} records from master_groundwater_data.csv")
        
        # Analyze states
        states = df['STATE'].dropna().unique().tolist()
        states = [s for s in states if s and str(s).strip()]
        states.sort()
        print(f"\nSTATES ({len(states)}):")
        for i, state in enumerate(states[:10], 1):
            print(f"  {i:2d}. {state}")
        if len(states) > 10:
            print(f"  ... and {len(states) - 10} more")
        
        # Analyze districts
        districts = df['DISTRICT'].dropna().unique().tolist()
        districts = [d for d in districts if d and str(d).strip()]
        districts.sort()
        print(f"\nDISTRICTS ({len(districts)}):")
        for i, district in enumerate(districts[:10], 1):
            print(f"  {i:2d}. {district}")
        if len(districts) > 10:
            print(f"  ... and {len(districts) - 10} more")
        
        # Check for taluk data
        taluk_columns = [col for col in df.columns if 'taluk' in col.lower()]
        print(f"\nTALUK COLUMNS FOUND: {taluk_columns}")
        
        if taluk_columns:
            for col in taluk_columns:
                taluks = df[col].dropna().unique().tolist()
                taluks = [t for t in taluks if t and str(t).strip()]
                taluks.sort()
                print(f"\n{col.upper()} ({len(taluks)}):")
                for i, taluk in enumerate(taluks[:10], 1):
                    print(f"  {i:2d}. {taluk}")
                if len(taluks) > 10:
                    print(f"  ... and {len(taluks) - 10} more")
        
        # Check for other administrative divisions
        admin_columns = [col for col in df.columns if any(term in col.lower() for term in ['block', 'mandal', 'village', 'tehsil'])]
        print(f"\nOTHER ADMINISTRATIVE COLUMNS: {admin_columns}")
        
        # Create hierarchical data structure
        hierarchical_data = {}
        for state in states[:5]:  # Sample first 5 states
            state_data = df[df['STATE'] == state]
            districts_in_state = state_data['DISTRICT'].dropna().unique().tolist()
            districts_in_state = [d for d in districts_in_state if d and str(d).strip()]
            
            hierarchical_data[state] = {
                'districts': sorted(districts_in_state),
                'district_count': len(districts_in_state)
            }
            
            # Add taluk data if available
            if taluk_columns:
                for col in taluk_columns:
                    taluks_in_state = state_data[col].dropna().unique().tolist()
                    taluks_in_state = [t for t in taluks_in_state if t and str(t).strip()]
                    hierarchical_data[state][col.lower()] = sorted(taluks_in_state)
        
        print(f"\nHIERARCHICAL DATA SAMPLE:")
        for state, data in hierarchical_data.items():
            print(f"  {state}: {data['district_count']} districts")
            if 'taluk' in data:
                print(f"    Taluks: {len(data['taluk'])}")
        
        # Save data for dropdowns
        dropdown_data = {
            'states': states,
            'districts': districts,
            'hierarchical': hierarchical_data
        }
        
        if taluk_columns:
            for col in taluk_columns:
                taluks = df[col].dropna().unique().tolist()
                taluks = [t for t in taluks if t and str(t).strip()]
                dropdown_data[col.lower()] = sorted(taluks)
        
        # Save to JSON file
        with open('dropdown_data.json', 'w', encoding='utf-8') as f:
            json.dump(dropdown_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDropdown data saved to 'dropdown_data.json'")
        
        return dropdown_data
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyze_dropdown_data()

#!/usr/bin/env python3
"""
Fix main2.py to use INGRIS data for state extraction
"""

import pandas as pd
import re

def load_ingris_data():
    """Load INGRIS data for state extraction"""
    try:
        print("🔄 Loading INGRIS data for state extraction...")
        df = pd.read_csv("ingris_rag_ready_complete.csv", skiprows=1)
        
        # Clean column names
        df.columns = [
            'serial_number', 'state', 'district', 'island', 'watershed_district',
            'rainfall_mm', 'total_geographical_area_ha', 'ground_water_recharge_ham',
            'inflows_and_outflows_ham', 'annual_ground_water_recharge_ham',
            'environmental_flows_ham', 'annual_extractable_ground_water_resource_ham',
            'ground_water_extraction_for_all_uses_ham', 'stage_of_ground_water_extraction_',
            'categorization_of_assessment_unit', 'pre_monsoon_of_gw_trend',
            'post_monsoon_of_gw_trend', 'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham',
            'net_annual_ground_water_availability_for_future_use_ham', 'quality_tagging',
            'additional_potential_resources_under_specific_conditionsham', 'coastal_areas',
            'instorage_unconfined_ground_water_resourcesham', 'total_ground_water_availability_in_unconfined_aquifier_ham',
            'dynamic_confined_ground_water_resourcesham', 'instorage_confined_ground_water_resourcesham',
            'total_confined_ground_water_resources_ham', 'dynamic_semi_confined_ground_water_resources_ham',
            'instorage_semi_confined_ground_water_resources_ham', 'total_semiconfined_ground_water_resources_ham',
            'total_ground_water_availability_in_the_area_ham', 'source_file', 'year',
            'tehsil', 'taluk', 'block', 'valley', 'assessment_unit', 'mandal',
            'village', 'watershed_category', 'firka', 'combined_text'
        ]
        
        print(f"✅ Loaded INGRIS data: {len(df)} records")
        return df
    except Exception as e:
        print(f"❌ Error loading INGRIS data: {e}")
        return None

def extract_state_from_query_ingris(query, ingris_df):
    """Extract state from query using INGRIS data"""
    if ingris_df is None:
        return None, None
    
    unique_states = ingris_df['state'].dropna().unique().tolist()
    unique_districts = ingris_df['district'].dropna().unique().tolist()
    
    target_state = None
    target_district = None
    
    # Try to find state with fuzzy matching
    for state in unique_states:
        if pd.notna(state):
            # Exact match
            if re.search(r'\b' + re.escape(str(state)) + r'\b', query, re.IGNORECASE):
                target_state = state
                break
            # Partial match
            elif str(state).lower() in query.lower():
                target_state = state
                break
    
    if target_state:
        districts_in_state = ingris_df[ingris_df['state'] == target_state]['district'].unique().tolist()
        for district in districts_in_state:
            if pd.notna(district):
                # Exact match
                if re.search(r'\b' + re.escape(str(district)) + r'\b', query, re.IGNORECASE):
                    target_district = district
                    break
                # Partial match
                elif str(district).lower() in query.lower():
                    target_district = district
                    break
    
    return target_state, target_district

def test_ingris_state_extraction():
    """Test state extraction with INGRIS data"""
    print("🔍 Testing State Extraction with INGRIS Data")
    print("=" * 50)
    
    # Load INGRIS data
    ingris_df = load_ingris_data()
    if ingris_df is None:
        return
    
    # Test queries
    test_queries = [
        "ground water estimation of odisha",
        "ground water estimation of chhattisgarh", 
        "ground water estimation of karnataka",
        "ground water estimation of telangana"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        target_state, target_district = extract_state_from_query_ingris(query, ingris_df)
        print(f"Extracted state: {target_state}")
        print(f"Extracted district: {target_district}")
        
        if target_state:
            # Count records for this state
            state_count = len(ingris_df[ingris_df['state'] == target_state])
            print(f"Records for {target_state}: {state_count}")

if __name__ == "__main__":
    test_ingris_state_extraction()

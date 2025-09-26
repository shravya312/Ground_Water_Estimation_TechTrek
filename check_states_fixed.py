#!/usr/bin/env python3
"""
Check what states are available in the data (fixed version)
"""

import pandas as pd
from collections import Counter

def check_available_states():
    """Check what states are available in the data"""
    print("🔍 Checking Available States in Data")
    print("=" * 50)
    
    try:
        # Load the CSV data, skipping the header row
        print("🔄 Loading data from ingris_rag_ready_complete.csv...")
        df = pd.read_csv("ingris_rag_ready_complete.csv", skiprows=1)  # Skip header row
        print(f"✅ Loaded {len(df)} records")
        
        # Rename columns to match the actual data structure
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
        
        # Check states
        print("\n📊 State Analysis:")
        states = df['state'].dropna().unique()
        print(f"Total unique states: {len(states)}")
        
        # Show all states
        print("\n🏛️ All States in Data:")
        for i, state in enumerate(sorted(states), 1):
            count = len(df[df['state'] == state])
            print(f"{i:2d}. {state} ({count:,} records)")
        
        # Check for Chhattisgarh variations
        print("\n🔍 Checking for Chhattisgarh variations:")
        chhattisgarh_variations = [
            'CHHATTISGARH', 'CHATTISGARH', 'CHHATISGARH', 'CHHATTISGARH',
            'chhattisgarh', 'chattisgarh', 'chhatisgarh', 'chhattisgarh',
            'Chhattisgarh', 'Chattisgarh', 'Chhatisgarh', 'Chhattisgarh'
        ]
        
        found_variations = []
        for variation in chhattisgarh_variations:
            matches = df[df['state'].str.contains(variation, case=False, na=False)]
            if len(matches) > 0:
                found_variations.append((variation, len(matches)))
        
        if found_variations:
            print("✅ Found Chhattisgarh data:")
            for variation, count in found_variations:
                print(f"   - {variation}: {count} records")
        else:
            print("❌ No Chhattisgarh data found")
        
        # Check similar state names
        print("\n🔍 Checking for similar state names:")
        similar_states = []
        for state in states:
            if any(word in state.upper() for word in ['CHHAT', 'CHAT', 'GARH']):
                similar_states.append(state)
        
        if similar_states:
            print("States with similar names:")
            for state in similar_states:
                count = len(df[df['state'] == state])
                print(f"   - {state}: {count} records")
        
        # Top 10 states by record count
        print("\n📈 Top 10 States by Record Count:")
        state_counts = df['state'].value_counts().head(10)
        for state, count in state_counts.items():
            print(f"{state}: {count:,} records")
        
        # Show sample data for first few states
        print("\n📄 Sample data for first 3 states:")
        for state in sorted(states)[:3]:
            sample = df[df['state'] == state].iloc[0]
            print(f"\n{state}:")
            print(f"  District: {sample['district']}")
            print(f"  Year: {sample['year']}")
            print(f"  Rainfall: {sample['rainfall_mm']} mm")
            print(f"  Ground Water Recharge: {sample['ground_water_recharge_ham']} ham")
        
        return states.tolist()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    check_available_states()

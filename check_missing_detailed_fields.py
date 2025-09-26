#!/usr/bin/env python3
"""
Check for specific missing detailed fields mentioned in the user query
"""

import pandas as pd
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv()

def check_missing_detailed_fields():
    """Check for specific missing detailed fields"""
    print("🔍 Checking for Missing Detailed Fields")
    print("=" * 50)
    
    # Fields that are mentioned as "No data available" in the user query
    missing_fields = {
        'extraction_purposes': [
            'ground_water_extraction_for_domestic_use',
            'ground_water_extraction_for_industrial_use', 
            'ground_water_extraction_for_irrigation'
        ],
        'groundwater_sources': [
            'canals',
            'surface_water_irrigation',
            'ground_water_irrigation',
            'tanks_and_ponds',
            'water_conservation_structures',
            'pipelines',
            'sewages_and_flash_flood_channels'
        ]
    }
    
    try:
        # Load CSV data
        print("🔄 Loading ingris_rag_ready_complete.csv...")
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
        
        print(f"✅ Loaded {len(df)} records")
        print(f"📊 Available columns: {sorted(df.columns)}")
        
        # Check for extraction purposes breakdown
        print("\n🔍 Checking for Extraction Purposes Breakdown:")
        print("Fields mentioned as 'No data available':")
        for field in missing_fields['extraction_purposes']:
            # Check for exact match
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                print(f"✅ {field}: {non_null_count} non-null values")
            else:
                # Check for similar field names
                similar_fields = [col for col in df.columns if any(word in col.lower() for word in field.lower().split('_'))]
                if similar_fields:
                    print(f"🔍 {field}: Not found, but similar fields exist: {similar_fields}")
                else:
                    print(f"❌ {field}: Not found")
        
        # Check for groundwater sources breakdown
        print("\n🔍 Checking for Groundwater Sources Breakdown:")
        print("Fields mentioned as 'No data available':")
        for field in missing_fields['groundwater_sources']:
            # Check for exact match
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                print(f"✅ {field}: {non_null_count} non-null values")
            else:
                # Check for similar field names
                similar_fields = [col for col in df.columns if any(word in col.lower() for word in field.lower().split('_'))]
                if similar_fields:
                    print(f"🔍 {field}: Not found, but similar fields exist: {similar_fields}")
                else:
                    print(f"❌ {field}: Not found")
        
        # Check what fields are actually available for detailed breakdown
        print("\n🔍 Available Fields for Detailed Analysis:")
        print("=" * 40)
        
        # All fields containing 'extraction' or 'recharge'
        extraction_fields = [col for col in df.columns if 'extraction' in col.lower() or 'recharge' in col.lower()]
        print("Extraction/Recharge related fields:")
        for field in extraction_fields:
            non_null_count = df[field].notna().sum()
            print(f"  - {field}: {non_null_count} non-null values")
        
        # All fields containing 'irrigation', 'domestic', 'industrial'
        purpose_fields = [col for col in df.columns if any(word in col.lower() for word in ['irrigation', 'domestic', 'industrial'])]
        print("\nPurpose-related fields:")
        for field in purpose_fields:
            non_null_count = df[field].notna().sum()
            print(f"  - {field}: {non_null_count} non-null values")
        
        # Check for Karnataka data specifically
        print("\n🔍 Karnataka Data Analysis:")
        karnataka_df = df[df['state'] == 'KARNATAKA']
        print(f"Total Karnataka records: {len(karnataka_df)}")
        
        if len(karnataka_df) > 0:
            # Check for the specific districts mentioned
            target_districts = ['CHAMARAJANAGARA', 'CHIKKABALLAPURA', 'CHIKKAMAGALURU']
            for district in target_districts:
                district_data = karnataka_df[karnataka_df['district'].str.contains(district, case=False, na=False)]
                if len(district_data) > 0:
                    print(f"\n{district} data:")
                    sample = district_data.iloc[0]
                    print(f"  Year: {sample.get('year', 'N/A')}")
                    print(f"  Ground Water Recharge: {sample.get('ground_water_recharge_ham', 'N/A')} ham")
                    print(f"  Annual Ground Water Recharge: {sample.get('annual_ground_water_recharge_ham', 'N/A')} ham")
                    print(f"  Ground Water Extraction: {sample.get('ground_water_extraction_for_all_uses_ham', 'N/A')} ham")
                    print(f"  Stage of Extraction: {sample.get('stage_of_ground_water_extraction_', 'N/A')}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    check_missing_detailed_fields()

#!/usr/bin/env python3
"""
Check if detailed extraction purposes and groundwater sources data is present
"""

import pandas as pd
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv()

def check_csv_detailed_fields():
    """Check detailed fields in CSV file"""
    print("🔍 Checking Detailed Fields in CSV File")
    print("=" * 50)
    
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
        print(f"📊 Total columns: {len(df.columns)}")
        
        # Check for extraction purposes fields
        print("\n🔍 Checking for Extraction Purposes Fields:")
        extraction_fields = [
            'ground_water_extraction_for_all_uses_ham',
            'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham'
        ]
        
        for field in extraction_fields:
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                print(f"✅ {field}: {non_null_count} non-null values")
                
                # Show sample values
                sample_values = df[field].dropna().head(3).tolist()
                print(f"   Sample values: {sample_values}")
            else:
                print(f"❌ {field}: Not found")
        
        # Check for groundwater sources fields
        print("\n🔍 Checking for Groundwater Sources Fields:")
        source_fields = [
            'ground_water_recharge_ham',
            'annual_ground_water_recharge_ham',
            'inflows_and_outflows_ham',
            'environmental_flows_ham'
        ]
        
        for field in source_fields:
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                print(f"✅ {field}: {non_null_count} non-null values")
                
                # Show sample values
                sample_values = df[field].dropna().head(3).tolist()
                print(f"   Sample values: {sample_values}")
            else:
                print(f"❌ {field}: Not found")
        
        # Check for specific Karnataka districts mentioned
        print("\n🔍 Checking for Karnataka Districts:")
        karnataka_df = df[df['state'] == 'KARNATAKA']
        print(f"✅ Karnataka records: {len(karnataka_df)}")
        
        if len(karnataka_df) > 0:
            districts = karnataka_df['district'].unique()
            print(f"Karnataka districts: {sorted(districts)}")
            
            # Check for specific districts mentioned
            target_districts = ['BENGALURU URBAN', 'CHAMARAJANAGARA', 'CHIKKABALLAPURA', 'CHIKKAMAGALURU']
            for district in target_districts:
                district_data = karnataka_df[karnataka_df['district'].str.contains(district, case=False, na=False)]
                if len(district_data) > 0:
                    print(f"✅ {district}: {len(district_data)} records")
                    
                    # Show sample data for this district
                    sample = district_data.iloc[0]
                    print(f"   Sample data for {district}:")
                    print(f"     Year: {sample.get('year', 'N/A')}")
                    print(f"     Ground Water Recharge: {sample.get('ground_water_recharge_ham', 'N/A')} ham")
                    print(f"     Annual Ground Water Recharge: {sample.get('annual_ground_water_recharge_ham', 'N/A')} ham")
                    print(f"     Ground Water Extraction: {sample.get('ground_water_extraction_for_all_uses_ham', 'N/A')} ham")
                else:
                    print(f"❌ {district}: No records found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_qdrant_detailed_fields():
    """Check detailed fields in Qdrant collection"""
    print("\n🔍 Checking Detailed Fields in Qdrant Collection")
    print("=" * 50)
    
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = "ingris_groundwater_collection"
    
    try:
        # Connect to Qdrant
        print("🔄 Connecting to Qdrant...")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        
        # Sample some data to check fields
        print("🔄 Sampling data from Qdrant...")
        sample_points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        print(f"✅ Retrieved {len(sample_points)} sample points")
        
        if sample_points:
            print("\n📊 Sample Point Fields:")
            payload = sample_points[0].payload
            print(f"Total fields: {len(payload.keys())}")
            
            # Check for extraction purposes fields
            print("\n🔍 Checking for Extraction Purposes Fields in Qdrant:")
            extraction_fields = [
                'ground_water_extraction_for_all_uses_ham',
                'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham'
            ]
            
            for field in extraction_fields:
                if field in payload:
                    print(f"✅ {field}: {payload[field]}")
                else:
                    print(f"❌ {field}: Not found")
            
            # Check for groundwater sources fields
            print("\n🔍 Checking for Groundwater Sources Fields in Qdrant:")
            source_fields = [
                'ground_water_recharge_ham',
                'annual_ground_water_recharge_ham',
                'inflows_and_outflows_ham',
                'environmental_flows_ham'
            ]
            
            for field in source_fields:
                if field in payload:
                    print(f"✅ {field}: {payload[field]}")
                else:
                    print(f"❌ {field}: Not found")
            
            # Check for Karnataka data
            print("\n🔍 Checking for Karnataka Data in Qdrant:")
            karnataka_filter = Filter(
                must=[
                    FieldCondition(
                        key="STATE",
                        match=MatchValue(value="KARNATAKA")
                    )
                ]
            )
            
            karnataka_sample, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=karnataka_filter,
                limit=3,
                with_payload=True,
                with_vectors=False
            )
            
            print(f"✅ Karnataka records in Qdrant: {len(karnataka_sample)}")
            
            if karnataka_sample:
                print("Sample Karnataka records:")
                for i, point in enumerate(karnataka_sample, 1):
                    payload = point.payload
                    district = payload.get('DISTRICT', 'Unknown')
                    year = payload.get('Assessment_Year', 'Unknown')
                    recharge = payload.get('ground_water_recharge_ham', 'N/A')
                    extraction = payload.get('ground_water_extraction_for_all_uses_ham', 'N/A')
                    print(f"  {i}. {district} ({year})")
                    print(f"     Recharge: {recharge} ham")
                    print(f"     Extraction: {extraction} ham")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function to check both data sources"""
    print("🔍 Checking Detailed Data Fields in Both Sources")
    print("=" * 60)
    
    csv_success = check_csv_detailed_fields()
    qdrant_success = check_qdrant_detailed_fields()
    
    print("\n📊 Summary:")
    print("=" * 20)
    print(f"CSV Data: {'✅ Available' if csv_success else '❌ Error'}")
    print(f"Qdrant Data: {'✅ Available' if qdrant_success else '❌ Error'}")

if __name__ == "__main__":
    main()

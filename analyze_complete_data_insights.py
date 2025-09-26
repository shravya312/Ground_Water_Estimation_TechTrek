#!/usr/bin/env python3
"""
Analyze complete INGRIS data to identify additional insights for chatbot enhancement
"""

import pandas as pd
import numpy as np
from collections import Counter
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv()

def analyze_complete_data_insights():
    """Analyze complete data for additional insights"""
    print("🔍 Analyzing Complete INGRIS Data for Enhanced Insights")
    print("=" * 60)
    
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
        
        # 1. Analyze temporal trends
        print("\n📅 1. TEMPORAL ANALYSIS")
        print("=" * 30)
        years = df['year'].dropna().unique()
        # Convert to numeric and filter out non-numeric values
        numeric_years = []
        for year in years:
            try:
                numeric_years.append(int(year))
            except (ValueError, TypeError):
                continue
        
        print(f"Years covered: {sorted(numeric_years)}")
        
        # Year-wise data distribution
        year_counts = df['year'].value_counts()
        print("\nYear-wise record distribution:")
        for year, count in year_counts.items():
            print(f"  {year}: {count} records")
        
        # 2. Analyze groundwater categorization
        print("\n🏷️ 2. GROUNDWATER CATEGORIZATION ANALYSIS")
        print("=" * 40)
        categories = df['categorization_of_assessment_unit'].value_counts()
        print("Groundwater status categories:")
        for category, count in categories.items():
            percentage = (count / len(df)) * 100
            print(f"  {category}: {count} records ({percentage:.1f}%)")
        
        # 3. Analyze extraction stage distribution
        print("\n📊 3. EXTRACTION STAGE ANALYSIS")
        print("=" * 35)
        extraction_stages = df['stage_of_ground_water_extraction_'].dropna()
        if len(extraction_stages) > 0:
            print(f"Extraction stage statistics:")
            print(f"  Mean: {extraction_stages.mean():.2f}%")
            print(f"  Median: {extraction_stages.median():.2f}%")
            print(f"  Min: {extraction_stages.min():.2f}%")
            print(f"  Max: {extraction_stages.max():.2f}%")
            
            # Categorize extraction stages
            safe = extraction_stages[extraction_stages < 70].count()
            semi_critical = extraction_stages[(extraction_stages >= 70) & (extraction_stages < 90)].count()
            critical = extraction_stages[(extraction_stages >= 90) & (extraction_stages < 100)].count()
            over_exploited = extraction_stages[extraction_stages >= 100].count()
            
            print(f"\nExtraction stage categories:")
            print(f"  Safe (<70%): {safe} records")
            print(f"  Semi-Critical (70-90%): {semi_critical} records")
            print(f"  Critical (90-100%): {critical} records")
            print(f"  Over-exploited (≥100%): {over_exploited} records")
        
        # 4. Analyze groundwater trends
        print("\n📈 4. GROUNDWATER TREND ANALYSIS")
        print("=" * 35)
        pre_monsoon_trends = df['pre_monsoon_of_gw_trend'].value_counts()
        post_monsoon_trends = df['post_monsoon_of_gw_trend'].value_counts()
        
        print("Pre-monsoon groundwater trends:")
        for trend, count in pre_monsoon_trends.items():
            print(f"  {trend}: {count} records")
        
        print("\nPost-monsoon groundwater trends:")
        for trend, count in post_monsoon_trends.items():
            print(f"  {trend}: {count} records")
        
        # 5. Analyze geographical distribution
        print("\n🗺️ 5. GEOGRAPHICAL DISTRIBUTION ANALYSIS")
        print("=" * 40)
        states = df['state'].value_counts()
        print("Top 10 states by record count:")
        for state, count in states.head(10).items():
            print(f"  {state}: {count} records")
        
        # 6. Analyze watershed categories
        print("\n🌊 6. WATERSHED CATEGORY ANALYSIS")
        print("=" * 35)
        watershed_categories = df['watershed_category'].value_counts()
        print("Watershed categories:")
        for category, count in watershed_categories.items():
            print(f"  {category}: {count} records")
        
        # 7. Analyze coastal areas
        print("\n🏖️ 7. COASTAL AREAS ANALYSIS")
        print("=" * 30)
        coastal_areas = df['coastal_areas'].value_counts()
        print("Coastal area distribution:")
        for area, count in coastal_areas.items():
            print(f"  {area}: {count} records")
        
        # 8. Analyze quality tagging
        print("\n🔬 8. WATER QUALITY ANALYSIS")
        print("=" * 30)
        quality_tags = df['quality_tagging'].value_counts()
        print("Water quality tags:")
        for tag, count in quality_tags.items():
            print(f"  {tag}: {count} records")
        
        # 9. Analyze additional potential resources
        print("\n💧 9. ADDITIONAL POTENTIAL RESOURCES")
        print("=" * 40)
        additional_resources = df['additional_potential_resources_under_specific_conditionsham'].dropna()
        if len(additional_resources) > 0:
            print(f"Records with additional potential resources: {len(additional_resources)}")
            print(f"Total additional potential: {additional_resources.sum():.2f} ham")
        
        # 10. Analyze storage resources
        print("\n🏗️ 10. GROUNDWATER STORAGE ANALYSIS")
        print("=" * 40)
        storage_fields = [
            'instorage_unconfined_ground_water_resourcesham',
            'total_ground_water_availability_in_unconfined_aquifier_ham',
            'dynamic_confined_ground_water_resources_ham',
            'instorage_confined_ground_water_resources_ham',
            'total_confined_ground_water_resources_ham',
            'dynamic_semi_confined_ground_water_resources_ham',
            'instorage_semi_confined_ground_water_resources_ham',
            'total_semiconfined_ground_water_resources_ham',
            'total_ground_water_availability_in_the_area_ham'
        ]
        
        for field in storage_fields:
            non_null_count = df[field].notna().sum()
            if non_null_count > 0:
                print(f"  {field}: {non_null_count} non-null values")
        
        # 11. Analyze rainfall patterns
        print("\n🌧️ 11. RAINFALL PATTERN ANALYSIS")
        print("=" * 35)
        rainfall_data = df['rainfall_mm'].dropna()
        if len(rainfall_data) > 0:
            print(f"Rainfall statistics:")
            print(f"  Mean: {rainfall_data.mean():.2f} mm")
            print(f"  Median: {rainfall_data.median():.2f} mm")
            print(f"  Min: {rainfall_data.min():.2f} mm")
            print(f"  Max: {rainfall_data.max():.2f} mm")
        
        # 12. Analyze administrative divisions
        print("\n🏛️ 12. ADMINISTRATIVE DIVISION ANALYSIS")
        print("=" * 40)
        tehsils = df['tehsil'].value_counts()
        taluks = df['taluk'].value_counts()
        blocks = df['block'].value_counts()
        mandals = df['mandal'].value_counts()
        villages = df['village'].value_counts()
        
        print(f"Administrative divisions:")
        print(f"  Tehsils: {len(tehsils)} unique")
        print(f"  Taluks: {len(taluks)} unique")
        print(f"  Blocks: {len(blocks)} unique")
        print(f"  Mandals: {len(mandals)} unique")
        print(f"  Villages: {len(villages)} unique")
        
        # 13. Identify high-value insights for chatbot
        print("\n💡 13. HIGH-VALUE INSIGHTS FOR CHATBOT ENHANCEMENT")
        print("=" * 55)
        
        insights = []
        
        # Critical areas analysis
        critical_areas = df[df['stage_of_ground_water_extraction_'] >= 100]
        if len(critical_areas) > 0:
            insights.append(f"🚨 {len(critical_areas)} areas are over-exploited (≥100% extraction)")
        
        # Trend analysis
        falling_trends = df[df['pre_monsoon_of_gw_trend'] == 'Falling']
        if len(falling_trends) > 0:
            insights.append(f"📉 {len(falling_trends)} areas show falling pre-monsoon trends")
        
        # Coastal vulnerability
        coastal_data = df[df['coastal_areas'].notna()]
        if len(coastal_data) > 0:
            insights.append(f"🏖️ {len(coastal_data)} coastal areas need special attention")
        
        # Quality issues
        quality_issues = df[df['quality_tagging'].notna()]
        if len(quality_issues) > 0:
            insights.append(f"🔬 {len(quality_issues)} areas have water quality concerns")
        
        # Additional potential
        additional_potential = df[df['additional_potential_resources_under_specific_conditionsham'].notna()]
        if len(additional_potential) > 0:
            insights.append(f"💧 {len(additional_potential)} areas have additional potential resources")
        
        print("Key insights that can enhance chatbot responses:")
        for insight in insights:
            print(f"  {insight}")
        
        # 14. Suggest enhanced response templates
        print("\n📝 14. SUGGESTED ENHANCEMENTS FOR CHATBOT RESPONSES")
        print("=" * 55)
        
        enhancements = [
            "Add groundwater trend analysis (rising/falling/static)",
            "Include extraction stage categorization (safe/semi-critical/critical/over-exploited)",
            "Provide coastal area vulnerability assessment",
            "Include water quality status and concerns",
            "Add temporal analysis showing year-wise changes",
            "Include administrative division breakdown (tehsil/taluk/block/village)",
            "Provide watershed category analysis",
            "Add additional potential resources assessment",
            "Include storage resource analysis (confined/unconfined/semi-confined)",
            "Provide comparative analysis with state/national averages",
            "Add sustainability indicators and recommendations",
            "Include climate resilience assessment based on rainfall patterns"
        ]
        
        for i, enhancement in enumerate(enhancements, 1):
            print(f"  {i:2d}. {enhancement}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    analyze_complete_data_insights()

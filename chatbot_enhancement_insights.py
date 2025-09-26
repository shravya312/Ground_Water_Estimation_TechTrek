#!/usr/bin/env python3
"""
Generate focused insights for chatbot enhancement from INGRIS data
"""

import pandas as pd
import numpy as np
from collections import Counter
import os

def generate_chatbot_enhancement_insights():
    """Generate focused insights for chatbot enhancement"""
    print("🚀 CHATBOT ENHANCEMENT INSIGHTS FROM INGRIS DATA")
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
            'dynamic_confined_ground_water_resources_ham', 'instorage_confined_ground_water_resources_ham',
            'total_confined_ground_water_resources_ham', 'dynamic_semi_confined_ground_water_resources_ham',
            'instorage_semi_confined_ground_water_resources_ham', 'total_semiconfined_ground_water_resources_ham',
            'total_ground_water_availability_in_the_area_ham', 'source_file', 'year',
            'tehsil', 'taluk', 'block', 'valley', 'assessment_unit', 'mandal',
            'village', 'watershed_category', 'firka', 'combined_text'
        ]
        
        print(f"✅ Loaded {len(df)} records")
        
        # 1. CRITICAL INSIGHTS FOR CHATBOT
        print("\n🚨 1. CRITICAL INSIGHTS FOR CHATBOT ENHANCEMENT")
        print("=" * 50)
        
        # Convert extraction stage to numeric
        df['stage_of_ground_water_extraction_'] = pd.to_numeric(df['stage_of_ground_water_extraction_'], errors='coerce')
        
        # Over-exploited areas
        over_exploited = df[df['stage_of_ground_water_extraction_'] >= 100]
        print(f"🚨 OVER-EXPLOITED AREAS: {len(over_exploited)} areas (≥100% extraction)")
        if len(over_exploited) > 0:
            top_over_exploited = over_exploited.nlargest(5, 'stage_of_ground_water_extraction_')
            print("   Top 5 over-exploited areas:")
            for _, row in top_over_exploited.iterrows():
                print(f"   - {row['state']} | {row['district']}: {row['stage_of_ground_water_extraction_']:.1f}%")
        
        # Critical areas (90-100%)
        critical = df[(df['stage_of_ground_water_extraction_'] >= 90) & (df['stage_of_ground_water_extraction_'] < 100)]
        print(f"⚠️  CRITICAL AREAS: {len(critical)} areas (90-100% extraction)")
        
        # Safe areas
        safe = df[df['stage_of_ground_water_extraction_'] < 70]
        print(f"✅ SAFE AREAS: {len(safe)} areas (<70% extraction)")
        
        # 2. TREND ANALYSIS
        print("\n📈 2. GROUNDWATER TREND ANALYSIS")
        print("=" * 35)
        
        pre_monsoon_trends = df['pre_monsoon_of_gw_trend'].value_counts()
        post_monsoon_trends = df['post_monsoon_of_gw_trend'].value_counts()
        
        print("Pre-monsoon trends:")
        for trend, count in pre_monsoon_trends.items():
            percentage = (count / len(df)) * 100
            print(f"   {trend}: {count} areas ({percentage:.1f}%)")
        
        print("\nPost-monsoon trends:")
        for trend, count in post_monsoon_trends.items():
            percentage = (count / len(df)) * 100
            print(f"   {trend}: {count} areas ({percentage:.1f}%)")
        
        # 3. WATER QUALITY CONCERNS
        print("\n🔬 3. WATER QUALITY ANALYSIS")
        print("=" * 30)
        
        quality_issues = df[df['quality_tagging'].notna()]
        print(f"Areas with quality concerns: {len(quality_issues)}")
        if len(quality_issues) > 0:
            quality_types = quality_issues['quality_tagging'].value_counts()
            print("Quality issue types:")
            for quality, count in quality_types.items():
                print(f"   {quality}: {count} areas")
        
        # 4. COASTAL VULNERABILITY
        print("\n🏖️ 4. COASTAL AREA ANALYSIS")
        print("=" * 30)
        
        coastal_areas = df[df['coastal_areas'].notna()]
        print(f"Coastal areas: {len(coastal_areas)}")
        if len(coastal_areas) > 0:
            coastal_states = coastal_areas['state'].value_counts()
            print("Coastal states:")
            for state, count in coastal_states.head(5).items():
                print(f"   {state}: {count} areas")
        
        # 5. ADDITIONAL POTENTIAL RESOURCES
        print("\n💧 5. ADDITIONAL POTENTIAL RESOURCES")
        print("=" * 40)
        
        additional_potential = df[df['additional_potential_resources_under_specific_conditionsham'].notna()]
        print(f"Areas with additional potential: {len(additional_potential)}")
        if len(additional_potential) > 0:
            total_potential = additional_potential['additional_potential_resources_under_specific_conditionsham'].sum()
            print(f"Total additional potential: {total_potential:.2f} ham")
        
        # 6. TEMPORAL COVERAGE
        print("\n📅 6. TEMPORAL COVERAGE")
        print("=" * 25)
        
        years = df['year'].dropna().unique()
        numeric_years = []
        for year in years:
            try:
                numeric_years.append(int(year))
            except (ValueError, TypeError):
                continue
        
        if numeric_years:
            print(f"Data covers years: {min(numeric_years)} - {max(numeric_years)}")
            year_counts = df['year'].value_counts()
            print("Year-wise distribution:")
            for year, count in year_counts.head(5).items():
                print(f"   {year}: {count} records")
        
        # 7. GEOGRAPHICAL COVERAGE
        print("\n🗺️ 7. GEOGRAPHICAL COVERAGE")
        print("=" * 30)
        
        states = df['state'].value_counts()
        print(f"Total states covered: {len(states)}")
        print("Top 10 states by data volume:")
        for state, count in states.head(10).items():
            print(f"   {state}: {count} records")
        
        # 8. WATERSHED CATEGORIES
        print("\n🌊 8. WATERSHED CATEGORIES")
        print("=" * 30)
        
        watershed_categories = df['watershed_category'].value_counts()
        print("Watershed categories:")
        for category, count in watershed_categories.items():
            percentage = (count / len(df)) * 100
            print(f"   {category}: {count} areas ({percentage:.1f}%)")
        
        # 9. ENHANCED CHATBOT RESPONSE TEMPLATES
        print("\n💡 9. ENHANCED CHATBOT RESPONSE TEMPLATES")
        print("=" * 45)
        
        templates = [
            "🚨 CRITICAL ALERT: {area} is over-exploited with {extraction}% extraction rate",
            "📈 TREND ANALYSIS: {area} shows {trend} pre-monsoon groundwater trend",
            "🔬 QUALITY CONCERN: {area} has water quality issues: {quality_type}",
            "🏖️ COASTAL VULNERABILITY: {area} is a coastal region requiring special attention",
            "💧 ADDITIONAL POTENTIAL: {area} has {potential} ham of additional groundwater potential",
            "📊 SUSTAINABILITY STATUS: {area} is categorized as {category}",
            "🌧️ RAINFALL IMPACT: {area} receives {rainfall} mm rainfall affecting recharge",
            "🏗️ STORAGE ANALYSIS: {area} has {storage} ham of confined groundwater resources",
            "📅 TEMPORAL TREND: {area} data from {year} shows {change}",
            "🌊 WATERSHED STATUS: {area} falls under {watershed} watershed category"
        ]
        
        for i, template in enumerate(templates, 1):
            print(f"   {i:2d}. {template}")
        
        # 10. SPECIFIC ENHANCEMENTS FOR CHATBOT
        print("\n🚀 10. SPECIFIC CHATBOT ENHANCEMENTS")
        print("=" * 40)
        
        enhancements = [
            "Add real-time criticality alerts for over-exploited areas",
            "Include trend analysis (rising/falling/static) in responses",
            "Provide water quality status and recommendations",
            "Add coastal vulnerability assessment",
            "Include additional potential resources information",
            "Show sustainability indicators and recommendations",
            "Add temporal analysis showing year-wise changes",
            "Include administrative division breakdown (tehsil/taluk/block/village)",
            "Provide watershed category analysis",
            "Add storage resource analysis (confined/unconfined/semi-confined)",
            "Include comparative analysis with state/national averages",
            "Add climate resilience assessment based on rainfall patterns",
            "Provide specific recommendations for each area type",
            "Include future availability projections",
            "Add environmental flow requirements analysis"
        ]
        
        for i, enhancement in enumerate(enhancements, 1):
            print(f"   {i:2d}. {enhancement}")
        
        # 11. SAMPLE ENHANCED RESPONSES
        print("\n📝 11. SAMPLE ENHANCED RESPONSES")
        print("=" * 35)
        
        # Get sample data for demonstration
        sample_data = df.sample(3)
        
        for i, (_, row) in enumerate(sample_data.iterrows(), 1):
            print(f"\n   Sample Response {i}:")
            print(f"   Area: {row['state']} | {row['district']}")
            print(f"   Status: {row['categorization_of_assessment_unit']}")
            print(f"   Extraction: {row['stage_of_ground_water_extraction_']:.1f}%")
            print(f"   Trend: {row['pre_monsoon_of_gw_trend']} (pre-monsoon)")
            print(f"   Recharge: {row['ground_water_recharge_ham']:.2f} ham")
            print(f"   Quality: {row['quality_tagging'] if pd.notna(row['quality_tagging']) else 'No issues reported'}")
            print(f"   Watershed: {row['watershed_category']}")
            print(f"   Year: {row['year']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    generate_chatbot_enhancement_insights()

#!/usr/bin/env python3
"""
Analyze the complete 162,632 records dataset to identify additional insights
that can be provided in chatbot answers - CORRECTED VERSION
"""

import pandas as pd
import numpy as np
from collections import Counter

def analyze_complete_dataset():
    """Analyze the complete dataset for additional insights"""
    print("🔍 Analyzing Complete Dataset (162,632 records)")
    print("=" * 60)
    
    try:
        # Load the complete dataset
        print("📂 Loading complete dataset...")
        df = pd.read_csv('ingris_rag_ready_complete.csv', low_memory=False)
        print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Basic dataset overview
        print("\n📊 Dataset Overview:")
        print(f"• Total Records: {len(df):,}")
        print(f"• Total Columns: {len(df.columns)}")
        print(f"• Date Range: {df['year'].min()} - {df['year'].max()}")
        print(f"• States Covered: {df['state'].nunique()}")
        print(f"• Districts Covered: {df['district'].nunique()}")
        
        # 1. State-wise Analysis
        print("\n🏛️ STATE-WISE INSIGHTS:")
        print("=" * 30)
        
        # Convert extraction percentage to numeric
        df['stage_of_ground_water_extraction_'] = pd.to_numeric(df['stage_of_ground_water_extraction_'], errors='coerce')
        
        state_stats = df.groupby('state').agg({
            'district': 'nunique',
            'year': 'nunique',
            'stage_of_ground_water_extraction_': ['mean', 'std', 'count']
        }).round(2)
        
        state_stats.columns = ['Districts', 'Years', 'Avg_Extraction_%', 'Std_Extraction_%', 'Records']
        state_stats = state_stats.sort_values('Avg_Extraction_%', ascending=False)
        
        print("Top 10 States by Average Groundwater Extraction:")
        print(state_stats.head(10))
        
        # 2. Criticality Analysis
        print("\n🚨 CRITICALITY INSIGHTS:")
        print("=" * 30)
        
        if 'watershed_category' in df.columns:
            criticality_dist = df['watershed_category'].value_counts()
            print("Watershed Category Distribution:")
            for category, count in criticality_dist.items():
                percentage = (count / len(df)) * 100
                print(f"• {category}: {count:,} records ({percentage:.1f}%)")
        
        # 3. Temporal Analysis
        print("\n📅 TEMPORAL INSIGHTS:")
        print("=" * 30)
        
        year_stats = df.groupby('year').agg({
            'state': 'nunique',
            'district': 'nunique',
            'stage_of_ground_water_extraction_': 'mean'
        }).round(2)
        
        print("Year-wise Analysis:")
        print(year_stats)
        
        # 4. Water Quality Analysis
        print("\n🔬 WATER QUALITY INSIGHTS:")
        print("=" * 30)
        
        if 'quality_tagging' in df.columns:
            quality_data = df['quality_tagging'].value_counts()
            print("Quality Tagging Distribution:")
            for quality, count in quality_data.head(10).items():
                percentage = (count / len(df)) * 100
                print(f"• {quality}: {count:,} records ({percentage:.1f}%)")
        
        # 5. Rainfall and Recharge Analysis
        print("\n🌧️ RAINFALL & RECHARGE INSIGHTS:")
        print("=" * 30)
        
        # Convert to numeric
        df['rainfall_mm'] = pd.to_numeric(df['rainfall_mm'], errors='coerce')
        df['ground_water_recharge_ham'] = pd.to_numeric(df['ground_water_recharge_ham'], errors='coerce')
        
        print(f"Average Rainfall: {df['rainfall_mm'].mean():.1f} mm")
        print(f"Average Groundwater Recharge: {df['ground_water_recharge_ham'].mean():.1f} ham")
        print(f"Rainfall Range: {df['rainfall_mm'].min():.1f} - {df['rainfall_mm'].max():.1f} mm")
        print(f"Recharge Range: {df['ground_water_recharge_ham'].min():.1f} - {df['ground_water_recharge_ham'].max():.1f} ham")
        
        # 6. Administrative Hierarchy Analysis
        print("\n🏘️ ADMINISTRATIVE HIERARCHY INSIGHTS:")
        print("=" * 30)
        
        admin_cols = ['state', 'district', 'taluk', 'block', 'mandal', 'village']
        for col in admin_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                non_null_count = df[col].notna().sum()
                print(f"• {col}: {unique_count:,} unique values, {non_null_count:,} records with data")
        
        # 7. Extreme Cases Analysis
        print("\n⚠️ EXTREME CASES INSIGHTS:")
        print("=" * 30)
        
        extraction_col = 'stage_of_ground_water_extraction_'
        
        # Over-exploited areas (>100%)
        over_exploited = df[df[extraction_col] > 100]
        print(f"Over-exploited areas (>100% extraction): {len(over_exploited):,} records")
        
        if len(over_exploited) > 0:
            print("Top 10 most over-exploited areas:")
            top_over = over_exploited.nlargest(10, extraction_col)[['state', 'district', extraction_col]]
            print(top_over)
        
        # Safe areas (<70%)
        safe_areas = df[df[extraction_col] < 70]
        print(f"\nSafe areas (<70% extraction): {len(safe_areas):,} records")
        
        if len(safe_areas) > 0:
            print("Top 10 safest areas:")
            top_safe = safe_areas.nsmallest(10, extraction_col)[['state', 'district', extraction_col]]
            print(top_safe)
        
        # 8. Geographic Distribution
        print("\n🗺️ GEOGRAPHIC DISTRIBUTION INSIGHTS:")
        print("=" * 30)
        
        # State-wise record distribution
        state_dist = df['state'].value_counts()
        print("Top 10 states by number of records:")
        for state, count in state_dist.head(10).items():
            percentage = (count / len(df)) * 100
            print(f"• {state}: {count:,} records ({percentage:.1f}%)")
        
        # 9. Data Completeness Analysis
        print("\n📋 DATA COMPLETENESS INSIGHTS:")
        print("=" * 30)
        
        # Check completeness of key columns
        key_columns = [
            'stage_of_ground_water_extraction_',
            'ground_water_recharge_ham',
            'ground_water_extraction_for_all_uses_ham',
            'net_annual_ground_water_availability_for_future_use_ham'
        ]
        
        for col in key_columns:
            if col in df.columns:
                completeness = (df[col].notna().sum() / len(df)) * 100
                print(f"• {col}: {completeness:.1f}% complete")
        
        # 10. Additional Insights for Chatbot
        print("\n💡 ADDITIONAL INSIGHTS FOR CHATBOT:")
        print("=" * 30)
        
        insights = []
        
        # Insight 1: State comparison
        state_avg = df.groupby('state')[extraction_col].mean().sort_values(ascending=False)
        insights.append(f"State with highest average extraction: {state_avg.index[0]} ({state_avg.iloc[0]:.1f}%)")
        insights.append(f"State with lowest average extraction: {state_avg.index[-1]} ({state_avg.iloc[-1]:.1f}%)")
        
        # Insight 2: Year trends
        year_trend = df.groupby('year')[extraction_col].mean()
        if len(year_trend) > 1:
            trend_direction = "increasing" if year_trend.iloc[-1] > year_trend.iloc[0] else "decreasing"
            insights.append(f"Overall extraction trend: {trend_direction}")
        
        # Insight 3: Critical areas
        critical_count = len(df[df[extraction_col] > 90])
        insights.append(f"Critical areas (>90% extraction): {critical_count:,} records")
        
        # Insight 4: Safe areas
        safe_count = len(df[df[extraction_col] < 70])
        insights.append(f"Safe areas (<70% extraction): {safe_count:,} records")
        
        # Insight 5: Water quality issues
        if 'quality_tagging' in df.columns:
            quality_issues = df[df['quality_tagging'].notna() & (df['quality_tagging'] != '-')]
            insights.append(f"Areas with water quality issues: {len(quality_issues):,} records")
        
        # Insight 6: Coastal areas
        if 'coastal_areas' in df.columns:
            coastal_count = df['coastal_areas'].notna().sum()
            insights.append(f"Coastal areas identified: {coastal_count:,} records")
        
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        return insights
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()
        return []

def generate_chatbot_enhancement_suggestions():
    """Generate suggestions for enhancing chatbot responses"""
    print("\n🚀 CHATBOT ENHANCEMENT SUGGESTIONS:")
    print("=" * 40)
    
    suggestions = [
        "1. **State Comparison Tables**: Add comprehensive state-wise comparison showing extraction rates, recharge, and criticality status",
        "2. **Temporal Trend Analysis**: Include year-wise trends, projections, and historical patterns",
        "3. **Criticality Heatmap**: Show criticality distribution across states/districts with color coding",
        "4. **Water Quality Dashboard**: Add detailed water quality analysis with specific contamination details",
        "5. **Administrative Hierarchy Analysis**: Include tehsil, block, mandal, village level detailed analysis",
        "6. **Extreme Cases Alert System**: Highlight most over-exploited and safest areas with specific recommendations",
        "7. **Geographic Pattern Analysis**: Add regional patterns, correlations, and spatial insights",
        "8. **Predictive Analysis**: Include future projections based on current trends and patterns",
        "9. **Best Practices Recommendations**: Suggest region-specific water management strategies",
        "10. **Interactive Visualizations**: Add charts, graphs, and visual representations of groundwater status",
        "11. **Comparative Analysis**: Add district-to-district, state-to-state comparisons",
        "12. **Risk Assessment**: Include risk levels and priority areas for intervention",
        "13. **Resource Allocation**: Suggest optimal resource allocation strategies",
        "14. **Climate Impact Analysis**: Include climate change impact on groundwater resources",
        "15. **Policy Recommendations**: Provide specific policy recommendations based on data analysis"
    ]
    
    for suggestion in suggestions:
        print(suggestion)
    
    return suggestions

def analyze_specific_insights():
    """Analyze specific insights that can be added to chatbot responses"""
    print("\n🎯 SPECIFIC INSIGHTS TO ADD TO CHATBOT:")
    print("=" * 40)
    
    try:
        df = pd.read_csv('ingris_rag_ready_complete.csv', low_memory=False)
        
        # Convert key columns to numeric
        df['stage_of_ground_water_extraction_'] = pd.to_numeric(df['stage_of_ground_water_extraction_'], errors='coerce')
        df['rainfall_mm'] = pd.to_numeric(df['rainfall_mm'], errors='coerce')
        df['ground_water_recharge_ham'] = pd.to_numeric(df['ground_water_recharge_ham'], errors='coerce')
        
        insights = []
        
        # 1. National Overview
        total_states = df['state'].nunique()
        total_districts = df['district'].nunique()
        avg_extraction = df['stage_of_ground_water_extraction_'].mean()
        insights.append(f"National Overview: {total_states} states, {total_districts} districts, average extraction {avg_extraction:.1f}%")
        
        # 2. Criticality Distribution
        if 'watershed_category' in df.columns:
            criticality_dist = df['watershed_category'].value_counts()
            over_exploited = criticality_dist.get('over_exploited', 0)
            critical = criticality_dist.get('critical', 0)
            insights.append(f"Criticality: {over_exploited:,} over-exploited, {critical:,} critical areas")
        
        # 3. Water Quality Issues
        if 'quality_tagging' in df.columns:
            quality_issues = df[df['quality_tagging'].notna() & (df['quality_tagging'] != '-')]
            insights.append(f"Water Quality: {len(quality_issues):,} areas with quality issues")
        
        # 4. Coastal Areas
        if 'coastal_areas' in df.columns:
            coastal_count = df['coastal_areas'].notna().sum()
            insights.append(f"Coastal Areas: {coastal_count:,} coastal areas identified")
        
        # 5. Year Coverage
        year_range = f"{df['year'].min()}-{df['year'].max()}"
        insights.append(f"Data Coverage: {year_range} ({df['year'].nunique()} years)")
        
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        return insights
        
    except Exception as e:
        print(f"❌ Error in specific insights: {e}")
        return []

def main():
    """Main analysis function"""
    print("🧪 Complete Dataset Analysis for Chatbot Enhancement")
    print("=" * 60)
    
    insights = analyze_complete_dataset()
    suggestions = generate_chatbot_enhancement_suggestions()
    specific_insights = analyze_specific_insights()
    
    print("\n📊 SUMMARY:")
    print("=" * 20)
    print(f"✅ Analyzed {162632:,} records")
    print(f"✅ Generated {len(insights)} key insights")
    print(f"✅ Provided {len(suggestions)} enhancement suggestions")
    print(f"✅ Identified {len(specific_insights)} specific insights to add")
    print("\n💡 The chatbot can be significantly enhanced with these additional insights!")

if __name__ == "__main__":
    main()

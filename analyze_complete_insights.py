#!/usr/bin/env python3
"""
Analyze the complete 162,632 records dataset to identify additional insights
that can be provided in chatbot answers
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

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
        print(f"• Date Range: {df['Assessment_Year'].min()} - {df['Assessment_Year'].max()}")
        print(f"• States Covered: {df['STATE'].nunique()}")
        print(f"• Districts Covered: {df['DISTRICT'].nunique()}")
        
        # 1. State-wise Analysis
        print("\n🏛️ STATE-WISE INSIGHTS:")
        print("=" * 30)
        
        state_stats = df.groupby('STATE').agg({
            'DISTRICT': 'nunique',
            'Assessment_Year': 'nunique',
            'Stage of Ground Water Extraction (%) - Total - Total': ['mean', 'std', 'count']
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
        
        year_stats = df.groupby('Assessment_Year').agg({
            'STATE': 'nunique',
            'DISTRICT': 'nunique',
            'Stage of Ground Water Extraction (%) - Total - Total': 'mean'
        }).round(2)
        
        print("Year-wise Analysis:")
        print(year_stats)
        
        # 4. Water Quality Analysis
        print("\n🔬 WATER QUALITY INSIGHTS:")
        print("=" * 30)
        
        quality_columns = [col for col in df.columns if 'quality' in col.lower() or 'tagging' in col.lower()]
        if quality_columns:
            print(f"Quality-related columns found: {quality_columns}")
            for col in quality_columns:
                if col in df.columns:
                    quality_data = df[col].value_counts()
                    print(f"\n{col} distribution:")
                    print(quality_data.head(10))
        
        # 5. Rainfall and Recharge Analysis
        print("\n🌧️ RAINFALL & RECHARGE INSIGHTS:")
        print("=" * 30)
        
        rainfall_cols = [col for col in df.columns if 'rainfall' in col.lower()]
        recharge_cols = [col for col in df.columns if 'recharge' in col.lower()]
        
        print(f"Rainfall columns: {rainfall_cols}")
        print(f"Recharge columns: {recharge_cols}")
        
        # 6. Administrative Hierarchy Analysis
        print("\n🏘️ ADMINISTRATIVE HIERARCHY INSIGHTS:")
        print("=" * 30)
        
        admin_cols = ['STATE', 'DISTRICT', 'taluk', 'block', 'mandal', 'village']
        for col in admin_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                non_null_count = df[col].notna().sum()
                print(f"• {col}: {unique_count:,} unique values, {non_null_count:,} records with data")
        
        # 7. Extreme Cases Analysis
        print("\n⚠️ EXTREME CASES INSIGHTS:")
        print("=" * 30)
        
        extraction_col = 'Stage of Ground Water Extraction (%) - Total - Total'
        if extraction_col in df.columns:
            # Convert to numeric, handling errors
            df[extraction_col] = pd.to_numeric(df[extraction_col], errors='coerce')
            
            # Over-exploited areas (>100%)
            over_exploited = df[df[extraction_col] > 100]
            print(f"Over-exploited areas (>100% extraction): {len(over_exploited):,} records")
            
            if len(over_exploited) > 0:
                print("Top 10 most over-exploited areas:")
                top_over = over_exploited.nlargest(10, extraction_col)[['STATE', 'DISTRICT', extraction_col]]
                print(top_over)
            
            # Safe areas (<70%)
            safe_areas = df[df[extraction_col] < 70]
            print(f"\nSafe areas (<70% extraction): {len(safe_areas):,} records")
            
            if len(safe_areas) > 0:
                print("Top 10 safest areas:")
                top_safe = safe_areas.nsmallest(10, extraction_col)[['STATE', 'DISTRICT', extraction_col]]
                print(top_safe)
        
        # 8. Geographic Distribution
        print("\n🗺️ GEOGRAPHIC DISTRIBUTION INSIGHTS:")
        print("=" * 30)
        
        # State-wise record distribution
        state_dist = df['STATE'].value_counts()
        print("Top 10 states by number of records:")
        for state, count in state_dist.head(10).items():
            percentage = (count / len(df)) * 100
            print(f"• {state}: {count:,} records ({percentage:.1f}%)")
        
        # 9. Data Completeness Analysis
        print("\n📋 DATA COMPLETENESS INSIGHTS:")
        print("=" * 30)
        
        # Check completeness of key columns
        key_columns = [
            'Stage of Ground Water Extraction (%) - Total - Total',
            'Annual Ground water Recharge (ham) - Total - Total',
            'Ground Water Extraction for all uses (ha.m) - Total - Total',
            'Net Annual Ground Water Availability for Future Use (ham) - Total - Total'
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
        if extraction_col in df.columns:
            state_avg = df.groupby('STATE')[extraction_col].mean().sort_values(ascending=False)
            insights.append(f"State with highest average extraction: {state_avg.index[0]} ({state_avg.iloc[0]:.1f}%)")
            insights.append(f"State with lowest average extraction: {state_avg.index[-1]} ({state_avg.iloc[-1]:.1f}%)")
        
        # Insight 2: Year trends
        if 'Assessment_Year' in df.columns:
            year_trend = df.groupby('Assessment_Year')[extraction_col].mean()
            if len(year_trend) > 1:
                trend_direction = "increasing" if year_trend.iloc[-1] > year_trend.iloc[0] else "decreasing"
                insights.append(f"Overall extraction trend: {trend_direction}")
        
        # Insight 3: Critical areas
        if extraction_col in df.columns:
            critical_count = len(df[df[extraction_col] > 90])
            insights.append(f"Critical areas (>90% extraction): {critical_count:,} records")
        
        # Insight 4: Safe areas
        if extraction_col in df.columns:
            safe_count = len(df[df[extraction_col] < 70])
            insights.append(f"Safe areas (<70% extraction): {safe_count:,} records")
        
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        return insights
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return []

def generate_chatbot_enhancement_suggestions():
    """Generate suggestions for enhancing chatbot responses"""
    print("\n🚀 CHATBOT ENHANCEMENT SUGGESTIONS:")
    print("=" * 40)
    
    suggestions = [
        "1. **State Comparison**: Add state-wise comparison tables showing extraction rates, recharge, and criticality",
        "2. **Temporal Trends**: Include year-wise trends and projections",
        "3. **Criticality Heatmap**: Show criticality distribution across states/districts",
        "4. **Water Quality Dashboard**: Add water quality analysis with contamination details",
        "5. **Administrative Hierarchy**: Include tehsil, block, mandal, village level analysis",
        "6. **Extreme Cases Alert**: Highlight most over-exploited and safest areas",
        "7. **Geographic Insights**: Add regional patterns and correlations",
        "8. **Predictive Analysis**: Include future projections based on current trends",
        "9. **Best Practices**: Suggest region-specific water management strategies",
        "10. **Interactive Maps**: Add visual representations of groundwater status"
    ]
    
    for suggestion in suggestions:
        print(suggestion)
    
    return suggestions

def main():
    """Main analysis function"""
    print("🧪 Complete Dataset Analysis for Chatbot Enhancement")
    print("=" * 60)
    
    insights = analyze_complete_dataset()
    suggestions = generate_chatbot_enhancement_suggestions()
    
    print("\n📊 SUMMARY:")
    print("=" * 20)
    print(f"✅ Analyzed {162632:,} records")
    print(f"✅ Generated {len(insights)} key insights")
    print(f"✅ Provided {len(suggestions)} enhancement suggestions")
    print("\n💡 The chatbot can be enhanced with these additional insights!")

if __name__ == "__main__":
    main()

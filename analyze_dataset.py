#!/usr/bin/env python3
"""
Analyze the groundwater dataset to understand what questions the chatbot should answer
"""

import pandas as pd
import numpy as np

def analyze_groundwater_dataset():
    """Analyze the groundwater dataset comprehensively"""
    
    print("🔍 ANALYZING GROUNDWATER DATASET")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('master_groundwater_data.csv', low_memory=False)
    
    print(f"📊 Dataset Overview:")
    print(f"   Total Records: {len(df):,}")
    print(f"   Total Columns: {len(df.columns)}")
    print(f"   States Covered: {df['STATE'].nunique()}")
    print(f"   Districts Covered: {df['DISTRICT'].nunique()}")
    print(f"   Assessment Year: {df['Assessment_Year'].iloc[0] if 'Assessment_Year' in df.columns else 'Unknown'}")
    
    # Key metrics columns
    key_columns = {
        'Stage of Ground Water Extraction (%) - Total - Total': 'Extraction Stage (%)',
        'Annual Ground water Recharge (ham) - Total - Total': 'Annual Recharge (ham)',
        'Annual Extractable Ground water Resource (ham) - Total - Total': 'Extractable Resource (ham)',
        'Ground Water Extraction for all uses (ha.m) - Total - Total': 'Total Extraction (ha.m)',
        'Net Annual Ground Water Availability for Future Use (ham) - Total - Total': 'Future Availability (ham)',
        'Rainfall (mm) - Total': 'Annual Rainfall (mm)',
        'Total Geographical Area (ha) - Total - Total': 'Total Area (ha)'
    }
    
    print(f"\n📈 Key Metrics Analysis:")
    for col, display_name in key_columns.items():
        if col in df.columns:
            non_null = df[col].notna().sum()
            if non_null > 0:
                data = df[col].dropna()
                print(f"   {display_name}:")
                print(f"     - Valid records: {non_null:,}")
                print(f"     - Range: {data.min():.2f} to {data.max():.2f}")
                print(f"     - Mean: {data.mean():.2f}")
                print(f"     - Median: {data.median():.2f}")
            else:
                print(f"   {display_name}: No valid data")
        else:
            print(f"   {display_name}: Column not found")
    
    # Criticality analysis
    print(f"\n🚨 GROUNDWATER CRITICALITY ANALYSIS:")
    extraction_col = 'Stage of Ground Water Extraction (%) - Total - Total'
    
    if extraction_col in df.columns:
        extraction_data = df[extraction_col].dropna()
        print(f"   Valid extraction data: {len(extraction_data):,} districts")
        
        # Categorize by criticality
        safe = extraction_data[extraction_data < 70].count()
        semi_critical = extraction_data[(extraction_data >= 70) & (extraction_data < 90)].count()
        critical = extraction_data[extraction_data >= 90].count()
        over_exploited = extraction_data[extraction_data >= 100].count()
        
        print(f"\n   📊 Criticality Distribution:")
        print(f"   🟢 Safe (<70%): {safe:,} districts ({safe/len(extraction_data)*100:.1f}%)")
        print(f"   🟡 Semi-Critical (70-90%): {semi_critical:,} districts ({semi_critical/len(extraction_data)*100:.1f}%)")
        print(f"   🔴 Critical (90-100%): {critical:,} districts ({critical/len(extraction_data)*100:.1f}%)")
        print(f"   ⚫ Over-Exploited (≥100%): {over_exploited:,} districts ({over_exploited/len(extraction_data)*100:.1f}%)")
        
        # State-wise criticality
        print(f"\n   🏛️ State-wise Criticality (Top 10):")
        state_criticality = df.groupby('STATE')[extraction_col].agg(['count', 'mean', 'max']).reset_index()
        state_criticality = state_criticality[state_criticality['count'] >= 5]  # States with at least 5 districts
        state_criticality = state_criticality.sort_values('mean', ascending=False)
        
        for _, row in state_criticality.head(10).iterrows():
            state = row['STATE']
            mean_extraction = row['mean']
            max_extraction = row['max']
            district_count = int(row['count'])
            
            if mean_extraction >= 100:
                status = "⚫ Over-Exploited"
            elif mean_extraction >= 90:
                status = "🔴 Critical"
            elif mean_extraction >= 70:
                status = "🟡 Semi-Critical"
            else:
                status = "🟢 Safe"
                
            print(f"     {state}: {mean_extraction:.1f}% avg, {max_extraction:.1f}% max ({district_count} districts) {status}")
    
    # Water quality analysis
    print(f"\n💧 WATER QUALITY ANALYSIS:")
    quality_columns = [
        'Quality Tagging - Major Parameter Present - C',
        'Quality Tagging - Major Parameter Present - NC', 
        'Quality Tagging - Major Parameter Present - PQ',
        'Quality Tagging - Other Parameters Present - C',
        'Quality Tagging - Other Parameters Present - NC',
        'Quality Tagging - Other Parameters Present - PQ'
    ]
    
    for col in quality_columns:
        if col in df.columns:
            non_null = df[col].notna().sum()
            if non_null > 0:
                unique_values = df[col].dropna().unique()
                print(f"   {col}: {non_null} records, {len(unique_values)} unique values")
                if len(unique_values) <= 10:
                    print(f"     Values: {list(unique_values)}")
    
    # Resource availability analysis
    print(f"\n💧 RESOURCE AVAILABILITY ANALYSIS:")
    recharge_col = 'Annual Ground water Recharge (ham) - Total - Total'
    extractable_col = 'Annual Extractable Ground water Resource (ham) - Total - Total'
    extraction_col = 'Ground Water Extraction for all uses (ha.m) - Total - Total'
    
    if all(col in df.columns for col in [recharge_col, extractable_col, extraction_col]):
        # Calculate efficiency and sustainability metrics
        df_analysis = df[[recharge_col, extractable_col, extraction_col, 'STATE', 'DISTRICT']].dropna()
        
        if len(df_analysis) > 0:
            # Calculate extraction efficiency
            df_analysis['extraction_efficiency'] = (df_analysis[extraction_col] / df_analysis[extractable_col] * 100).replace([np.inf, -np.inf], np.nan)
            
            print(f"   📊 Resource Efficiency (Top 10 Districts):")
            top_efficient = df_analysis.nlargest(10, 'extraction_efficiency')[['STATE', 'DISTRICT', 'extraction_efficiency']]
            for _, row in top_efficient.iterrows():
                print(f"     {row['STATE']} - {row['DISTRICT']}: {row['extraction_efficiency']:.1f}%")
    
    return df

def generate_chatbot_questions():
    """Generate possible questions the chatbot should answer"""
    
    print(f"\n🤖 POSSIBLE CHATBOT QUESTIONS:")
    print("=" * 60)
    
    questions = {
        "Groundwater Status Questions": [
            "What is the groundwater extraction stage in [State/District]?",
            "Is [State/District] groundwater safe, semi-critical, or critical?",
            "How much groundwater is being extracted in [State/District]?",
            "What percentage of groundwater resources are being used in [State/District]?",
            "Is [State/District] over-exploiting its groundwater resources?"
        ],
        
        "Resource Availability Questions": [
            "What is the annual groundwater recharge in [State/District]?",
            "How much extractable groundwater resource is available in [State/District]?",
            "What is the net groundwater availability for future use in [State/District]?",
            "What is the total geographical area of [State/District]?",
            "How much rainfall does [State/District] receive annually?"
        ],
        
        "Water Quality Questions": [
            "What are the major water quality parameters in [State/District]?",
            "Are there any water quality issues in [State/District]?",
            "What other water quality parameters are present in [State/District]?",
            "Is the groundwater in [State/District] suitable for drinking?"
        ],
        
        "Comparative Analysis Questions": [
            "Which districts in [State] have the highest groundwater extraction?",
            "Which states have the most critical groundwater situation?",
            "Compare groundwater status between [State1] and [State2]",
            "What is the national average groundwater extraction stage?",
            "Which regions have the best groundwater management?"
        ],
        
        "Improvement Recommendations": [
            "How can [State/District] improve its groundwater situation?",
            "What measures should be taken for critical groundwater areas?",
            "What are the best practices for groundwater conservation?",
            "How can groundwater recharge be increased in [State/District]?",
            "What policies should be implemented for sustainable groundwater use?"
        ]
    }
    
    for category, question_list in questions.items():
        print(f"\n📋 {category}:")
        for i, question in enumerate(question_list, 1):
            print(f"   {i}. {question}")
    
    print(f"\n💡 CRITICALITY ASSESSMENT CRITERIA:")
    print("   🟢 Safe: <70% extraction")
    print("   🟡 Semi-Critical: 70-90% extraction") 
    print("   🔴 Critical: 90-100% extraction")
    print("   ⚫ Over-Exploited: ≥100% extraction")
    
    print(f"\n📊 NUMERICAL ANSWERS TO PROVIDE:")
    print("   • Extraction percentage with criticality status")
    print("   • Annual recharge in ham (hectare-meters)")
    print("   • Extractable resource in ham")
    print("   • Total extraction in ha.m")
    print("   • Future availability in ham")
    print("   • Rainfall in mm")
    print("   • Area in hectares")
    print("   • District count and rankings")
    
    print(f"\n🔧 IMPROVEMENT RECOMMENDATIONS:")
    print("   • Water conservation measures")
    print("   • Rainwater harvesting")
    print("   • Artificial recharge techniques")
    print("   • Crop pattern optimization")
    print("   • Groundwater monitoring")
    print("   • Policy interventions")
    print("   • Community awareness programs")

if __name__ == "__main__":
    df = analyze_groundwater_dataset()
    generate_chatbot_questions()

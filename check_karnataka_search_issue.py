#!/usr/bin/env python3
"""
Check why Karnataka search isn't finding records with Taluk data
"""

import pandas as pd

def check_karnataka_search_issue():
    """Check why the search isn't finding records with available data"""
    print("🔍 Checking Karnataka Search Issue")
    print("=" * 50)
    
    df = pd.read_csv('ingris_rag_ready_complete.csv', low_memory=False)
    
    # Find Karnataka records
    karnataka = df[df['state'].str.contains('KARNATAKA', case=False, na=False)]
    print(f"Total Karnataka records: {len(karnataka)}")
    
    # Check what data is actually available
    print("\n📊 Data Availability in Karnataka:")
    print(f"  Taluk data: {karnataka['taluk'].notna().sum()} records")
    print(f"  Block data: {karnataka['block'].notna().sum()} records")
    print(f"  Mandal data: {karnataka['mandal'].notna().sum()} records")
    print(f"  Village data: {karnataka['village'].notna().sum()} records")
    
    # Find records with Taluk data
    taluk_records = karnataka[karnataka['taluk'].notna() & (karnataka['taluk'] != 'nan')]
    print(f"\n📋 Records with Taluk data: {len(taluk_records)}")
    
    if len(taluk_records) > 0:
        print("\nDistricts with Taluk data:")
        districts_with_taluk = taluk_records['district'].value_counts()
        print(districts_with_taluk.head(10))
        
        print("\nSample records with Taluk data:")
        sample = taluk_records[['state', 'district', 'taluk', 'block', 'mandal', 'village']].head(5)
        for idx, row in sample.iterrows():
            print(f"  {row['district']}: Taluk={row['taluk']}, Block={row['block']}, Mandal={row['mandal']}, Village={row['village']}")
    
    # Check if the search results are including these records
    print("\n🔍 Simulating Search Results:")
    
    # Simulate what the search might return
    # Let's check if the search is prioritizing records without taluk data
    records_without_taluk = karnataka[karnataka['taluk'].isna() | (karnataka['taluk'] == 'nan')]
    records_with_taluk = karnataka[karnataka['taluk'].notna() & (karnataka['taluk'] != 'nan')]
    
    print(f"Records WITHOUT taluk data: {len(records_without_taluk)}")
    print(f"Records WITH taluk data: {len(records_with_taluk)}")
    
    # Check if the search is biased towards records without taluk data
    print("\n🎯 Search Bias Analysis:")
    print("If search is returning records without taluk data, that explains why 'No data available' appears")
    
    # Show sample of records that might be returned by search
    print("\nSample records that might be returned by search (first 5):")
    sample_search = karnataka.head(5)
    for idx, row in sample_search.iterrows():
        print(f"  {row['district']}: Taluk={row['taluk']}, Block={row['block']}, Mandal={row['mandal']}, Village={row['village']}")

def check_search_algorithm():
    """Check if the search algorithm is working correctly"""
    print("\n🔍 Search Algorithm Analysis")
    print("=" * 50)
    
    df = pd.read_csv('ingris_rag_ready_complete.csv', low_memory=False)
    
    # Find Karnataka records
    karnataka = df[df['state'].str.contains('KARNATAKA', case=False, na=False)]
    
    # Check if the search is finding records with the most complete data
    print("Checking data completeness by record:")
    
    # Calculate completeness score for each record
    completeness_scores = []
    for idx, row in karnataka.iterrows():
        score = 0
        if pd.notna(row['taluk']) and row['taluk'] != 'nan':
            score += 1
        if pd.notna(row['block']) and row['block'] != 'nan':
            score += 1
        if pd.notna(row['mandal']) and row['mandal'] != 'nan':
            score += 1
        if pd.notna(row['village']) and row['village'] != 'nan':
            score += 1
        completeness_scores.append((idx, score, row['district'], row['taluk'], row['block'], row['mandal'], row['village']))
    
    # Sort by completeness score
    completeness_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 most complete records:")
    for i, (idx, score, district, taluk, block, mandal, village) in enumerate(completeness_scores[:10]):
        print(f"  {i+1}. {district}: Score={score}, Taluk={taluk}, Block={block}, Mandal={mandal}, Village={village}")
    
    print("\nBottom 10 least complete records:")
    for i, (idx, score, district, taluk, block, mandal, village) in enumerate(completeness_scores[-10:]):
        print(f"  {i+1}. {district}: Score={score}, Taluk={taluk}, Block={block}, Mandal={mandal}, Village={village}")

def main():
    """Main function"""
    check_karnataka_search_issue()
    check_search_algorithm()
    
    print("\n💡 Recommendations:")
    print("1. The search should prioritize records with more complete data")
    print("2. If records with taluk data exist, they should be included in search results")
    print("3. The search algorithm might need to be adjusted to find the most informative records")

if __name__ == "__main__":
    main()

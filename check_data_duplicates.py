#!/usr/bin/env python3
"""
Check for duplicate data entries that might be causing repetition
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main2 import search_excel_chunks
import pandas as pd

def check_data_duplicates():
    """Check for duplicate data entries"""
    print("🔍 Checking for Data Duplicates")
    print("=" * 50)
    
    # Test query for Karnataka
    query = "groundwater estimation in karnataka"
    
    print(f"🔍 Testing query: '{query}'")
    print("-" * 30)
    
    try:
        # Perform search
        results = search_excel_chunks(query, target_state="KARNATAKA")
        
        if not results:
            print("❌ No results found")
            return
        
        print(f"✅ Found {len(results)} results")
        
        # Check for duplicates
        print("\n🔍 Checking for duplicate entries:")
        print("-" * 40)
        
        # Track unique combinations
        unique_combinations = set()
        duplicates = []
        
        for i, result in enumerate(results):
            if 'data' in result:
                data = result['data']
                # Create a unique key based on key fields
                key_fields = ['state', 'district', 'taluk', 'year']
                key_values = []
                for field in key_fields:
                    value = data.get(field, 'N/A')
                    key_values.append(str(value))
                
                unique_key = '|'.join(key_values)
                
                if unique_key in unique_combinations:
                    duplicates.append({
                        'index': i,
                        'key': unique_key,
                        'data': data
                    })
                else:
                    unique_combinations.add(unique_key)
        
        print(f"📊 Analysis Results:")
        print(f"  Total results: {len(results)}")
        print(f"  Unique combinations: {len(unique_combinations)}")
        print(f"  Duplicates found: {len(duplicates)}")
        
        if duplicates:
            print(f"\n❌ Duplicate entries found:")
            for dup in duplicates[:5]:  # Show first 5 duplicates
                data = dup['data']
                print(f"  Index {dup['index']}: {data.get('state', 'N/A')}, {data.get('district', 'N/A')}, {data.get('taluk', 'N/A')}, {data.get('year', 'N/A')}")
        else:
            print(f"\n✅ No duplicate entries found")
        
        # Check for similar districts with different data
        print(f"\n🔍 Checking for similar districts:")
        print("-" * 40)
        
        district_data = {}
        for result in results:
            if 'data' in result:
                data = result['data']
                district = data.get('district', 'N/A')
                taluk = data.get('taluk', 'N/A')
                
                if district not in district_data:
                    district_data[district] = []
                district_data[district].append(taluk)
        
        for district, taluks in district_data.items():
            if len(taluks) > 1:
                print(f"  {district}: {len(taluks)} entries - Taluks: {list(set(taluks))}")
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_data_duplicates()

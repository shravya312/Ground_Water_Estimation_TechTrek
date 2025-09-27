#!/usr/bin/env python3
"""
Check what data Karnataka actually has
"""

import pandas as pd

def check_karnataka_data():
    """Check Karnataka data availability"""
    print("🔍 Checking Karnataka Data")
    print("=" * 30)
    
    df = pd.read_csv('ingris_rag_ready_complete.csv', low_memory=False)
    
    # Find Karnataka records
    karnataka = df[df['state'].str.contains('KARNATAKA', case=False, na=False)]
    print(f"Total Karnataka records: {len(karnataka)}")
    
    if len(karnataka) > 0:
        print("\n📊 Data Coverage:")
        print(f"  Storage data: {karnataka['instorage_unconfined_ground_water_resourcesham'].notna().sum()} non-null")
        print(f"  Watershed category: {karnataka['watershed_category'].notna().sum()} non-null")
        print(f"  Mandal: {karnataka['mandal'].notna().sum()} non-null")
        print(f"  Village: {karnataka['village'].notna().sum()} non-null")
        print(f"  Taluk: {karnataka['taluk'].notna().sum()} non-null")
        print(f"  Block: {karnataka['block'].notna().sum()} non-null")
        
        print("\n📋 Sample Karnataka Record:")
        sample = karnataka.head(1).iloc[0]
        print(f"  State: {sample['state']}")
        print(f"  District: {sample['district']}")
        print(f"  Storage: {sample['instorage_unconfined_ground_water_resourcesham']}")
        print(f"  Watershed Category: {sample['watershed_category']}")
        print(f"  Mandal: {sample['mandal']}")
        print(f"  Village: {sample['village']}")
        print(f"  Taluk: {sample['taluk']}")
        print(f"  Block: {sample['block']}")
        
        # Check for any non-zero storage values
        storage_values = pd.to_numeric(karnataka['instorage_unconfined_ground_water_resourcesham'], errors='coerce')
        non_zero_storage = (storage_values > 0).sum()
        print(f"\n💧 Non-zero storage values: {non_zero_storage}")
        
        # Check for non-null watershed categories
        watershed_cats = karnataka['watershed_category'].dropna()
        non_dash_watershed = watershed_cats[watershed_cats != '-'].count()
        print(f"🏞️ Non-dash watershed categories: {non_dash_watershed}")
        
        if non_dash_watershed > 0:
            print(f"   Sample watershed categories: {list(watershed_cats[watershed_cats != '-'].unique()[:5])}")

if __name__ == "__main__":
    check_karnataka_data()

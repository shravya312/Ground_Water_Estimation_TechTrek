#!/usr/bin/env python3
"""
Check what data is in the Qdrant collection
"""

import main

def check_collection_data():
    """Check what data is available in the collection."""
    print("🔍 Checking Data in Qdrant Collection")
    print("=" * 50)
    
    # Initialize components
    main._init_components()
    
    # Get sample data
    results = main._qdrant_client.scroll(
        collection_name='groundwater_excel_collection', 
        limit=20, 
        with_payload=True
    )
    
    print(f"📊 Total points in collection: {main._qdrant_client.get_collection('groundwater_excel_collection').points_count}")
    print("\n📋 Sample Records:")
    print("-" * 50)
    
    states = set()
    districts = set()
    years = set()
    
    for i, point in enumerate(results[0][:10]):
        payload = point.payload
        state = payload.get("STATE", "N/A")
        district = payload.get("DISTRICT", "N/A")
        year = payload.get("Assessment_Year", "N/A")
        
        states.add(state)
        districts.add(district)
        years.add(year)
        
        print(f"Record {i+1}:")
        print(f"  State: {state}")
        print(f"  District: {district}")
        print(f"  Year: {year}")
        print()
    
    print("📈 Data Summary:")
    print(f"  Unique States: {len(states)} - {sorted(list(states))}")
    print(f"  Unique Districts: {len(districts)} - {sorted(list(districts))[:10]}...")
    print(f"  Unique Years: {len(years)} - {sorted(list(years))}")
    
    # Check specifically for Karnataka
    karnataka_count = 0
    for point in results[0]:
        if point.payload.get("STATE", "").lower() == "karnataka":
            karnataka_count += 1
    
    print(f"\n🔍 Karnataka Records: {karnataka_count}")

if __name__ == "__main__":
    check_collection_data()

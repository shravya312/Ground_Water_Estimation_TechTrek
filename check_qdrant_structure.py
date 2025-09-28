#!/usr/bin/env python3
"""
Check the Qdrant collection structure and data
"""

import main2

def check_qdrant_structure():
    """Check the Qdrant collection structure and sample data."""
    print("🔍 Checking Qdrant Collection Structure")
    print("=" * 50)
    
    # Initialize components
    main2._init_components()
    
    # Get collection info
    info = main2._qdrant_client.get_collection('groundwater_excel_collection')
    print(f"📊 Collection Name: groundwater_excel_collection")
    print(f"📊 Vector Size: {info.config.params.vectors.size}")
    print(f"📊 Distance Metric: {info.config.params.vectors.distance}")
    print(f"📊 Total Points: {info.points_count}")
    
    # Get sample data
    print("\n📋 Sample Data in Collection:")
    print("-" * 50)
    
    results = main2._qdrant_client.scroll(
        collection_name='groundwater_excel_collection', 
        limit=10, 
        with_payload=True
    )
    
    states = set()
    districts = set()
    years = set()
    
    for i, point in enumerate(results[0][:5]):
        payload = point.payload
        state = payload.get("STATE", "N/A")
        district = payload.get("DISTRICT", "N/A")
        year = payload.get("Assessment_Year", "N/A")
        
        states.add(state)
        districts.add(district)
        years.add(year)
        
        print(f"\nRecord {i+1}:")
        print(f"  STATE: {state}")
        print(f"  DISTRICT: {district}")
        print(f"  Assessment_Year: {year}")
        
        # Show text preview
        text = payload.get("text", "N/A")
        if text != "N/A":
            print(f"  Text preview: {text[:150]}...")
        else:
            print(f"  Text: {text}")
    
    # Summary
    print(f"\n📈 Data Summary:")
    print(f"  Unique States: {len(states)} - {sorted(list(states))}")
    print(f"  Unique Districts: {len(districts)} - {sorted(list(districts))[:10]}...")
    print(f"  Unique Years: {len(years)} - {sorted(list(years))}")
    
    # Check for Karnataka specifically
    karnataka_count = 0
    for point in results[0]:
        if point.payload.get("STATE", "").upper() == "KARNATAKA":
            karnataka_count += 1
    
    print(f"\n🔍 Karnataka Records: {karnataka_count}")
    
    # Check all data for states
    print(f"\n🔍 Checking all data for states...")
    all_results = main2._qdrant_client.scroll(
        collection_name='groundwater_excel_collection', 
        limit=1000, 
        with_payload=True
    )
    
    all_states = set()
    for point in all_results[0]:
        state = point.payload.get("STATE", "N/A")
        if state != "N/A":
            all_states.add(state)
    
    print(f"📊 All States in Collection: {sorted(list(all_states))}")
    
    # Check if this is the old data (Andaman only) or new data (all states)
    if len(all_states) == 1 and "ANDAMAN" in str(list(all_states)[0]).upper():
        print(f"\n⚠️  WARNING: Collection contains only Andaman data!")
        print(f"   This is the old limited dataset.")
        print(f"   Need to upload master CSV with all states.")
    elif len(all_states) > 10:
        print(f"\n✅ GOOD: Collection contains data for {len(all_states)} states!")
        print(f"   This appears to be the comprehensive dataset.")
    else:
        print(f"\n⚠️  Collection contains limited data for {len(all_states)} states.")

if __name__ == "__main__":
    check_qdrant_structure()

#!/usr/bin/env python3
"""
Check the actual Qdrant payload structure
"""

import main2

def check_qdrant_payload():
    """Check the actual payload structure in Qdrant."""
    print("🔍 Checking Qdrant Payload Structure")
    print("=" * 50)
    
    # Initialize components
    main2._init_components()
    
    # Get sample data
    results = main2._qdrant_client.scroll(
        collection_name='groundwater_excel_collection', 
        limit=5, 
        with_payload=True
    )
    
    print("📋 Sample Payloads:")
    print("-" * 50)
    
    for i, point in enumerate(results[0]):
        payload = point.payload
        print(f"\nRecord {i+1}:")
        print(f"  Payload keys: {list(payload.keys())}")
        
        # Check if it has individual fields or just combined text
        if 'STATE' in payload:
            print(f"  STATE: {payload['STATE']}")
        if 'DISTRICT' in payload:
            print(f"  DISTRICT: {payload['DISTRICT']}")
        if 'Assessment_Year' in payload:
            print(f"  Assessment_Year: {payload['Assessment_Year']}")
        
        # Show the text content
        text = payload.get('text', 'N/A')
        print(f"  Text content: {text[:300]}...")
        
        # Try to extract state from text
        if 'STATE' not in payload and 'text' in payload:
            text_lower = text.lower()
            if 'karnataka' in text_lower:
                print(f"  🎯 KARNATAKA found in text!")
            elif 'maharashtra' in text_lower:
                print(f"  🎯 MAHARASHTRA found in text!")
            elif 'gujarat' in text_lower:
                print(f"  🎯 GUJARAT found in text!")
            elif 'rajasthan' in text_lower:
                print(f"  🎯 RAJASTHAN found in text!")
            elif 'tamil' in text_lower:
                print(f"  🎯 TAMIL NADU found in text!")
    
    # Check for Karnataka in all data
    print(f"\n🔍 Searching for Karnataka in all data...")
    all_results = main2._qdrant_client.scroll(
        collection_name='groundwater_excel_collection', 
        limit=1000, 
        with_payload=True
    )
    
    karnataka_found = 0
    states_found = set()
    
    for point in all_results[0]:
        text = point.payload.get('text', '').lower()
        if 'karnataka' in text:
            karnataka_found += 1
            print(f"  ✅ Karnataka record found: {point.payload.get('text', '')[:100]}...")
        
        # Extract state from text
        if 'unnamed: 1:' in text:
            # This is the old format with Unnamed columns
            parts = text.split('unnamed: 1:')
            if len(parts) > 1:
                state_part = parts[1].split('|')[0].strip()
                if state_part:
                    states_found.add(state_part.upper())
    
    print(f"\n📊 Results:")
    print(f"  Karnataka records found: {karnataka_found}")
    print(f"  States found in data: {sorted(list(states_found))}")
    
    if karnataka_found > 0:
        print(f"\n✅ GOOD: Karnataka data is available in the collection!")
    else:
        print(f"\n❌ PROBLEM: No Karnataka data found in the collection!")
        print(f"   The collection contains data for: {sorted(list(states_found))}")

if __name__ == "__main__":
    check_qdrant_payload()

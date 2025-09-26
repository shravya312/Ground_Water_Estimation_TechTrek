#!/usr/bin/env python3
"""
Check what states are available in the data
"""

import pandas as pd
from collections import Counter

def check_available_states():
    """Check what states are available in the data"""
    print("🔍 Checking Available States in Data")
    print("=" * 50)
    
    try:
        # Load the CSV data
        print("🔄 Loading data from ingris_rag_ready_complete.csv...")
        df = pd.read_csv("ingris_rag_ready_complete.csv")
        print(f"✅ Loaded {len(df)} records")
        
        # Check states
        print("\n📊 State Analysis:")
        states = df['STATE'].dropna().unique()
        print(f"Total unique states: {len(states)}")
        
        # Show all states
        print("\n🏛️ All States in Data:")
        for i, state in enumerate(sorted(states), 1):
            count = len(df[df['STATE'] == state])
            print(f"{i:2d}. {state} ({count:,} records)")
        
        # Check for Chhattisgarh variations
        print("\n🔍 Checking for Chhattisgarh variations:")
        chhattisgarh_variations = [
            'CHHATTISGARH', 'CHATTISGARH', 'CHHATISGARH', 'CHHATTISGARH',
            'chhattisgarh', 'chattisgarh', 'chhatisgarh', 'chhattisgarh'
        ]
        
        found_variations = []
        for variation in chhattisgarh_variations:
            matches = df[df['STATE'].str.contains(variation, case=False, na=False)]
            if len(matches) > 0:
                found_variations.append((variation, len(matches)))
        
        if found_variations:
            print("✅ Found Chhattisgarh data:")
            for variation, count in found_variations:
                print(f"   - {variation}: {count} records")
        else:
            print("❌ No Chhattisgarh data found")
        
        # Check similar state names
        print("\n🔍 Checking for similar state names:")
        similar_states = []
        for state in states:
            if any(word in state.upper() for word in ['CHHAT', 'CHAT', 'GARH']):
                similar_states.append(state)
        
        if similar_states:
            print("States with similar names:")
            for state in similar_states:
                count = len(df[df['STATE'] == state])
                print(f"   - {state}: {count} records")
        
        # Top 10 states by record count
        print("\n📈 Top 10 States by Record Count:")
        state_counts = df['STATE'].value_counts().head(10)
        for state, count in state_counts.items():
            print(f"{state}: {count:,} records")
        
        return states.tolist()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    check_available_states()
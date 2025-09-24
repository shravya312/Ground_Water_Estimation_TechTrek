#!/usr/bin/env python3
"""
Check INGRIS State Data
"""

import pandas as pd

def check_ingris_states():
    """Check the state data in INGRIS CSV."""
    print("📊 INGRIS Data State Analysis")
    print("=" * 50)
    
    try:
        df = pd.read_csv('ingris_rag_ready.csv', low_memory=False)
        print(f"Total records: {len(df)}")
        
        print(f"\n📋 All States Found ({df['state'].nunique()} unique states):")
        states = df['state'].value_counts()
        print(states)
        
        print(f"\n✅ States with proper names:")
        valid_states = states[states.index != 'Unknown']
        print(f"   {len(valid_states)} states with proper names")
        print(f"   {valid_states.sum()} records with valid state names")
        
        print(f"\n❌ Unknown states:")
        unknown_count = states.get('Unknown', 0)
        print(f"   {unknown_count} records still have 'Unknown' state")
        
        print(f"\n📋 Top 10 States with Data:")
        for i, (state, count) in enumerate(valid_states.head(10).items(), 1):
            print(f"   {i:2d}. {state}: {count} records")
        
        # Check if we have major states
        major_states = ['KARNATAKA', 'MAHARASHTRA', 'GUJARAT', 'RAJASTHAN', 'TAMILNADU', 'BIHAR']
        print(f"\n🔍 Major States Check:")
        for state in major_states:
            if state in states.index:
                print(f"   ✅ {state}: {states[state]} records")
            else:
                print(f"   ❌ {state}: Not found")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_ingris_states()

#!/usr/bin/env python3
"""
Check the master CSV file for all available states
"""

import pandas as pd

def check_master_csv():
    """Check what states are available in the master CSV."""
    print("🔍 Checking Master CSV for All States")
    print("=" * 50)
    
    try:
        # Read the master CSV with proper data type handling
        df = pd.read_csv('master_groundwater_data.csv', low_memory=False)
        print(f"📊 Total records in master CSV: {len(df)}")
        
        # Get unique states
        states = df['STATE'].unique()
        print(f"📈 Total unique states: {len(states)}")
        
        print("\n📋 All Available States:")
        print("-" * 30)
        # Filter out NaN values and sort
        valid_states = [state for state in states if pd.notna(state)]
        for i, state in enumerate(sorted(valid_states), 1):
            print(f"{i:2d}. {state}")
        
        # Check for Karnataka specifically
        karnataka_available = "KARNATAKA" in states
        print(f"\n🔍 Karnataka Available: {karnataka_available}")
        
        if karnataka_available:
            karnataka_data = df[df['STATE'] == 'KARNATAKA']
            print(f"📊 Karnataka records: {len(karnataka_data)}")
            print(f"📋 Karnataka districts: {sorted(karnataka_data['DISTRICT'].unique())}")
        
        # Check for other major states
        major_states = ['MAHARASHTRA', 'TAMIL NADU', 'GUJARAT', 'RAJASTHAN', 'KERALA', 'ANDHRA PRADESH']
        print(f"\n🔍 Major States Available:")
        for state in major_states:
            available = state in states
            if available:
                count = len(df[df['STATE'] == state])
                print(f"  ✅ {state}: {count} records")
            else:
                print(f"  ❌ {state}: Not available")
        
        return states
        
    except Exception as e:
        print(f"❌ Error reading master CSV: {e}")
        return []

if __name__ == "__main__":
    states = check_master_csv()
    
    if states:
        print(f"\n💡 The master CSV contains data for {len(states)} states!")
        print("   This is the comprehensive dataset you should use.")
    else:
        print("\n❌ Could not read the master CSV file.")

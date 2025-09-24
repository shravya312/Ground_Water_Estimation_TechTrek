#!/usr/bin/env python3
"""
Fix INGRIS State Mapping - Map districts to correct states
"""

import pandas as pd
import re

def map_district_to_state(district):
    """Map district name to state name."""
    if pd.isna(district):
        return "Unknown"
    
    district = str(district).strip().upper()
    
    # Andhra Pradesh districts
    andhra_districts = [
        'ANANTHAPURAMU', 'CHITTOOR', 'EAST GODAVARI', 'GUNTUR', 'KRISHNA', 
        'KURNOOL', 'PRAKASAM', 'SRI POTTI SRIRAMULU NELLORE', 'SRIKAKULAM', 
        'VISAKHAPATNAM', 'VIZIANAGARAM', 'WEST GODAVARI', 'Y.S.R KADAPA'
    ]
    
    # Arunachal Pradesh districts
    arunachal_districts = [
        'ANJAW', 'CHANGLANG', 'DIBANG VALLEY', 'EAST KAMENG', 'EAST SIANG',
        'KURUNG KUMEY', 'LOHIT', 'LONGDING', 'LOWER DIBANG VALLEY', 'LOWER SUBANSIRI',
        'NAMSAI', 'PAPUM PARE', 'SIANG', 'TAWANG', 'TIROP', 'UPPER SIANG',
        'UPPER SUBANSIRI', 'WEST KAMENG', 'WEST SIANG'
    ]
    
    # Andaman and Nicobar Islands
    andaman_districts = [
        'N & M ANDAMAN', 'NICOBAR', 'SOUTH ANDAMAN', 'NORTH AND MIDDLE ANDAMAN'
    ]
    
    # Check mappings
    if any(ap_dist in district for ap_dist in andhra_districts):
        return "ANDHRA PRADESH"
    elif any(ar_dist in district for ar_dist in arunachal_districts):
        return "ARUNACHAL PRADESH"
    elif any(an_dist in district for an_dist in andaman_districts):
        return "ANDAMAN AND NICOBAR ISLANDS"
    else:
        return "Unknown"

def fix_ingris_states():
    """Fix state mapping in INGRIS data."""
    print("🔧 Fixing INGRIS State Mapping")
    print("=" * 50)
    
    try:
        # Read the CSV
        print("📄 Reading ingris_rag_ready.csv...")
        df = pd.read_csv('ingris_rag_ready.csv', low_memory=False)
        print(f"📊 Found {len(df)} records")
        
        # Show current state distribution
        print("\n📋 Current state distribution:")
        state_counts = df['state'].value_counts()
        print(state_counts)
        
        # Fix state mapping
        print("\n🔄 Fixing state mapping based on districts...")
        df['state'] = df['district'].apply(map_district_to_state)
        
        # Show new state distribution
        print("\n📋 New state distribution:")
        new_state_counts = df['state'].value_counts()
        print(new_state_counts)
        
        # Show some examples
        print("\n📋 Sample mappings:")
        sample_df = df[['state', 'district']].drop_duplicates().head(10)
        for _, row in sample_df.iterrows():
            print(f"  {row['district']} → {row['state']}")
        
        # Save the fixed CSV
        print("\n💾 Saving fixed CSV...")
        df.to_csv('ingris_rag_ready_fixed.csv', index=False)
        print("✅ Saved as ingris_rag_ready_fixed.csv")
        
        # Also update the original file
        df.to_csv('ingris_rag_ready.csv', index=False)
        print("✅ Updated original ingris_rag_ready.csv")
        
        print(f"\n🎉 State mapping fixed! {len(new_state_counts)} states identified.")
        
    except Exception as e:
        print(f"❌ Error fixing states: {e}")

if __name__ == "__main__":
    fix_ingris_states()

#!/usr/bin/env python3
"""
Quick analysis of INGRIS collection - sample based
"""

from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from collections import Counter

load_dotenv()
QDRANT_URL = os.getenv('QDRANT_URL', 'http://35.245.15.233:6333')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', '')
COLLECTION_NAME = 'ingris_groundwater_collection'

def quick_analysis():
    """Quick analysis using sample data."""
    print("🔍 Quick Analysis of INGRIS Groundwater Collection")
    print("=" * 60)
    
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        
        # Get collection info
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"📊 Collection: {COLLECTION_NAME}")
        print(f"📊 Total Points: {collection_info.points_count:,}")
        print(f"📊 Vector Size: {collection_info.config.params.vectors.size}")
        print(f"📊 Distance Metric: {collection_info.config.params.vectors.distance}")
        print(f"📊 Status: {collection_info.status}")
        
        # Get sample data (larger sample)
        print(f"\n🔍 Analyzing sample data...")
        sample_size = 10000  # Analyze 10k records as sample
        
        points = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=sample_size,
            with_payload=True,
            with_vectors=False
        )[0]
        
        print(f"📊 Sample size: {len(points):,} records")
        
        # Analyze sample
        states = set()
        districts = set()
        state_district_pairs = set()
        years = set()
        
        for point in points:
            payload = point.payload
            state = payload.get('STATE', 'N/A')
            district = payload.get('DISTRICT', 'N/A')
            year = payload.get('Assessment_Year', 'N/A')
            
            states.add(state)
            districts.add(district)
            state_district_pairs.add((state, district))
            years.add(year)
        
        # Display results
        print(f"\n📈 SAMPLE DATA SUMMARY:")
        print(f"   Unique States in sample: {len(states)}")
        print(f"   Unique Districts in sample: {len(districts)}")
        print(f"   State-District Combinations: {len(state_district_pairs)}")
        print(f"   Assessment Years: {sorted(years)}")
        
        # Show all states found in sample
        print(f"\n🗺️  STATES IN SAMPLE ({len(states)}):")
        print("-" * 40)
        for i, state in enumerate(sorted(states), 1):
            print(f"{i:2d}. {state}")
        
        # Show districts per state
        print(f"\n🏘️  DISTRICTS BY STATE (Sample):")
        print("-" * 40)
        
        # Group districts by state
        districts_by_state = {}
        for state, district in state_district_pairs:
            if state not in districts_by_state:
                districts_by_state[state] = set()
            districts_by_state[state].add(district)
        
        # Sort states and show districts
        for state in sorted(districts_by_state.keys()):
            districts = sorted(districts_by_state[state])
            print(f"\n{state} ({len(districts)} districts in sample):")
            for i, district in enumerate(districts, 1):
                print(f"  {i:2d}. {district}")
        
        # Show some statistics
        print(f"\n📊 SAMPLE STATISTICS:")
        print(f"   Average districts per state: {len(districts) / len(states):.1f}")
        
        # Find states with most districts in sample
        state_district_counts = {state: len(districts) for state, districts in districts_by_state.items()}
        top_states = sorted(state_district_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\n🏆 TOP 10 STATES BY DISTRICT COUNT (Sample):")
        for i, (state, count) in enumerate(top_states, 1):
            print(f"  {i:2d}. {state}: {count} districts")
        
        # Show sample records
        print(f"\n📋 SAMPLE RECORDS:")
        print("-" * 40)
        for i, point in enumerate(points[:5], 1):
            payload = point.payload
            print(f"\nRecord {i}:")
            print(f"  State: {payload.get('STATE', 'N/A')}")
            print(f"  District: {payload.get('DISTRICT', 'N/A')}")
            print(f"  Year: {payload.get('Assessment_Year', 'N/A')}")
            print(f"  Village: {payload.get('village', 'N/A')}")
            print(f"  Block: {payload.get('block', 'N/A')}")
            print(f"  Watershed Category: {payload.get('watershed_category', 'N/A')}")
            print(f"  Serial Number: {payload.get('serial_number', 'N/A')}")
            
        # Estimate total districts (rough calculation)
        print(f"\n📊 ESTIMATED TOTALS (based on sample):")
        sample_ratio = len(points) / collection_info.points_count
        estimated_districts = len(districts) / sample_ratio
        print(f"   Estimated total districts: {int(estimated_districts)}")
        print(f"   Sample represents {sample_ratio:.1%} of total data")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_analysis()

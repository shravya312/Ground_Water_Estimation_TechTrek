#!/usr/bin/env python3
"""
Analyze districts and states in the INGRIS groundwater collection
"""

from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from collections import Counter

load_dotenv()
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', '')
COLLECTION_NAME = 'ingris_groundwater_collection'

def analyze_ingris_data():
    """Analyze the INGRIS collection to get states and districts."""
    print("🔍 Analyzing INGRIS Groundwater Collection")
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
        
        # Get all data to analyze states and districts
        print(f"\n🔍 Analyzing all data...")
        print("This may take a moment for 162,632 records...")
        
        # Scroll through all data
        all_states = set()
        all_districts = set()
        state_district_pairs = set()
        years = set()
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        offset = None
        total_processed = 0
        
        while True:
            # Scroll with offset
            if offset is None:
                points, next_offset = client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=batch_size,
                    with_payload=True,
                    with_vectors=False
                )
            else:
                points, next_offset = client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
            
            # Process this batch
            for point in points:
                payload = point.payload
                state = payload.get('STATE', 'N/A')
                district = payload.get('DISTRICT', 'N/A')
                year = payload.get('Assessment_Year', 'N/A')
                
                all_states.add(state)
                all_districts.add(district)
                state_district_pairs.add((state, district))
                years.add(year)
            
            total_processed += len(points)
            print(f"Processed {total_processed:,} records...", end='\r')
            
            # Check if we've processed all records
            if next_offset is None or len(points) < batch_size:
                break
                
            offset = next_offset
        
        print(f"\n✅ Analysis complete! Processed {total_processed:,} records")
        
        # Display results
        print(f"\n📈 DATA SUMMARY:")
        print(f"   Total States: {len(all_states)}")
        print(f"   Total Districts: {len(all_districts)}")
        print(f"   Total State-District Combinations: {len(state_district_pairs)}")
        print(f"   Assessment Years: {sorted(years)}")
        
        # Show all states
        print(f"\n🗺️  ALL STATES ({len(all_states)}):")
        print("-" * 40)
        for i, state in enumerate(sorted(all_states), 1):
            print(f"{i:2d}. {state}")
        
        # Show districts per state
        print(f"\n🏘️  DISTRICTS BY STATE:")
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
            print(f"\n{state} ({len(districts)} districts):")
            for i, district in enumerate(districts, 1):
                print(f"  {i:2d}. {district}")
        
        # Show some statistics
        print(f"\n📊 STATISTICS:")
        print(f"   Average districts per state: {len(all_districts) / len(all_states):.1f}")
        
        # Find states with most districts
        state_district_counts = {state: len(districts) for state, districts in districts_by_state.items()}
        top_states = sorted(state_district_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\n🏆 TOP 10 STATES BY DISTRICT COUNT:")
        for i, (state, count) in enumerate(top_states, 1):
            print(f"  {i:2d}. {state}: {count} districts")
        
        # Show sample data
        print(f"\n📋 SAMPLE RECORDS:")
        print("-" * 40)
        sample_points = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5,
            with_payload=True,
            with_vectors=False
        )[0]
        
        for i, point in enumerate(sample_points, 1):
            payload = point.payload
            print(f"\nRecord {i}:")
            print(f"  State: {payload.get('STATE', 'N/A')}")
            print(f"  District: {payload.get('DISTRICT', 'N/A')}")
            print(f"  Year: {payload.get('Assessment_Year', 'N/A')}")
            print(f"  Village: {payload.get('village', 'N/A')}")
            print(f"  Block: {payload.get('block', 'N/A')}")
            print(f"  Watershed Category: {payload.get('watershed_category', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    analyze_ingris_data()

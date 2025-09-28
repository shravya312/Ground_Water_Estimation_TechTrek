#!/usr/bin/env python3
"""
Check Qdrant collection and CSV data to understand the structure
"""

import os
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

load_dotenv()

def check_qdrant_data():
    """Check Qdrant collection data"""
    print("Checking Qdrant Collection Data")
    print("=" * 50)
    
    try:
        # Connect to Qdrant
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        
        # Get collection info
        try:
            collection_info = client.get_collection('ingris_groundwater_collection')
            print(f"Collection points: {collection_info.points_count}")
            print(f"Vector size: {collection_info.config.params.vectors.size}")
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return
        
        # Get sample data
        print("\n1. Sample data from Qdrant:")
        try:
            results = client.scroll(
                collection_name='ingris_groundwater_collection',
                limit=5,
                with_payload=True
            )
            
            for i, result in enumerate(results[0]):
                print(f"\nPoint {i+1}:")
                payload = result.payload
                print(f"  STATE: {payload.get('STATE', 'N/A')}")
                print(f"  DISTRICT: {payload.get('DISTRICT', 'N/A')}")
                print(f"  Assessment_Year: {payload.get('Assessment_Year', 'N/A')}")
                print(f"  serial_number: {payload.get('serial_number', 'N/A')}")
                
        except Exception as e:
            print(f"Error getting sample data: {e}")
        
        # Check Karnataka data specifically
        print("\n2. Checking Karnataka data in Qdrant:")
        try:
            karnataka_filter = Filter(
                must=[FieldCondition(key='STATE', match=MatchValue(value='KARNATAKA'))]
            )
            
            karnataka_results = client.scroll(
                collection_name='ingris_groundwater_collection',
                scroll_filter=karnataka_filter,
                limit=10,
                with_payload=True
            )
            
            karnataka_count = len(karnataka_results[0])
            print(f"Karnataka records found: {karnataka_count}")
            
            if karnataka_count > 0:
                print("Sample Karnataka records:")
                for i, result in enumerate(karnataka_results[0][:3]):
                    payload = result.payload
                    print(f"  {i+1}. {payload.get('STATE')} - {payload.get('DISTRICT')} - {payload.get('Assessment_Year')}")
            else:
                print("No Karnataka records found!")
                
        except Exception as e:
            print(f"Error checking Karnataka data: {e}")
        
        # Check all states
        print("\n3. Checking all states in Qdrant:")
        try:
            all_results = client.scroll(
                collection_name='ingris_groundwater_collection',
                limit=100,
                with_payload=True
            )
            
            states = set()
            for result in all_results[0]:
                state = result.payload.get('STATE', 'N/A')
                if state and state != 'N/A':
                    states.add(state)
            
            print(f"States found in Qdrant: {sorted(states)}")
            print(f"Total unique states: {len(states)}")
            
        except Exception as e:
            print(f"Error checking states: {e}")
            
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")

def check_csv_data():
    """Check CSV data structure"""
    print("\n\nChecking CSV Data")
    print("=" * 50)
    
    try:
        # Load CSV
        df = pd.read_csv('ingris_rag_ready_complete.csv', low_memory=False)
        print(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Check column names
        print(f"\nColumn names: {list(df.columns)}")
        
        # Check state data
        if 'state' in df.columns:
            print(f"\nStates in CSV:")
            states = df['state'].unique()
            print(f"Unique states: {len(states)}")
            print(f"States: {sorted(states)}")
            
            # Check Karnataka specifically
            karnataka_data = df[df['state'].str.upper() == 'KARNATAKA']
            print(f"\nKarnataka records in CSV: {len(karnataka_data)}")
            
            if len(karnataka_data) > 0:
                print("Sample Karnataka records:")
                sample = karnataka_data[['state', 'district', 'year']].head(5)
                for idx, row in sample.iterrows():
                    print(f"  {row['state']} - {row['district']} - {row['year']}")
        else:
            print("No 'state' column found in CSV")
        
        # Check year data
        if 'year' in df.columns:
            print(f"\nYear range: {df['year'].min()} - {df['year'].max()}")
        elif 'Assessment_Year' in df.columns:
            print(f"\nAssessment_Year range: {df['Assessment_Year'].min()} - {df['Assessment_Year'].max()}")
        
    except Exception as e:
        print(f"Error loading CSV: {e}")

def main():
    """Main function"""
    print("Checking Qdrant and CSV Data")
    print("=" * 60)
    
    check_qdrant_data()
    check_csv_data()
    
    print("\n" + "=" * 60)
    print("Data check complete!")

if __name__ == "__main__":
    main()

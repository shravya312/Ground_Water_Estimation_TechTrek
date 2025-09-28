#!/usr/bin/env python3
"""
Simple check for Chhattisgarh data in CSV and Qdrant
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_csv_data():
    """Check if Chhattisgarh exists in CSV file"""
    print("Checking CSV data...")
    
    try:
        # Load CSV data
        df = pd.read_csv("ingris_rag_ready_complete.csv", low_memory=False)
        print(f"CSV loaded: {len(df)} records")
        
        # Check for Chhattisgarh variations
        chhattisgarh_variations = [
            'CHHATTISGARH', 'chhattisgarh', 'CHATTISGARH', 'chattisgarh',
            'Chhattisgarh', 'Chattisgarh'
        ]
        
        found_states = set()
        for col in ['state', 'STATE']:
            if col in df.columns:
                unique_states = df[col].dropna().unique()
                found_states.update(unique_states)
        
        print(f"Found {len(found_states)} unique states in CSV")
        
        # Check for Chhattisgarh
        chhattisgarh_found = False
        for state in found_states:
            if any(var in str(state).upper() for var in ['CHHATTISGARH', 'CHATTISGARH']):
                print(f"FOUND Chhattisgarh in CSV: '{state}'")
                chhattisgarh_found = True
                
                # Show sample records
                chhattisgarh_data = df[df[col].str.contains('CHHATTISGARH|CHATTISGARH', case=False, na=False)]
                print(f"   Records found: {len(chhattisgarh_data)}")
                if len(chhattisgarh_data) > 0:
                    print(f"   Sample districts: {chhattisgarh_data['district'].dropna().unique()[:5].tolist()}")
                break
        
        if not chhattisgarh_found:
            print("Chhattisgarh NOT found in CSV")
            print("Available states sample:")
            for state in sorted(list(found_states))[:20]:
                print(f"  - {state}")
        
        return chhattisgarh_found
        
    except Exception as e:
        print(f"Error checking CSV: {e}")
        return False

def check_qdrant_states():
    """Check what states are available in Qdrant using scroll method"""
    print("\nChecking Qdrant states...")
    
    try:
        from qdrant_client import QdrantClient
        
        # Initialize Qdrant client
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY') if os.getenv('QDRANT_API_KEY') else None,
            timeout=30
        )
        print("Qdrant client connected")
        
        # Use scroll to get sample records
        try:
            scroll_result = client.scroll(
                collection_name="ingris_groundwater_collection",
                limit=100
            )
            
            records = scroll_result[0]  # Get the records
            print(f"Retrieved {len(records)} sample records from Qdrant")
            
            states = set()
            for record in records:
                state = record.payload.get('state', '')
                if state:
                    states.add(state)
            
            print(f"Found {len(states)} unique states in Qdrant sample")
            
            # Check for Chhattisgarh
            chhattisgarh_found = False
            for state in states:
                if 'CHHATTISGARH' in state.upper() or 'CHATTISGARH' in state.upper():
                    print(f"FOUND Chhattisgarh in Qdrant: '{state}'")
                    chhattisgarh_found = True
                    break
            
            if not chhattisgarh_found:
                print("Chhattisgarh NOT found in Qdrant sample")
                print("Available states from Qdrant:")
                for state in sorted(list(states))[:20]:
                    print(f"  - {state}")
            
            return chhattisgarh_found
            
        except Exception as e:
            print(f"Error scrolling Qdrant: {e}")
            return False
        
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return False

def main():
    """Main function"""
    print("Checking Chhattisgarh data availability")
    print("=" * 50)
    
    csv_found = check_csv_data()
    qdrant_found = check_qdrant_states()
    
    print(f"\nSummary:")
    print(f"  CSV data: {'FOUND' if csv_found else 'NOT FOUND'}")
    print(f"  Qdrant data: {'FOUND' if qdrant_found else 'NOT FOUND'}")
    
    if csv_found and qdrant_found:
        print("SUCCESS: Chhattisgarh data is available in both sources!")
    elif csv_found:
        print("WARNING: Chhattisgarh data only in CSV, not in Qdrant")
    elif qdrant_found:
        print("WARNING: Chhattisgarh data only in Qdrant, not in CSV")
    else:
        print("ERROR: Chhattisgarh data not found in either source")

if __name__ == "__main__":
    main()

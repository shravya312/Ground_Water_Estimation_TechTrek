#!/usr/bin/env python3
"""
Check Chhattisgarh data in current Qdrant collection
"""

import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "groundwater_excel_collection"

def check_chhattisgarh_data():
    """Check Chhattisgarh data in current collection."""
    print("🔍 Checking Chhattisgarh Data in Current Collection")
    print("=" * 60)
    
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        timeout=60
    )
    
    try:
        # Get collection info
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"📊 Collection: {COLLECTION_NAME}")
        print(f"   Total Points: {collection_info.points_count:,}")
        
        # Search for Chhattisgarh data
        print(f"\n🔍 Searching for Chhattisgarh data...")
        
        # Try different variations of Chhattisgarh
        chhattisgarh_variations = [
            "CHHATTISGARH",
            "CHATTISGARH", 
            "CHHATISGARH",
            "CHHATTISGARH",
            "CHHATISGARH"
        ]
        
        chhattisgarh_records = []
        
        for variation in chhattisgarh_variations:
            try:
                # Search by state
                results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=[0.0] * 768,  # Dummy vector for filtering
                    query_filter={
                        "must": [
                            {"key": "STATE", "match": {"value": variation}}
                        ]
                    },
                    limit=100,
                    with_payload=True,
                    with_vectors=False
                )
                
                if results:
                    print(f"   ✅ Found {len(results)} records for '{variation}'")
                    chhattisgarh_records.extend(results)
                else:
                    print(f"   ❌ No records found for '{variation}'")
                    
            except Exception as e:
                print(f"   ⚠️ Error searching for '{variation}': {e}")
        
        # Remove duplicates
        unique_records = {}
        for record in chhattisgarh_records:
            record_id = record.id
            if record_id not in unique_records:
                unique_records[record_id] = record
        
        chhattisgarh_records = list(unique_records.values())
        
        print(f"\n📊 Chhattisgarh Data Summary:")
        print(f"   Total unique records: {len(chhattisgarh_records)}")
        
        if chhattisgarh_records:
            print(f"\n📋 Sample Chhattisgarh Records:")
            for i, record in enumerate(chhattisgarh_records[:3]):
                payload = record.payload
                print(f"\nRecord {i+1}:")
                print(f"   STATE: {payload.get('STATE', 'N/A')}")
                print(f"   DISTRICT: {payload.get('DISTRICT', 'N/A')}")
                print(f"   Assessment_Year: {payload.get('Assessment_Year', 'N/A')}")
                
                # Check key fields
                key_fields = [
                    'Rainfall (mm) - Total',
                    'Ground Water Recharge (ham) - Total',
                    'Ground Water Extraction for all uses (ha.m) - Total',
                    'Stage of Ground Water Extraction (%) - Total',
                    'Net Annual Ground Water Availability for Future Use (ham) - Total'
                ]
                
                print(f"   Key Data Fields:")
                for field in key_fields:
                    value = payload.get(field, 'N/A')
                    if value == 0.0 or value == '0.0' or value == '':
                        print(f"     {field}: {value} (NULL/EMPTY)")
                    else:
                        print(f"     {field}: {value}")
            
            # Check data quality
            print(f"\n🔍 Data Quality Analysis:")
            total_fields = len(chhattisgarh_records[0].payload) if chhattisgarh_records else 0
            null_count = 0
            
            for record in chhattisgarh_records:
                for key, value in record.payload.items():
                    if value is None or value == 0.0 or value == '0.0' or value == '':
                        null_count += 1
            
            if total_fields > 0:
                null_percentage = (null_count / (len(chhattisgarh_records) * total_fields)) * 100
                print(f"   Total fields checked: {len(chhattisgarh_records) * total_fields}")
                print(f"   Null/Empty fields: {null_count}")
                print(f"   Data completeness: {100 - null_percentage:.1f}%")
            
            # Get unique districts
            districts = set()
            years = set()
            for record in chhattisgarh_records:
                districts.add(record.payload.get('DISTRICT', 'Unknown'))
                years.add(record.payload.get('Assessment_Year', 'Unknown'))
            
            print(f"\n📊 Coverage:")
            print(f"   Districts: {len(districts)} - {list(districts)}")
            print(f"   Years: {len(years)} - {list(years)}")
            
        else:
            print(f"\n❌ No Chhattisgarh data found in current collection!")
            print(f"   This explains why the reports show incomplete data.")
            print(f"   Need to upload complete INGRIS dataset (162,632 records)")
        
        return len(chhattisgarh_records)
        
    except Exception as e:
        print(f"❌ Error checking collection: {e}")
        return 0

if __name__ == "__main__":
    records = check_chhattisgarh_data()
    
    if records == 0:
        print(f"\n🚨 ISSUE IDENTIFIED:")
        print(f"   Current collection has NO Chhattisgarh data!")
        print(f"   This is why reports show null/empty values.")
        print(f"\n💡 SOLUTION:")
        print(f"   Upload complete INGRIS dataset (162,632 records)")
        print(f"   which includes comprehensive Chhattisgarh data.")
    elif records < 10:
        print(f"\n⚠️ LIMITED DATA:")
        print(f"   Only {records} Chhattisgarh records found.")
        print(f"   Need complete dataset for comprehensive analysis.")
    else:
        print(f"\n✅ SUFFICIENT DATA:")
        print(f"   Found {records} Chhattisgarh records.")

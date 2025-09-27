#!/usr/bin/env python3
"""
Check for specific administrative data in the CSV file
"""

import pandas as pd

def check_specific_administrative_data():
    """Check for specific administrative data mentioned in the user's query"""
    print("🔍 Checking Specific Administrative Data")
    print("=" * 50)
    
    try:
        # Load the CSV file
        print("📂 Loading CSV file...")
        df = pd.read_csv('ingris_rag_ready_complete.csv', low_memory=False)
        print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Check for Hanuru taluk
        print("\n🔍 Searching for 'Hanuru' in taluk column...")
        hanuru_data = df[df['taluk'].str.contains('Hanuru', case=False, na=False)]
        print(f"Found {len(hanuru_data)} records with 'Hanuru'")
        
        if len(hanuru_data) > 0:
            print("\n📋 Hanuru Records:")
            admin_cols = ['STATE', 'DISTRICT', 'taluk', 'block', 'mandal', 'village', 'watershed_district', 'watershed_category']
            available_cols = [col for col in admin_cols if col in hanuru_data.columns]
            print(hanuru_data[available_cols].head())
        else:
            print("❌ No records found with 'Hanuru'")
        
        # Check for Bangalore-East
        print("\n🔍 Searching for 'Bangalore-East' in taluk column...")
        bangalore_data = df[df['taluk'].str.contains('Bangalore-East', case=False, na=False)]
        print(f"Found {len(bangalore_data)} records with 'Bangalore-East'")
        
        if len(bangalore_data) > 0:
            print("\n📋 Bangalore-East Records:")
            admin_cols = ['STATE', 'DISTRICT', 'taluk', 'block', 'mandal', 'village', 'watershed_district', 'watershed_category']
            available_cols = [col for col in admin_cols if col in bangalore_data.columns]
            print(bangalore_data[available_cols].head())
        else:
            print("❌ No records found with 'Bangalore-East'")
        
        # Check for Bangalore North
        print("\n🔍 Searching for 'Bangalore North' in taluk column...")
        bangalore_north_data = df[df['taluk'].str.contains('Bangalore North', case=False, na=False)]
        print(f"Found {len(bangalore_north_data)} records with 'Bangalore North'")
        
        if len(bangalore_north_data) > 0:
            print("\n📋 Bangalore North Records:")
            admin_cols = ['STATE', 'DISTRICT', 'taluk', 'block', 'mandal', 'village', 'watershed_district', 'watershed_category']
            available_cols = [col for col in admin_cols if col in bangalore_north_data.columns]
            print(bangalore_north_data[available_cols].head())
        else:
            print("❌ No records found with 'Bangalore North'")
        
        # Check watershed categories
        print("\n🔍 Checking watershed categories...")
        if 'watershed_category' in df.columns:
            watershed_cats = df['watershed_category'].value_counts()
            print("📊 Watershed Categories:")
            print(watershed_cats.head(10))
        else:
            print("❌ No watershed_category column found")
        
        # Check for any data in administrative columns
        print("\n🔍 Checking data availability in administrative columns...")
        admin_columns = ['watershed_district', 'tehsil', 'taluk', 'block', 'mandal', 'village', 'watershed_category']
        
        for col in admin_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                total_count = len(df)
                percentage = (non_null_count / total_count) * 100
                print(f"📊 {col}: {non_null_count}/{total_count} ({percentage:.1f}%) records have data")
                
                # Show sample values
                sample_values = df[col].dropna().unique()[:3]
                print(f"   Sample values: {list(sample_values)}")
            else:
                print(f"❌ {col}: Column not found")
        
        # Check for Karnataka data specifically
        print("\n🔍 Checking Karnataka administrative data...")
        karnataka_data = df[df['STATE'].str.contains('KARNATAKA', case=False, na=False)]
        print(f"Found {len(karnataka_data)} Karnataka records")
        
        if len(karnataka_data) > 0:
            print("\n📋 Karnataka Administrative Data Sample:")
            admin_cols = ['STATE', 'DISTRICT', 'taluk', 'block', 'mandal', 'village', 'watershed_district', 'watershed_category']
            available_cols = [col for col in admin_cols if col in karnataka_data.columns]
            print(karnataka_data[available_cols].head(10))
            
            # Check unique values in each administrative column for Karnataka
            print("\n📊 Unique values in Karnataka administrative columns:")
            for col in available_cols:
                if col != 'STATE':  # Skip STATE since we're already filtering by it
                    unique_vals = karnataka_data[col].dropna().unique()
                    print(f"  {col}: {len(unique_vals)} unique values")
                    if len(unique_vals) <= 10:
                        print(f"    Values: {list(unique_vals)}")
                    else:
                        print(f"    Sample: {list(unique_vals[:5])}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🧪 Specific Administrative Data Checker")
    print("=" * 40)
    
    check_specific_administrative_data()
    
    print("\n💡 Summary:")
    print("This script checks for specific administrative data mentioned in your query:")
    print("- Hanuru taluk")
    print("- Bangalore-East")
    print("- Bangalore North")
    print("- Watershed categories")
    print("- General administrative data availability")

if __name__ == "__main__":
    main()

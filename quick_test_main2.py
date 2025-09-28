#!/usr/bin/env python3
"""
Quick test for main2.py to check steps and response
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """Quick test to see what's happening"""
    print("Quick Test - main2.py")
    print("=" * 40)
    
    try:
        # Import main2
        print("1. Importing main2...")
        from main2 import answer_query
        print("   [SUCCESS] Imported")
        
        # Test with a simple query
        query = "karnataka groundwater"
        print(f"\n2. Testing: '{query}'")
        
        # Call the function
        result = answer_query(query)
        
        print(f"\n3. Response received:")
        print(f"   Length: {len(result)} characters")
        print(f"   Contains 'KARNATAKA': {'KARNATAKA' in result.upper()}")
        
        # Show first part of response
        print(f"\n4. First 1000 characters:")
        print("-" * 50)
        print(result[:1000])
        print("-" * 50)
        
        # Check for specific issues
        print(f"\n5. Analysis:")
        print(f"   'No data available' count: {result.count('No data available')}")
        print(f"   'Karnataka' mentions: {result.upper().count('KARNATAKA')}")
        print(f"   'Maharashtra' mentions: {result.upper().count('MAHARASHTRA')}")
        print(f"   'Telangana' mentions: {result.upper().count('TELANGANA')}")
        
        # Show last part
        print(f"\n6. Last 500 characters:")
        print("-" * 50)
        print(result[-500:])
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    quick_test()

#!/usr/bin/env python3
"""
Quick test with new Gemini API key
"""

import os
import sys

# Set the new API key
os.environ['GEMINI_API_KEY'] = 'AIzaSyDcN4c67_aRGrnt8OzbmZG4SGK_EAie06g'

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """Quick test of answer generation"""
    print("Quick Test with New Gemini API Key")
    print("=" * 40)
    
    try:
        from main2 import answer_query
        
        # Test Chhattisgarh
        print("Testing Chhattisgarh...")
        query = "groundwater estimation in chhattisgarh"
        
        answer = answer_query(query, "en", "test_user")
        
        print(f"Answer length: {len(answer)} characters")
        
        if "error" in answer.lower() and "quota" in answer.lower():
            print("FAILED: Still getting quota error")
            print(f"Error: {answer[:200]}...")
        elif len(answer) > 1000:
            print("SUCCESS: Answer generation working!")
            print(f"Preview: {answer[:200]}...")
        else:
            print("ISSUE: Answer too short or other problem")
            print(f"Answer: {answer}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    quick_test()

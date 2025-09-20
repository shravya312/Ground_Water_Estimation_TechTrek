#!/usr/bin/env python3
"""
INGRES ChatBOT Demo Script
Demonstrates the key features and capabilities of the INGRES ChatBOT
"""

import requests
import json
import time

def demo_ingres_chatbot():
    """Demonstrate INGRES ChatBOT capabilities"""
    
    base_url = "http://localhost:8000"
    
    print("🤖 INGRES CHATBOT DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the key features of the INGRES ChatBOT")
    print("for groundwater data analysis and management.")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend server is not running!")
            print("\n📋 To start the server:")
            print("1. Open a terminal in the project directory")
            print("2. Run: uvicorn main:app --reload --port 8000")
            print("3. Then run this demo script again")
            return
    except:
        print("❌ Cannot connect to backend server!")
        print("\n📋 To start the server:")
        print("1. Open a terminal in the project directory")
        print("2. Run: uvicorn main:app --reload --port 8000")
        print("3. Then run this demo script again")
        return
    
    print("✅ Backend server is running")
    print()
    
    # Demo 1: National Overview
    print("1️⃣ NATIONAL GROUNDWATER OVERVIEW")
    print("-" * 40)
    try:
        response = requests.get(f"{base_url}/ingres/criticality-summary", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Total Districts Analyzed: {data['total_districts']:,}")
            print(f"📈 National Average Extraction: {data['national_average_extraction']:.1f}%")
            print()
            print("🏛️ Criticality Distribution:")
            for status, info in data['criticality_distribution'].items():
                emoji = {"safe": "🟢", "semi_critical": "🟡", "critical": "🔴", "over_exploited": "⚫"}[status]
                print(f"   {emoji} {status.replace('_', '-').title()}: {info['count']:,} districts ({info['percentage']:.1f}%)")
        else:
            print("❌ Failed to get national overview")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Demo 2: Critical States Analysis
    print("2️⃣ CRITICAL STATES ANALYSIS")
    print("-" * 40)
    critical_states = ["PUNJAB", "RAJASTHAN", "HARYANA", "DELHI"]
    
    for state in critical_states:
        try:
            query_data = {
                "query": f"What is the groundwater status in {state}?",
                "state": state,
                "include_visualizations": False,
                "language": "en"
            }
            
            response = requests.post(f"{base_url}/ingres/query", json=query_data, timeout=15)
            if response.status_code == 200:
                data = response.json()
                print(f"🏛️ {state}:")
                print(f"   Status: {data['criticality_emoji']} {data['criticality_status']}")
                print(f"   Extraction: {data['numerical_values']['extraction_stage']:.1f}%")
                print(f"   Recharge: {data['numerical_values']['annual_recharge']:,.0f} ham")
                print(f"   Future Availability: {data['numerical_values']['future_availability']:,.0f} ham")
                print(f"   Quality Issues: {len(data['quality_issues'])} detected")
                print()
            else:
                print(f"   ❌ {state}: Query failed")
        except Exception as e:
            print(f"   ❌ {state}: Error - {e}")
    
    # Demo 3: Safe State Analysis
    print("3️⃣ SAFE STATE ANALYSIS")
    print("-" * 40)
    safe_states = ["CHHATTISGARH", "ARUNACHAL PRADESH", "MEGHALAYA"]
    
    for state in safe_states:
        try:
            query_data = {
                "query": f"What is the groundwater status in {state}?",
                "state": state,
                "include_visualizations": False,
                "language": "en"
            }
            
            response = requests.post(f"{base_url}/ingres/query", json=query_data, timeout=15)
            if response.status_code == 200:
                data = response.json()
                print(f"🏛️ {state}:")
                print(f"   Status: {data['criticality_emoji']} {data['criticality_status']}")
                print(f"   Extraction: {data['numerical_values']['extraction_stage']:.1f}%")
                print(f"   Rainfall: {data['numerical_values']['rainfall']:.0f} mm")
                print(f"   Area: {data['numerical_values']['total_area']:,.0f} ha")
                print()
            else:
                print(f"   ❌ {state}: Query failed")
        except Exception as e:
            print(f"   ❌ {state}: Error - {e}")
    
    # Demo 4: Location-based Analysis
    print("4️⃣ LOCATION-BASED ANALYSIS")
    print("-" * 40)
    
    locations = [
        {"name": "Mumbai", "lat": 19.0760, "lng": 72.8777},
        {"name": "Delhi", "lat": 28.7041, "lng": 77.1025},
        {"name": "Bangalore", "lat": 12.9716, "lng": 77.5946},
        {"name": "Chennai", "lat": 13.0827, "lng": 80.2707}
    ]
    
    for location in locations:
        try:
            location_data = {
                "lat": location["lat"],
                "lng": location["lng"],
                "include_visualizations": False,
                "language": "en"
            }
            
            response = requests.post(f"{base_url}/ingres/location-analysis", json=location_data, timeout=15)
            if response.status_code == 200:
                data = response.json()
                print(f"📍 {location['name']}:")
                print(f"   Detected State: {data['state']}")
                print(f"   Status: {data['criticality_emoji']} {data['criticality_status']}")
                print(f"   Extraction: {data['numerical_values']['extraction_stage']:.1f}%")
                print()
            else:
                print(f"   ❌ {location['name']}: Analysis failed")
        except Exception as e:
            print(f"   ❌ {location['name']}: Error - {e}")
    
    # Demo 5: Recommendations Demo
    print("5️⃣ RECOMMENDATIONS DEMO")
    print("-" * 40)
    
    try:
        # Query a critical state for recommendations
        query_data = {
            "query": "What recommendations do you have for Punjab's groundwater situation?",
            "state": "PUNJAB",
            "include_visualizations": False,
            "language": "en"
        }
        
        response = requests.post(f"{base_url}/ingres/query", json=query_data, timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"🏛️ {data['data']['state']} - {data['criticality_emoji']} {data['criticality_status']}")
            print(f"📊 Extraction Stage: {data['numerical_values']['extraction_stage']:.1f}%")
            print()
            print("🔧 Recommendations:")
            for i, rec in enumerate(data['recommendations'], 1):
                print(f"   {i}. {rec}")
            print()
        else:
            print("❌ Failed to get recommendations")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Demo 6: Water Quality Analysis
    print("6️⃣ WATER QUALITY ANALYSIS")
    print("-" * 40)
    
    quality_states = ["WEST BENGAL", "BIHAR", "ASSAM"]
    
    for state in quality_states:
        try:
            query_data = {
                "query": f"What are the water quality issues in {state}?",
                "state": state,
                "include_visualizations": False,
                "language": "en"
            }
            
            response = requests.post(f"{base_url}/ingres/query", json=query_data, timeout=15)
            if response.status_code == 200:
                data = response.json()
                print(f"🏛️ {state}:")
                print(f"   Status: {data['criticality_emoji']} {data['criticality_status']}")
                if data['quality_issues']:
                    print("   💧 Quality Issues:")
                    for issue in data['quality_issues']:
                        print(f"      • {issue}")
                else:
                    print("   💧 No major quality issues detected")
                print()
            else:
                print(f"   ❌ {state}: Query failed")
        except Exception as e:
            print(f"   ❌ {state}: Error - {e}")
    
    print("=" * 60)
    print("🎯 INGRES CHATBOT DEMO COMPLETED")
    print("=" * 60)
    print()
    print("📋 Key Features Demonstrated:")
    print("   ✅ National groundwater overview")
    print("   ✅ Critical states analysis")
    print("   ✅ Safe states analysis")
    print("   ✅ Location-based analysis")
    print("   ✅ Intelligent recommendations")
    print("   ✅ Water quality assessment")
    print()
    print("🚀 The INGRES ChatBOT provides comprehensive groundwater")
    print("   analysis with criticality assessment and actionable")
    print("   recommendations for sustainable water management.")

if __name__ == "__main__":
    demo_ingres_chatbot()

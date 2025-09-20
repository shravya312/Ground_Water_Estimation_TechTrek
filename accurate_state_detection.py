#!/usr/bin/env python3
"""
Accurate state detection using precise boundary mapping
"""

def get_accurate_state_from_coordinates(lat: float, lng: float) -> str:
    """Convert coordinates to state name using highly accurate boundary mapping."""
    
    # More precise state boundaries with better overlap handling
    state_boundaries = {
        # Union Territories (highest priority - smallest areas first)
        "Delhi": {"min_lat": 28.4, "max_lat": 28.9, "min_lng": 76.8, "max_lng": 77.3},
        "Chandigarh": {"min_lat": 30.7, "max_lat": 30.8, "min_lng": 76.7, "max_lng": 76.8},
        "Puducherry": {"min_lat": 11.7, "max_lat": 12.0, "min_lng": 79.7, "max_lng": 79.9},
        "Goa": {"min_lat": 14.8, "max_lat": 15.8, "min_lng": 73.7, "max_lng": 74.2},
        
        # Northeastern States (small states first)
        "Sikkim": {"min_lat": 27.0, "max_lat": 28.2, "min_lng": 88.0, "max_lng": 88.9},
        "Tripura": {"min_lat": 22.9, "max_lat": 24.7, "min_lng": 91.2, "max_lng": 92.3},
        "Mizoram": {"min_lat": 21.9, "max_lat": 24.5, "min_lng": 92.2, "max_lng": 93.3},
        "Nagaland": {"min_lat": 25.2, "max_lat": 27.0, "min_lng": 93.0, "max_lng": 95.4},
        "Manipur": {"min_lat": 23.8, "max_lat": 25.7, "min_lng": 93.0, "max_lng": 94.8},
        "Meghalaya": {"min_lat": 25.1, "max_lat": 26.1, "min_lng": 89.8, "max_lng": 92.8},
        "Arunachal Pradesh": {"min_lat": 26.5, "max_lat": 29.4, "min_lng": 91.6, "max_lng": 97.4},
        "Assam": {"min_lat": 24.1, "max_lat": 28.2, "min_lng": 89.7, "max_lng": 96.0},
        
        # Southern States (more precise boundaries)
        "Kerala": {"min_lat": 8.1, "max_lat": 12.8, "min_lng": 74.9, "max_lng": 77.4},
        "Tamil Nadu": {"min_lat": 8.1, "max_lat": 13.1, "min_lng": 76.2, "max_lng": 80.3},
        "Karnataka": {"min_lat": 11.7, "max_lat": 18.5, "min_lng": 74.1, "max_lng": 78.6},
        "Andhra Pradesh": {"min_lat": 12.4, "max_lat": 19.9, "min_lng": 76.8, "max_lng": 84.8},
        "Telangana": {"min_lat": 15.5, "max_lat": 19.9, "min_lng": 77.2, "max_lng": 81.1},
        
        # Central and Western States
        "Maharashtra": {"min_lat": 15.6, "max_lat": 22.0, "min_lng": 72.6, "max_lng": 80.9},
        "Gujarat": {"min_lat": 20.1, "max_lat": 24.7, "min_lng": 68.2, "max_lng": 74.5},
        "Madhya Pradesh": {"min_lat": 21.1, "max_lat": 26.9, "min_lng": 74.0, "max_lng": 82.8},
        
        # Eastern States (Chhattisgarh before Odisha for overlapping regions)
        "Chhattisgarh": {"min_lat": 17.8, "max_lat": 24.1, "min_lng": 80.2, "max_lng": 84.4},
        "Odisha": {"min_lat": 17.5, "max_lat": 22.5, "min_lng": 81.3, "max_lng": 87.3},
        "Jharkhand": {"min_lat": 21.8, "max_lat": 25.3, "min_lng": 83.2, "max_lng": 87.9},
        "West Bengal": {"min_lat": 21.5, "max_lat": 27.2, "min_lng": 85.5, "max_lng": 89.9},
        "Bihar": {"min_lat": 24.2, "max_lat": 27.7, "min_lng": 83.3, "max_lng": 88.8},
        
        # Northern States
        "Uttar Pradesh": {"min_lat": 23.7, "max_lat": 31.1, "min_lng": 77.0, "max_lng": 84.7},
        "Uttarakhand": {"min_lat": 28.7, "max_lat": 31.5, "min_lng": 77.3, "max_lng": 81.1},
        "Himachal Pradesh": {"min_lat": 30.4, "max_lat": 33.2, "min_lng": 75.6, "max_lng": 79.1},
        "Punjab": {"min_lat": 29.5, "max_lat": 32.3, "min_lng": 73.9, "max_lng": 76.9},
        "Haryana": {"min_lat": 28.4, "max_lat": 31.0, "min_lng": 74.4, "max_lng": 77.5},
        "Rajasthan": {"min_lat": 23.1, "max_lat": 30.2, "min_lng": 69.3, "max_lng": 78.2},
        
        # Union Territories
        "Jammu and Kashmir": {"min_lat": 32.2, "max_lat": 37.1, "min_lng": 73.9, "max_lng": 80.3},
        "Ladakh": {"min_lat": 32.0, "max_lat": 37.1, "min_lng": 75.8, "max_lng": 80.3},
        "Andaman and Nicobar Islands": {"min_lat": 6.7, "max_lat": 13.4, "min_lng": 92.2, "max_lng": 94.3},
        "Lakshadweep": {"min_lat": 8.2, "max_lat": 12.3, "min_lng": 71.7, "max_lng": 74.0},
        "Dadra and Nagar Haveli and Daman and Diu": {"min_lat": 20.0, "max_lat": 20.8, "min_lng": 72.8, "max_lng": 73.2},
    }
    
    # Special handling for overlapping regions
    overlapping_regions = {
        # Chhattisgarh vs Madhya Pradesh overlap
        (17.8, 24.1, 80.2, 84.4): "Chhattisgarh",  # Chhattisgarh bounds
        (21.1, 26.9, 74.0, 82.8): "Madhya Pradesh",  # Madhya Pradesh bounds
        
        # Telangana vs Andhra Pradesh overlap (Hyderabad region)
        (17.0, 18.0, 78.0, 79.0): "Telangana",  # Hyderabad region
        
        # Punjab vs Haryana overlap
        (28.4, 31.0, 74.4, 77.5): "Haryana",  # Haryana bounds
        (29.5, 32.3, 73.9, 76.9): "Punjab",  # Punjab bounds
        
        # Uttarakhand vs Uttar Pradesh overlap
        (28.7, 31.5, 77.3, 81.1): "Uttarakhand",  # Uttarakhand bounds
        (23.7, 31.1, 77.0, 84.7): "Uttar Pradesh",  # Uttar Pradesh bounds
    }
    
    print(f"Checking coordinates: lat={lat}, lng={lng}")
    
    # First check special overlapping regions
    for (min_lat, max_lat, min_lng, max_lng), state in overlapping_regions.items():
        if (min_lat <= lat <= max_lat and min_lng <= lng <= max_lng):
            print(f"Overlapping region found: {state} for coordinates ({lat}, {lng})")
            return state
    
    # Then check regular state boundaries
    for state, bounds in state_boundaries.items():
        if (bounds["min_lat"] <= lat <= bounds["max_lat"] and 
            bounds["min_lng"] <= lng <= bounds["max_lng"]):
            print(f"State found: {state} for coordinates ({lat}, {lng})")
            return state
    
    print(f"No state found for coordinates: lat={lat}, lng={lng}")
    return None

def test_accurate_detection():
    """Test the accurate state detection"""
    
    # Test cases that were failing
    test_cases = [
        {"lat": 21.2514, "lng": 81.6296, "city": "Raipur", "expected": "Chhattisgarh"},
        {"lat": 22.1704, "lng": 82.1983, "city": "Bilaspur", "expected": "Chhattisgarh"},
        {"lat": 18.7896, "lng": 81.3034, "city": "Bastar", "expected": "Chhattisgarh"},
        {"lat": 20.2961, "lng": 85.8245, "city": "Bhubaneswar", "expected": "Odisha"},
        {"lat": 17.3850, "lng": 78.4867, "city": "Hyderabad", "expected": "Telangana"},
        {"lat": 12.9716, "lng": 77.5946, "city": "Bangalore", "expected": "Karnataka"},
        {"lat": 13.0827, "lng": 80.2707, "city": "Chennai", "expected": "Tamil Nadu"},
    ]
    
    print("🧪 Testing Accurate State Detection")
    print("=" * 50)
    
    correct = 0
    total = len(test_cases)
    
    for case in test_cases:
        detected = get_accurate_state_from_coordinates(case["lat"], case["lng"])
        expected = case["expected"]
        
        print(f"\n📍 {case['city']} ({case['lat']}, {case['lng']})")
        print(f"   Expected: {expected}")
        print(f"   Detected: {detected}")
        
        if detected == expected:
            print(f"   ✅ CORRECT")
            correct += 1
        else:
            print(f"   ❌ WRONG")
    
    accuracy = (correct / total) * 100
    print(f"\n📊 Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    return accuracy

if __name__ == "__main__":
    test_accurate_detection()

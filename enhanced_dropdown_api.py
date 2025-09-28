#!/usr/bin/env python3
"""
Enhanced Dropdown API with Taluk Support
This module adds taluk-level support to the existing dropdown system.
"""

import pandas as pd
import json
from typing import Dict, List, Optional, Any
from fastapi import HTTPException

# Global variable to store the enhanced dataframe
_enhanced_df = None

def _load_enhanced_data():
    """Load the enhanced dataset with taluk data."""
    global _enhanced_df
    if _enhanced_df is None:
        try:
            _enhanced_df = pd.read_csv("ingris_rag_ready_complete.csv", low_memory=False)
            # Clean and standardize the data
            _enhanced_df['state'] = _enhanced_df['state'].fillna('').astype(str)
            _enhanced_df['district'] = _enhanced_df['district'].fillna('').astype(str)
            _enhanced_df['taluk'] = _enhanced_df['taluk'].fillna('').astype(str)
            _enhanced_df['tehsil'] = _enhanced_df['tehsil'].fillna('').astype(str)
            _enhanced_df['block'] = _enhanced_df['block'].fillna('').astype(str)
            _enhanced_df['mandal'] = _enhanced_df['mandal'].fillna('').astype(str)
            _enhanced_df['village'] = _enhanced_df['village'].fillna('').astype(str)
            print(f"[SUCCESS] Loaded enhanced dataset with {len(_enhanced_df)} records")
        except FileNotFoundError:
            raise Exception("Error: ingris_rag_ready_complete.csv not found.")
        except Exception as e:
            raise Exception(f"Error loading enhanced dataset: {e}")

def get_all_states() -> Dict[str, Any]:
    """Get list of all available states."""
    try:
        _load_enhanced_data()
        states = _enhanced_df['state'].dropna().unique().tolist()
        states = [s for s in states if s and str(s).strip()]
        states.sort()
        
        return {
            "success": True,
            "states": states,
            "count": len(states)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_all_districts() -> Dict[str, Any]:
    """Get list of all available districts."""
    try:
        _load_enhanced_data()
        districts = _enhanced_df['district'].dropna().unique().tolist()
        districts = [d for d in districts if d and str(d).strip()]
        districts.sort()
        
        return {
            "success": True,
            "districts": districts,
            "count": len(districts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_all_taluks() -> Dict[str, Any]:
    """Get list of all available taluks."""
    try:
        _load_enhanced_data()
        taluks = _enhanced_df['taluk'].dropna().unique().tolist()
        taluks = [t for t in taluks if t and str(t).strip()]
        taluks.sort()
        
        return {
            "success": True,
            "taluks": taluks,
            "count": len(taluks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_districts_by_state(state: str) -> Dict[str, Any]:
    """Get districts for a specific state."""
    try:
        _load_enhanced_data()
        
        # Find districts for the given state
        state_data = _enhanced_df[_enhanced_df['state'].str.upper() == state.upper()]
        if state_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for state: {state}")
        
        districts = state_data['district'].dropna().unique().tolist()
        districts = [d for d in districts if d and str(d).strip()]
        districts.sort()
        
        return {
            "success": True,
            "state": state,
            "districts": districts,
            "count": len(districts)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_taluks_by_state(state: str) -> Dict[str, Any]:
    """Get taluks for a specific state."""
    try:
        _load_enhanced_data()
        
        # Find taluks for the given state
        state_data = _enhanced_df[_enhanced_df['state'].str.upper() == state.upper()]
        if state_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for state: {state}")
        
        taluks = state_data['taluk'].dropna().unique().tolist()
        taluks = [t for t in taluks if t and str(t).strip()]
        taluks.sort()
        
        return {
            "success": True,
            "state": state,
            "taluks": taluks,
            "count": len(taluks)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_taluks_by_district(state: str, district: str) -> Dict[str, Any]:
    """Get taluks for a specific district in a state."""
    try:
        _load_enhanced_data()
        
        # Find taluks for the given state and district
        district_data = _enhanced_df[
            (_enhanced_df['state'].str.upper() == state.upper()) & 
            (_enhanced_df['district'].str.upper() == district.upper())
        ]
        if district_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for district: {district} in state: {state}")
        
        taluks = district_data['taluk'].dropna().unique().tolist()
        taluks = [t for t in taluks if t and str(t).strip()]
        taluks.sort()
        
        return {
            "success": True,
            "state": state,
            "district": district,
            "taluks": taluks,
            "count": len(taluks)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_hierarchical_data() -> Dict[str, Any]:
    """Get hierarchical data (states with their districts and taluks)."""
    try:
        _load_enhanced_data()
        
        hierarchical_data = {}
        states = _enhanced_df['state'].dropna().unique().tolist()
        states = [s for s in states if s and str(s).strip()]
        states.sort()
        
        for state in states:
            state_data = _enhanced_df[_enhanced_df['state'] == state]
            
            # Get districts
            districts = state_data['district'].dropna().unique().tolist()
            districts = [d for d in districts if d and str(d).strip()]
            districts.sort()
            
            # Get taluks
            taluks = state_data['taluk'].dropna().unique().tolist()
            taluks = [t for t in taluks if t and str(t).strip()]
            taluks.sort()
            
            hierarchical_data[state] = {
                "districts": districts,
                "district_count": len(districts),
                "taluks": taluks,
                "taluk_count": len(taluks)
            }
        
        return {
            "success": True,
            "hierarchical": hierarchical_data,
            "total_states": len(states),
            "total_districts": sum(len(data["districts"]) for data in hierarchical_data.values()),
            "total_taluks": sum(len(data["taluks"]) for data in hierarchical_data.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_detailed_hierarchical_data() -> Dict[str, Any]:
    """Get detailed hierarchical data with district-taluk relationships."""
    try:
        _load_enhanced_data()
        
        hierarchical_data = {}
        states = _enhanced_df['state'].dropna().unique().tolist()
        states = [s for s in states if s and str(s).strip()]
        states.sort()
        
        for state in states:
            state_data = _enhanced_df[_enhanced_df['state'] == state]
            
            # Get districts
            districts = state_data['district'].dropna().unique().tolist()
            districts = [d for d in districts if d and str(d).strip()]
            districts.sort()
            
            district_data = {}
            for district in districts:
                district_records = state_data[state_data['district'] == district]
                taluks = district_records['taluk'].dropna().unique().tolist()
                taluks = [t for t in taluks if t and str(t).strip()]
                taluks.sort()
                
                district_data[district] = {
                    "taluks": taluks,
                    "taluk_count": len(taluks)
                }
            
            hierarchical_data[state] = {
                "districts": district_data,
                "district_count": len(districts)
            }
        
        return {
            "success": True,
            "hierarchical": hierarchical_data,
            "total_states": len(states),
            "total_districts": sum(len(data["districts"]) for data in hierarchical_data.values()),
            "total_taluks": sum(
                sum(district_data["taluk_count"] for district_data in state_data["districts"].values())
                for state_data in hierarchical_data.values()
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_taluk_data(state: str, district: str, taluk: str) -> Dict[str, Any]:
    """Get detailed data for a specific taluk."""
    try:
        _load_enhanced_data()
        
        # Find data for the specific taluk
        taluk_data = _enhanced_df[
            (_enhanced_df['state'].str.upper() == state.upper()) & 
            (_enhanced_df['district'].str.upper() == district.upper()) &
            (_enhanced_df['taluk'].str.upper() == taluk.upper())
        ]
        
        if taluk_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for taluk: {taluk} in district: {district}, state: {state}")
        
        # Get the first record (assuming all records for a taluk have similar data)
        record = taluk_data.iloc[0]
        
        return {
            "success": True,
            "state": state,
            "district": district,
            "taluk": taluk,
            "data": {
                "stage_of_ground_water_extraction": record.get('stage_of_ground_water_extraction_', 'N/A'),
                "categorization": record.get('categorization_of_assessment_unit', 'N/A'),
                "rainfall_mm": record.get('rainfall_mm', 'N/A'),
                "ground_water_recharge_ham": record.get('ground_water_recharge_ham', 'N/A'),
                "ground_water_extraction_ham": record.get('ground_water_extraction_for_all_uses_ham', 'N/A'),
                "pre_monsoon_trend": record.get('pre_monsoon_of_gw_trend', 'N/A'),
                "post_monsoon_trend": record.get('post_monsoon_of_gw_trend', 'N/A'),
                "quality_tagging": record.get('quality_tagging', 'N/A'),
                "assessment_unit": record.get('assessment_unit', 'N/A'),
                "year": record.get('year', 'N/A')
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_available_administrative_units() -> Dict[str, Any]:
    """Get information about available administrative units."""
    try:
        _load_enhanced_data()
        
        # Check which administrative units have data
        units_info = {}
        
        for unit in ['state', 'district', 'taluk', 'tehsil', 'block', 'mandal', 'village']:
            if unit in _enhanced_df.columns:
                non_empty = _enhanced_df[unit].dropna()
                non_empty = non_empty[non_empty.astype(str).str.strip() != '']
                units_info[unit] = {
                    "available": True,
                    "count": len(non_empty.unique()),
                    "sample_values": non_empty.unique()[:5].tolist()
                }
            else:
                units_info[unit] = {"available": False}
        
        return {
            "success": True,
            "administrative_units": units_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Test the functions
    print("Testing Enhanced Dropdown API...")
    
    try:
        # Test basic functions
        print("\n1. Testing get_all_states()...")
        states_result = get_all_states()
        print(f"Found {states_result['count']} states")
        
        print("\n2. Testing get_all_districts()...")
        districts_result = get_all_districts()
        print(f"Found {districts_result['count']} districts")
        
        print("\n3. Testing get_all_taluks()...")
        taluks_result = get_all_taluks()
        print(f"Found {taluks_result['count']} taluks")
        
        print("\n4. Testing Karnataka districts...")
        karnataka_districts = get_districts_by_state("KARNATAKA")
        print(f"Karnataka has {karnataka_districts['count']} districts")
        
        print("\n5. Testing Chikkamagaluru taluks...")
        chikkamagaluru_taluks = get_taluks_by_district("KARNATAKA", "Chikkamagaluru")
        print(f"Chikkamagaluru has {chikkamagaluru_taluks['count']} taluks:")
        for taluk in chikkamagaluru_taluks['taluks']:
            print(f"  - {taluk}")
        
        print("\n6. Testing taluk data for Ajjampura...")
        ajjampura_data = get_taluk_data("KARNATAKA", "Chikkamagaluru", "Ajjampura")
        print(f"Ajjampura data: {ajjampura_data['data']}")
        
        print("\n7. Testing administrative units availability...")
        units_info = get_available_administrative_units()
        for unit, info in units_info['administrative_units'].items():
            if info['available']:
                print(f"  {unit}: {info['count']} units available")
            else:
                print(f"  {unit}: Not available")
        
        print("\n[SUCCESS] All tests passed!")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

#!/usr/bin/env python3
"""
Location Synonyms and Fuzzy Matching for Groundwater System
"""

import re
import pandas as pd
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional

# Location synonyms mapping
LOCATION_SYNONYMS = {
    # State synonyms
    'states': {
        'tamilnadu': ['tamilnadu', 'tamil nadu', 'tamilnadu', 'tn', 'tamil'],
        'andhra pradesh': ['andhra pradesh', 'andhra', 'ap', 'andhra pradesh'],
        'karnataka': ['karnataka', 'karnatak', 'ka', 'karnataka'],
        'kerala': ['kerala', 'ker', 'kerala'],
        'maharashtra': ['maharashtra', 'maha', 'mh', 'maharashtra'],
        'gujarat': ['gujarat', 'guj', 'gujarat'],
        'rajasthan': ['rajasthan', 'raj', 'rajasthan'],
        'uttar pradesh': ['uttar pradesh', 'up', 'uttar pradesh'],
        'west bengal': ['west bengal', 'wb', 'west bengal', 'bengal'],
        'bihar': ['bihar', 'bihar'],
        'odisha': ['odisha', 'orissa', 'odisha'],
        'assam': ['assam', 'assam'],
        'punjab': ['punjab', 'punjab'],
        'haryana': ['haryana', 'haryana'],
        'himachal pradesh': ['himachal pradesh', 'hp', 'himachal'],
        'jammu and kashmir': ['jammu and kashmir', 'j&k', 'jammu kashmir', 'kashmir'],
        'uttarakhand': ['uttarakhand', 'uk', 'uttarakhand'],
        'chhattisgarh': ['chhattisgarh', 'chhattisgarh', 'cg'],
        'jharkhand': ['jharkhand', 'jharkhand', 'jh'],
        'madhya pradesh': ['madhya pradesh', 'mp', 'madhya pradesh'],
        'telangana': ['telangana', 'telangana', 'ts'],
        'goa': ['goa', 'goa'],
        'sikkim': ['sikkim', 'sikkim'],
        'arunachal pradesh': ['arunachal pradesh', 'arunachal', 'ap'],
        'nagaland': ['nagaland', 'nagaland'],
        'manipur': ['manipur', 'manipur'],
        'mizoram': ['mizoram', 'mizoram'],
        'tripura': ['tripura', 'tripura'],
        'meghalaya': ['meghalaya', 'meghalaya'],
        'delhi': ['delhi', 'new delhi', 'delhi'],
        'chandigarh': ['chandigarh', 'chandigarh'],
        'puducherry': ['puducherry', 'pondicherry', 'puducherry'],
        'andaman and nicobar': ['andaman and nicobar', 'andaman', 'nicobar'],
        'dadra and nagar haveli': ['dadra and nagar haveli', 'dadra nagar haveli'],
        'daman and diu': ['daman and diu', 'daman diu'],
        'lakshadweep': ['lakshadweep', 'lakshadweep'],
    },
    
    # District synonyms (focusing on common ones)
    'districts': {
        'THE NILGIRIS': ['nilgiris', 'nilgiri', 'ooty', 'ootacamund', 'udhagamandalam', 'the nilgiris'],
        'bangalore urban': ['bangalore', 'bengaluru', 'bangalore urban', 'bengaluru urban'],
        'bangalore rural': ['bangalore rural', 'bengaluru rural'],
        'mumbai': ['mumbai', 'bombay', 'greater mumbai'],
        'delhi': ['delhi', 'new delhi', 'delhi'],
        'hyderabad': ['hyderabad', 'hyderabad'],
        'chennai': ['chennai', 'madras', 'chennai'],
        'kolkata': ['kolkata', 'calcutta', 'kolkata'],
        'pune': ['pune', 'pune'],
        'ahmedabad': ['ahmedabad', 'ahmedabad'],
        'jaipur': ['jaipur', 'jaipur'],
        'lucknow': ['lucknow', 'lucknow'],
        'kanpur': ['kanpur', 'kanpur'],
        'nagpur': ['nagpur', 'nagpur'],
        'indore': ['indore', 'indore'],
        'bhopal': ['bhopal', 'bhopal'],
        'visakhapatnam': ['visakhapatnam', 'vizag', 'visakhapatnam'],
        'vijayawada': ['vijayawada', 'vijayawada'],
        'guntur': ['guntur', 'guntur'],
        'krishna': ['krishna', 'krishna'],
        'west godavari': ['west godavari', 'west godavari'],
        'east godavari': ['east godavari', 'east godavari'],
        'coimbatore': ['coimbatore', 'coimbatore'],
        'madurai': ['madurai', 'madurai'],
        'salem': ['salem', 'salem'],
        'tiruchirappalli': ['tiruchirappalli', 'trichy', 'tiruchirappalli'],
        'tirunelveli': ['tirunelveli', 'tirunelveli'],
        'vellore': ['vellore', 'vellore'],
        'erode': ['erode', 'erode'],
        'dindigul': ['dindigul', 'dindigul'],
        'thanjavur': ['thanjavur', 'tanjore', 'thanjavur'],
        'tiruppur': ['tiruppur', 'tiruppur'],
        'namakkal': ['namakkal', 'namakkal'],
        'karur': ['karur', 'karur'],
        'tiruvallur': ['tiruvallur', 'tiruvallur'],
        'kancheepuram': ['kancheepuram', 'kanchipuram', 'kancheepuram'],
        'cuddalore': ['cuddalore', 'cuddalore'],
        'villupuram': ['villupuram', 'villupuram'],
        'dharmapuri': ['dharmapuri', 'dharmapuri'],
        'krishnagiri': ['krishnagiri', 'krishnagiri'],
        'tiruvannamalai': ['tiruvannamalai', 'tiruvannamalai'],
        'perambalur': ['perambalur', 'perambalur'],
        'ariyalur': ['ariyalur', 'ariyalur'],
        'pudukkottai': ['pudukkottai', 'pudukkottai'],
        'sivagangai': ['sivagangai', 'sivagangai'],
        'ramanathapuram': ['ramanathapuram', 'ramnad', 'ramanathapuram'],
        'virudhunagar': ['virudhunagar', 'virudhunagar'],
        'thoothukudi': ['thoothukudi', 'tuticorin', 'thoothukudi'],
        'tirunelveli': ['tirunelveli', 'tirunelveli'],
        'kanniyakumari': ['kanniyakumari', 'kanyakumari', 'kanniyakumari'],
        'theni': ['theni', 'theni'],
        'theni': ['theni', 'theni'],
        'tenkasi': ['tenkasi', 'tenkasi'],
        'tirupathur': ['tirupathur', 'tirupathur'],
        'ranipet': ['ranipet', 'ranipet'],
        'chengalpattu': ['chengalpattu', 'chengalpattu'],
        'mayiladuthurai': ['mayiladuthurai', 'mayiladuthurai'],
        'kallakurichi': ['kallakurichi', 'kallakurichi'],
        'kallakurichchi': ['kallakurichchi', 'kallakurichchi'],
    }
}

def normalize_location_name(name: str) -> str:
    """Normalize location name for comparison"""
    if not name:
        return ""
    return re.sub(r'[^\w\s]', '', str(name).lower().strip())

def get_similarity_score(str1: str, str2: str) -> float:
    """Calculate similarity score between two strings"""
    return SequenceMatcher(None, normalize_location_name(str1), normalize_location_name(str2)).ratio()

def find_best_state_match(query: str, available_states: List[str], threshold: float = 0.6) -> Optional[str]:
    """Find the best matching state from available states"""
    query_normalized = normalize_location_name(query)
    best_match = None
    best_score = 0
    
    # First try exact synonym matching
    for canonical_state, synonyms in LOCATION_SYNONYMS['states'].items():
        for synonym in synonyms:
            if synonym.lower() in query_normalized:
                # Find the actual state name in available_states
                for state in available_states:
                    if normalize_location_name(state) == normalize_location_name(canonical_state):
                        return state
    
    # Then try fuzzy matching
    for state in available_states:
        if not state or pd.isna(state):
            continue
            
        # Direct similarity
        score = get_similarity_score(query, state)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = state
            
        # Check against synonyms
        for canonical_state, synonyms in LOCATION_SYNONYMS['states'].items():
            if normalize_location_name(state) == normalize_location_name(canonical_state):
                for synonym in synonyms:
                    if synonym.lower() in query_normalized:
                        return state
                    score = get_similarity_score(query, synonym)
                    if score > best_score and score >= threshold:
                        best_score = score
                        best_match = state
    
    return best_match

def find_best_district_match(query: str, available_districts: List[str], threshold: float = 0.6) -> Optional[str]:
    """Find the best matching district from available districts"""
    query_normalized = normalize_location_name(query)
    best_match = None
    best_score = 0
    
    # First try exact synonym matching
    for canonical_district, synonyms in LOCATION_SYNONYMS['districts'].items():
        for synonym in synonyms:
            if synonym.lower() in query_normalized:
                # Find the actual district name in available_districts
                for district in available_districts:
                    if normalize_location_name(district) == normalize_location_name(canonical_district):
                        return district
    
    # Then try fuzzy matching
    for district in available_districts:
        if not district or pd.isna(district):
            continue
            
        # Direct similarity
        score = get_similarity_score(query, district)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = district
            
        # Check against synonyms
        for canonical_district, synonyms in LOCATION_SYNONYMS['districts'].items():
            if normalize_location_name(district) == normalize_location_name(canonical_district):
                for synonym in synonyms:
                    if synonym.lower() in query_normalized:
                        return district
                    score = get_similarity_score(query, synonym)
                    if score > best_score and score >= threshold:
                        best_score = score
                        best_match = district
    
    return best_match

def extract_location_from_query(query: str, available_states: List[str], available_districts: List[str] = None, df: pd.DataFrame = None) -> Tuple[Optional[str], Optional[str]]:
    """Extract state and district from query using improved matching"""
    query_lower = query.lower()
    target_state = None
    target_district = None
    
    # Try to find state first
    target_state = find_best_state_match(query, available_states)
    
    # Try to find district
    if available_districts:
        target_district = find_best_district_match(query, available_districts)
    
    # If we found a district but no state, try to infer state from district
    if target_district and not target_state and df is not None:
        # Find which state this district belongs to
        district_states = df[df['DISTRICT'] == target_district]['STATE'].unique()
        if len(district_states) > 0:
            target_state = district_states[0]  # Take the first state (should be only one)
    
    # If we found a state but no district, and the query contains district-like terms, try harder
    if target_state and not target_district and available_districts:
        # Filter districts by state
        if df is not None:
            state_districts = df[df['STATE'] == target_state]['DISTRICT'].unique().tolist()
            target_district = find_best_district_match(query, state_districts)
        else:
            target_district = find_best_district_match(query, available_districts)
    
    return target_state, target_district

# pandas is already imported at the top

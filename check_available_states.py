#!/usr/bin/env python3
"""
Check what states are available in the data files
"""

import pandas as pd
import glob

def check_available_states():
    """Check what states are available in the data files."""
    print("🔍 Checking Available States in Data Files")
    print("=" * 50)
    
    files = glob.glob('datasets123/datasets/*.xlsx')
    states = []
    
    for i, file in enumerate(files):
        print(f"\n📄 File {i+1}: {file.split('/')[-1]}")
        try:
            df = pd.read_excel(file, header=None)
            
            # Look for state information in the first few rows
            for row_idx in range(min(5, len(df))):
                cell_value = str(df.iloc[row_idx, 0]) if len(df.columns) > 0 else ""
                
                if "for :" in cell_value and "for" in cell_value:
                    # Extract state name
                    parts = cell_value.split("for :")
                    if len(parts) > 1:
                        state_part = parts[1].split("for")[0].strip()
                        states.append(state_part)
                        print(f"   State found: {state_part}")
                        break
                elif "ANDAMAN" in cell_value or "NICOBAR" in cell_value:
                    states.append("ANDAMAN AND NICOBAR ISLANDS")
                    print(f"   State found: ANDAMAN AND NICOBAR ISLANDS")
                    break
                elif "KARNATAKA" in cell_value.upper():
                    states.append("KARNATAKA")
                    print(f"   State found: KARNATAKA")
                    break
                elif "MAHARASHTRA" in cell_value.upper():
                    states.append("MAHARASHTRA")
                    print(f"   State found: MAHARASHTRA")
                    break
                elif "TAMIL NADU" in cell_value.upper():
                    states.append("TAMIL NADU")
                    print(f"   State found: TAMIL NADU")
                    break
                elif "GUJARAT" in cell_value.upper():
                    states.append("GUJARAT")
                    print(f"   State found: GUJARAT")
                    break
                elif "RAJASTHAN" in cell_value.upper():
                    states.append("RAJASTHAN")
                    print(f"   State found: RAJASTHAN")
                    break
                    
        except Exception as e:
            print(f"   Error reading file: {e}")
    
    print(f"\n📊 Available States: {list(set(states))}")
    
    # Check if Karnataka is available
    karnataka_available = any("karnataka" in state.lower() for state in states)
    print(f"\n🔍 Karnataka Available: {karnataka_available}")
    
    if not karnataka_available:
        print("\n💡 Suggestion: Try querying for available states like:")
        for state in set(states):
            print(f"   - {state}")

if __name__ == "__main__":
    check_available_states()

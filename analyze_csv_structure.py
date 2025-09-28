#!/usr/bin/env python3
"""
Analyze CSV data structure
"""

import pandas as pd

def analyze_csv_structure():
    # Load the CSV data
    df = pd.read_csv('ingris_rag_ready_complete.csv', skiprows=1)

    # Clean column names to match the structure
    df.columns = [
        'serial_number', 'state', 'district', 'island', 'watershed_district',
        'rainfall_mm', 'total_geographical_area_ha', 'ground_water_recharge_ham',
        'inflows_and_outflows_ham', 'annual_ground_water_recharge_ham',
        'environmental_flows_ham', 'annual_extractable_ground_water_resource_ham',
        'ground_water_extraction_for_all_uses_ham', 'stage_of_ground_water_extraction_',
        'categorization_of_assessment_unit', 'pre_monsoon_of_gw_trend',
        'post_monsoon_of_gw_trend', 'allocation_of_ground_water_resource_for_domestic_utilisation_for_projected_year_2025_ham',
        'net_annual_ground_water_availability_for_future_use_ham', 'quality_tagging',
        'additional_potential_resources_under_specific_conditionsham', 'coastal_areas',
        'instorage_unconfined_ground_water_resourcesham', 'total_ground_water_availability_in_unconfined_aquifier_ham',
        'dynamic_confined_ground_water_resourcesham', 'instorage_confined_ground_water_resourcesham',
        'total_confined_ground_water_resources_ham', 'dynamic_semi_confined_ground_water_resources_ham',
        'instorage_semi_confined_ground_water_resources_ham', 'total_semiconfined_ground_water_resources_ham',
        'total_ground_water_availability_in_the_area_ham', 'source_file', 'year',
        'tehsil', 'taluk', 'block', 'valley', 'assessment_unit', 'mandal',
        'village', 'watershed_category', 'firka', 'combined_text'
    ]

    print('CSV Data Structure Analysis')
    print('=' * 50)
    print(f'Total records: {len(df)}')
    print(f'Total columns: {len(df.columns)}')

    print('\nColumn names:')
    for i, col in enumerate(df.columns):
        print(f'{i+1:2d}. {col}')

    print('\nState field analysis:')
    print(f'Unique states: {df["state"].nunique()}')
    print(f'State field type: {df["state"].dtype}')
    print(f'Null values in state: {df["state"].isnull().sum()}')

    print('\nSample state values:')
    state_counts = df['state'].value_counts()
    for state, count in state_counts.head(10).items():
        print(f'  {state}: {count} records')

    print('\nKarnataka data:')
    karnataka_data = df[df['state'].str.contains('KARNATAKA', case=False, na=False)]
    print(f'Karnataka records: {len(karnataka_data)}')

    if len(karnataka_data) > 0:
        print('Sample Karnataka record:')
        sample = karnataka_data.iloc[0]
        print(f'  State: {sample["state"]}')
        print(f'  District: {sample["district"]}')
        print(f'  Year: {sample["year"]}')
        print(f'  Taluk: {sample["taluk"]}')
        
        print('\nKarnataka districts:')
        karnataka_districts = karnataka_data['district'].unique()
        for district in sorted(karnataka_districts):
            count = len(karnataka_data[karnataka_data['district'] == district])
            print(f'  {district}: {count} records')

    # Check for data quality issues
    print('\nData Quality Analysis:')
    print(f'Records with null state: {df["state"].isnull().sum()}')
    print(f'Records with empty state: {(df["state"] == "").sum()}')
    print(f'Records with null district: {df["district"].isnull().sum()}')
    print(f'Records with null taluk: {df["taluk"].isnull().sum()}')

if __name__ == "__main__":
    analyze_csv_structure()

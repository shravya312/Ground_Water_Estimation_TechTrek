
import pandas as pd

try:
    df = pd.read_csv("cleaned_groundwater_data.csv")
except FileNotFoundError:
    print("Error: cleaned_groundwater_data.csv not found.")
    exit()

# Filter for Karnataka
karnataka_df = df[df['STATE'].fillna('').str.contains('KARNATAKA', case=False, na=False)].copy()

if karnataka_df.empty:
    print("No data found for Karnataka.")
else:
    # Calculate averages
    avg_fresh_availability = karnataka_df['Total Ground Water Availability in the area (ham) - Other Parameters Present - Fresh'].astype(float).mean()
    avg_saline_availability = karnataka_df['Total Ground Water Availability in the area (ham) - Other Parameters Present - Saline'].astype(float).mean()
    avg_rainfall = karnataka_df['Rainfall (mm) - Total'].astype(float).mean()

    print(f"Actual Averages for Karnataka:")
    print(f"  - Average Fresh Groundwater Availability: {avg_fresh_availability:.2f} ham")
    print(f"  - Average Saline Groundwater Availability: {avg_saline_availability:.2f} ham")
    print(f"  - Average Total Rainfall: {avg_rainfall:.2f} mm")

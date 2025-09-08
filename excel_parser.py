
import pandas as pd

file_path = "Ground_Water_Estimation_TechTrek/datasets123/datasets/CentralReport1757330502951.xlsx"

# Read the Excel file, including the rows that will form the header
full_df = pd.read_excel(file_path, header=None, skiprows=6) # Start from row 7 (index 6)

# Extract the rows that contain header information (original rows 7, 8, 9, 10)
header_rows = full_df.iloc[0:4] # These are now 0-indexed within full_df

# Transpose the header rows to fill NaN values easily
header_t = header_rows.T

# Forward-fill NaN values in the higher-level header rows
# Using .ffill() as fillna(method='ffill') is deprecated
header_t[0] = header_t[0].ffill()
header_t[1] = header_t[1].ffill()
header_t[2] = header_t[2].ffill()
header_t[3] = header_t[3].ffill()

# Combine the header rows to create new column names
def combine_headers(row):
    parts = [str(x).strip() for x in row if pd.notna(x)]
    if not parts: # Handle completely empty rows after stripping
        return 'Unnamed'
    # Special handling for numerical parts that appear to be sub-categories but are not
    cleaned_parts = []
    for part in parts:
        # Exclude parts that are just 'nan' or 'Unnamed' from intermediate levels if not the only part
        if part.lower() == 'nan' or part == 'Unnamed':
            continue
        cleaned_parts.append(part)
    
    if len(cleaned_parts) > 0:
        return ' - '.join(cleaned_parts)
    else:
        return 'Unnamed'

new_columns = header_t.apply(combine_headers, axis=1).tolist()

# Clean up column names further (remove 'Unnamed' if it's the only part or at the end)
cleaned_new_columns = []
for col in new_columns:
    if col == 'Unnamed':
        cleaned_new_columns.append(col)
    else:
        # Remove 'Unnamed' if it's part of a longer string at the beginning or end
        if col.startswith('Unnamed - '):
            col = col[len('Unnamed - '):]
        if col.endswith(' - Unnamed'):
            col = col[:-len(' - Unnamed')]
        cleaned_new_columns.append(col)

# Assign new columns to the data DataFrame, dropping the header rows
data_df = full_df.iloc[4:].copy() # Actual data starts from original row 11 (index 4 within full_df)
data_df.columns = cleaned_new_columns

# Drop the first row of data_df, which still contains NaN values from the original header
data_df = data_df.drop(data_df.index[0]).reset_index(drop=True)

# Save the cleaned DataFrame to a CSV file
data_df.to_csv("cleaned_groundwater_data.csv", index=False)

# Display the cleaned columns and first few rows
print(data_df.columns)
print(data_df.head())

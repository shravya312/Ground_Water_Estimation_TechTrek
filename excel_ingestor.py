
import pandas as pd
import re
import os

def extract_assessment_year(file_path):
    try:
        df_meta = pd.read_excel(file_path, header=None, nrows=5)
        for index, row in df_meta.iterrows():
            for cell_value in row.values:
                if isinstance(cell_value, str) and "Assessment Year" in cell_value:
                    match = re.search(r'(\d{4})', cell_value)
                    if match:
                        return int(match.group(1))
                elif isinstance(cell_value, str) and re.match(r'^\d{4}-\d{4}$', cell_value.strip()):
                    return int(cell_value.split('-')[0])
        return None
    except Exception as e:
        print(f"Error extracting year from {file_path}: {e}")
        return None

def parse_excel_file(file_path, year):
    try:
        full_df = pd.read_excel(file_path, header=None, skiprows=6)
        header_rows = full_df.iloc[0:4]
        header_t = header_rows.T

        header_t[0] = header_t[0].ffill()
        header_t[1] = header_t[1].ffill()
        header_t[2] = header_t[2].ffill()
        header_t[3] = header_t[3].ffill()

        def combine_headers(row):
            parts = [str(x).strip() for x in row if pd.notna(x)]
            if not parts:
                return 'Unnamed'
            cleaned_parts = []
            for part in parts:
                if part.lower() == 'nan' or part == 'Unnamed':
                    continue
                cleaned_parts.append(part)
            
            if len(cleaned_parts) > 0:
                return ' - '.join(cleaned_parts)
            else:
                return 'Unnamed'

        new_columns = header_t.apply(combine_headers, axis=1).tolist()

        cleaned_new_columns = []
        for col in new_columns:
            if col == 'Unnamed':
                cleaned_new_columns.append(col)
            else:
                if col.startswith('Unnamed - '):
                    col = col[len('Unnamed - '):]
                if col.endswith(' - Unnamed'):
                    col = col[:-len(' - Unnamed')]
                cleaned_new_columns.append(col)

        data_df = full_df.iloc[4:].copy()
        data_df.columns = cleaned_new_columns
        data_df = data_df.drop(data_df.index[0]).reset_index(drop=True)

        data_df['Assessment_Year'] = year # Add the extracted year column
        return data_df

    except Exception as e:
        print(f"Error parsing Excel file {file_path}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    datasets_dir = "datasets123/datasets/"
    all_excel_files = [os.path.join(datasets_dir, f) for f in os.listdir(datasets_dir) if f.endswith('.xlsx')]

    master_df_list = []
    for excel_file in all_excel_files:
        print(f"Processing {excel_file}...")
        year = extract_assessment_year(excel_file)
        if year:
            print(f"  Extracted year: {year}") # Added this line to print the extracted year
            parsed_df = parse_excel_file(excel_file, year)
            if not parsed_df.empty:
                master_df_list.append(parsed_df)
        else:
            print(f"Could not extract assessment year for {excel_file}. Skipping.")

    if master_df_list:
        master_groundwater_df = pd.concat(master_df_list, ignore_index=True)
        # Ensure 'ASSESSMENT UNIT' and 'DISTRICT' are treated as strings before combining
        master_groundwater_df['ASSESSMENT UNIT'] = master_groundwater_df['ASSESSMENT UNIT'].astype(str)
        master_groundwater_df['DISTRICT'] = master_groundwater_df['DISTRICT'].astype(str)
        master_groundwater_df.to_csv("master_groundwater_data.csv", index=False)
        print("Successfully created master_groundwater_data.csv")
        print(f"Master DataFrame columns: {master_groundwater_df.columns.tolist()}") # Added this line
        print(f"Master DataFrame head:\n{master_groundwater_df.head()}")
    else:
        print("No Excel files were processed successfully to create the master dataset.")

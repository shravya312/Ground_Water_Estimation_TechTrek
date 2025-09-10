import os
import sys
import pandas as pd

# Ensure we can import local modules when executed from repo root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import excel_ingestor as ei


def main() -> int:
    datasets_dir = os.path.join(BASE_DIR, "datasets123", "datasets")
    if not os.path.isdir(datasets_dir):
        print(f"ERROR: datasets directory not found: {datasets_dir}")
        return 2

    excel_files = [f for f in os.listdir(datasets_dir) if f.lower().endswith(".xlsx")]
    excel_files.sort()

    print("file,year_extracted,year,rows")
    all_ok = True
    total_rows = 0
    for fname in excel_files:
        fpath = os.path.join(datasets_dir, fname)
        try:
            year = ei.extract_assessment_year(fpath)
            if year is None:
                print(f"{fname},False,,0")
                all_ok = False
                continue

            df = ei.parse_excel_file(fpath, year)
            rows = len(df) if isinstance(df, pd.DataFrame) else 0
            print(f"{fname},True,{year},{rows}")
            if rows == 0:
                all_ok = False
            else:
                total_rows += rows
        except Exception as e:
            print(f"{fname},ERROR,,0")
            all_ok = False

    print(f"\nALL_EXTRACTED={all_ok}")
    print(f"TOTAL_ROWS={total_rows}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())



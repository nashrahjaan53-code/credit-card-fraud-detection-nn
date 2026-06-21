import pandas as pd
import sys
import os

def main():
    print("Running data validation...")
    
    csv_path = "data/creditcard.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        sys.exit(1)
        
    # Validate columns
    expected_columns = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
    columns_match = list(df.columns) == expected_columns
    
    # Validate Class column is binary
    class_values_valid = df["Class"].isin([0, 1]).all()
    
    # Validate Class has no missing values
    class_not_null = df["Class"].notna().all()
    
    # Validate no null values in feature columns
    no_nulls = df.drop(columns=["Class"]).notna().all().all()
    
    # Validate minimum row count
    min_rows = len(df) >= 100
    
    all_passed = all([columns_match, class_values_valid, class_not_null, no_nulls, min_rows])
    
    if all_passed:
        print("Data validation PASSED!")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Fraud rate: {df['Class'].mean()*100:.2f}%")
        sys.exit(0)
    else:
        print("Data validation FAILED!")
        print(f"  Column match: {columns_match}")
        print(f"  Class values in [0,1]: {class_values_valid}")
        print(f"  Class non-null: {class_not_null}")
        print(f"  No null features: {no_nulls}")
        print(f"  Min rows (>=100): {min_rows}")
        sys.exit(1)

if __name__ == "__main__":
    main()

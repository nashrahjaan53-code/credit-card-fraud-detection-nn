import pandas as pd
import great_expectations as ge
import sys
import os

def main():
    print("Running data validation using Great Expectations...")
    
    csv_path = "data/creditcard.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        sys.exit(1)
        
    try:
        df = ge.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        sys.exit(1)
        
    # Validate columns
    expected_columns = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
    result_cols = df.expect_table_columns_to_match_ordered_list(expected_columns)
    
    # Validate Class column is binary
    result_class = df.expect_column_values_to_be_in_set("Class", [0, 1])
    
    # Validate Class has no missing values
    result_null = df.expect_column_values_to_not_be_null("Class")
    
    if result_cols["success"] and result_class["success"] and result_null["success"]:
        print("Data validation PASSED!")
        sys.exit(0)
    else:
        print("Data validation FAILED!")
        print("Column match:", result_cols["success"])
        print("Class values in [0,1]:", result_class["success"])
        print("Class non-null:", result_null["success"])
        sys.exit(1)

if __name__ == "__main__":
    main()

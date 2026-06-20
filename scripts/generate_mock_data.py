import pandas as pd
import numpy as np
import os

def main():
    os.makedirs("data", exist_ok=True)
    
    # Generate mock creditcard.csv (5000 records)
    n_samples = 5000
    columns = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
    
    # Random normal features
    data = np.random.randn(n_samples, len(columns) - 1)
    
    # Highly imbalanced Class (25 fraud cases out of 5000)
    classes = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, size=25, replace=False)
    classes[fraud_indices] = 1
    
    df = pd.DataFrame(data, columns=columns[:-1])
    df["Class"] = classes.astype(int)
    
    # Save creditcard.csv
    df.to_csv("data/creditcard.csv", index=False)
    print("Generated mock data/creditcard.csv successfully!")
    
    # Generate mock reference.csv (2000 records)
    ref_df = df.sample(2000, random_state=42)
    ref_df.to_csv("data/reference.csv", index=False)
    print("Generated mock data/reference.csv successfully!")

if __name__ == "__main__":
    main()

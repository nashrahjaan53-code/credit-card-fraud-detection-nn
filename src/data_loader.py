import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import TensorDataset, DataLoader

def prepare_fraud_data(csv_path, batch_size=64):
    """
    Loads, splits, scales, and oversamples data correctly to prevent leakage.
    """
    # Load dataset (Assuming standard Kaggle Credit Card Fraud dataset)
    df = pd.read_csv(csv_path)
    
    # 'Time' is usually irrelevant, drop it. 'Amount' needs scaling.
    X = df.drop(columns=['Class', 'Time'])
    y = df['Class'].values
    
    # Step 1: Strict Train/Test Split (Never touch test data with SMOTE!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 2: Scale numerical features based ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 3: Apply SMOTE ONLY to the training set
    print(f"Original training class distribution: {np.bincount(y_train)}")
    smote = SMOTE(sampling_strategy=0.1, random_state=42) # Bring minority class up to 10% of majority
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"Resampled training class distribution: {np.bincount(y_train_resampled)}")
    
    # Step 4: Convert to PyTorch Tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train_resampled, dtype=torch.float32),
        torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    )
    
    # Step 5: Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, X_train.shape[1]
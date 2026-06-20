import torch
from src.model import FraudClassifier
from src.data_loader import prepare_fraud_data

def test_fraud_classifier_architecture():
    # Instantiate model with 29 input features (V1-V28 + Amount)
    input_dim = 29
    model = FraudClassifier(input_dim=input_dim)
    
    # Generate dummy input tensor
    batch_size = 16
    x = torch.randn(batch_size, input_dim)
    
    # Run forward pass
    out = model(x)
    
    # Assert output shape matches expectation (batch_size, 1)
    assert out.shape == (batch_size, 1)

def test_prepare_fraud_data_dimensions():
    # Verify utility functions work with mock paths or verify dimension
    # Let's mock a tiny csv to verify it returns expected structures
    import pandas as pd
    import tempfile
    import os
    
    # Create a small dummy CSV file representing the dataset structure
    columns = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
    data = [[0.1] * 28 + [0, 10.0, 0] for _ in range(200)]  # 200 samples
    data[0][-1] = 1  # Add at least one fraud sample so stratify/SMOTE can run
    data[1][-1] = 1
    data[2][-1] = 1
    data[3][-1] = 1
    data[4][-1] = 1
    data[5][-1] = 1
    data[6][-1] = 1
    data[7][-1] = 1
    data[8][-1] = 1
    data[9][-1] = 1
    data[10][-1] = 1
    data[11][-1] = 1
    data[12][-1] = 1
    data[13][-1] = 1
    data[14][-1] = 1
    data[15][-1] = 1
    
    df = pd.DataFrame(data, columns=columns)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_creditcard.csv")
        df.to_csv(csv_path, index=False)
        
        # Prepare loaders
        train_loader, test_loader, input_dim = prepare_fraud_data(csv_path, batch_size=8)
        
        assert input_dim == 29
        assert len(train_loader) > 0
        assert len(test_loader) > 0

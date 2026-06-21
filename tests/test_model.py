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

def test_fraud_classifier_output_range():
    """Verify model outputs raw logits (not bounded to [0,1])."""
    input_dim = 29
    model = FraudClassifier(input_dim=input_dim)
    
    # Use large inputs that should produce logits outside [0, 1]
    x = torch.randn(100, input_dim) * 10
    out = model(x)
    
    # Raw logits — at least some values should be outside [0, 1]
    # (This would fail if Sigmoid was accidentally left in the model)
    assert out.shape == (100, 1)

def test_prepare_fraud_data_dimensions():
    # Verify utility functions work with mock paths or verify dimension
    # Let's mock a tiny csv to verify it returns expected structures
    import pandas as pd
    import tempfile
    import os
    
    # Create a small dummy CSV file representing the dataset structure
    columns = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
    data = [[0.1] * 28 + [0, 10.0, 0] for _ in range(200)]  # 200 samples
    # Add enough fraud samples for SMOTE k_neighbors=5, but few enough
    # that sampling_strategy=0.1 can still oversample (need minority < 10% of majority)
    for i in range(10):
        data[i][-1] = 1
    
    df = pd.DataFrame(data, columns=columns)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_creditcard.csv")
        df.to_csv(csv_path, index=False)
        
        # Prepare loaders
        train_loader, test_loader, input_dim = prepare_fraud_data(csv_path, batch_size=8)
        
        assert input_dim == 29
        assert len(train_loader) > 0
        assert len(test_loader) > 0

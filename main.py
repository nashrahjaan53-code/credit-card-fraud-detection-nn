import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import prepare_fraud_data
from src.model import FraudClassifier
from src.engine import train_epoch, evaluate_and_tune

def main():
    # Setup configurations
    CSV_PATH = "data/creditcard.csv" # Path to your downloaded Kaggle dataset
    BATCH_SIZE = 2048                # Larger batch size works well for fraud tabular data
    EPOCHS = 15
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")
    
    # 1. Pipeline: Load, Split, SMOTE, and Tensorize
    train_loader, test_loader, input_dim = prepare_fraud_data(CSV_PATH, batch_size=BATCH_SIZE)
    
    # 2. Instantiate Model, Loss Function, and Optimizer
    model = FraudClassifier(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 3. Training Loop
    print("\n Starting Neural Network Training...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {loss:.4f}")
            
    # 4. Evaluation and Business Optimization Tuning
    print("\n Running Threshold Tuning Optimization...")
    best_thresh = evaluate_and_tune(model, test_loader, device)
    
    # 5. Save the production-ready weights
    torch.save(model.state_dict(), "src/fraud_model_weights.pth")
    print(f"\n Model weights saved successfully! Tuned Threshold: {best_thresh:.4f}")

if __name__ == "__main__":
    main()
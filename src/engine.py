import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_curve, classification_report

def train_epoch(model, loader, optimizer, criterion, device):
    """Runs a single training epoch over the resampled SMOTE data."""
    model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
        
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_and_tune(model, loader, device, cost_fn=500, cost_fp=15):
    """
    Evaluates the model on raw validation/test data (highly imbalanced)
    and tunes the classification threshold based on business metrics.
    """
    model.eval()
    all_probs = []
    all_targets = []
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits) # Convert stable logits to probabilities
        
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(y_batch.numpy())
        
    all_probs = np.array(all_probs).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate Precision-Recall pairs for all thresholds
    precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)
    
    # Find optimal threshold using Business Cost Optimization
    # Goal: Minimize (False Negatives * cost_fn) + (False Positives * cost_fp)
    best_threshold = 0.5
    min_cost = float('inf')
    
    for t in np.linspace(0.01, 0.99, 100):
        preds = (all_probs >= t).astype(int)
        
        # Calculate matrix elements manually for threshold evaluation
        fn = np.sum((all_targets == 1) & (preds == 0))
        fp = np.sum((all_targets == 0) & (preds == 1))
        
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        
        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = t
            
    print(f"\nOptimal Financial Threshold Found: {best_threshold:.4f}")
    print(f" Minimum Expected Fraud Overhead Cost: ${min_cost:,}")
    
    # Print the final polished report at the tuned threshold
    final_preds = (all_probs >= best_threshold).astype(int)
    print("\n Final Classification Report:")
    print(classification_report(all_targets, final_preds, target_names=['Legit', 'Fraud']))
    
    return best_threshold
import torch.nn as nn

class FraudClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FraudClassifier, self).__init__()
        
        # Simple but highly effective architecture for tabular feature sets
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1) # Output raw logits (safer for BCEWithLogitsLoss)
        )
        
    def forward(self, x):
        return self.network(x)
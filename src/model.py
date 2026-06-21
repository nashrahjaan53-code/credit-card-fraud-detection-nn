import torch.nn as nn

class FraudClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FraudClassifier, self).__init__()
        
        # Production architecture for tabular fraud detection
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)  # Output raw logits (safer for BCEWithLogitsLoss)
        )
        
    def forward(self, x):
        return self.network(x)
import torch
import torch.nn as nn

class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator for adversarial training.
    The input is the outer product of features and class predictions in CDAN.
    
    Args:
        in_features: Input feature dimension (2048 * num_classes)
        hidden_size: Hidden layer dimension, default is 1024
        dropout_rate: Dropout rate, default is 0.5
    """
    def __init__(self, in_features, hidden_size=1024, dropout_rate=0.5):
        super(DomainDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input features, shape (N, in_features)
        Returns:
            domain_pred: Domain prediction logits, shape (N, 1)
        """
        return self.discriminator(x)

    def get_parameters(self):
        """Get trainable parameters"""
        return [{"params": self.parameters(), "lr_mult": 1.}]

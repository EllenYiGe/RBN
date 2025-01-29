import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    """
    Classifier Head that maps features to the class space.
    
    Args:
        in_features: Input feature dimension
        num_classes: Number of classes
        dropout_rate: Dropout rate, default is 0.5
    """
    def __init__(self, in_features, num_classes, dropout_rate=0.5):
        super(ClassifierHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input features, shape (N, in_features)
        Returns:
            logits: Class logits, shape (N, num_classes)
        """
        return self.classifier(x)

import torch
import torch.nn as nn
from .backbone import get_resnet50_backbone
from .classifier import ClassifierHead

class CDAN(nn.Module):
    def __init__(self, num_classes=31, use_rbn=True, replace_layer=3, pretrained=True):
        super(CDAN, self).__init__()
        self.feature_extractor = get_resnet50_backbone(
            use_rbn=use_rbn,
            replace_starting_layer=replace_layer,
            pretrained=pretrained
        )
        self.classifier = ClassifierHead(
            in_features=2048,
            num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input image, shape (N, C, H, W)
        Returns:
            features: Extracted features, shape (N, 2048)
            logits: Class prediction logits, shape (N, num_classes)
        """
        # Feature extraction
        feat = self.feature_extractor(x)
        feat = feat.view(feat.size(0), -1)  # (N, 2048)
        
        # Classification prediction
        logits = self.classifier(feat)
        
        return feat, logits

    def get_parameters(self, base_lr=1.0):
        """
        Get model parameters for the optimizer
        Args:
            base_lr: Base learning rate
        Returns:
            Parameter list with learning rate multipliers
        """
        params = [
            {"params": self.feature_extractor.parameters(), "lr_mult": 0.1},
            {"params": self.classifier.parameters(), "lr_mult": 1.},
        ]
        return params

import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    """
    分类器头部，将特征映射到类别空间
    
    Args:
        in_features: 输入特征维度
        num_classes: 类别数
        dropout_rate: dropout比率，默认0.5
    """
    def __init__(self, in_features, num_classes, dropout_rate=0.5):
        super(ClassifierHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征，形状为 (N, in_features)
        Returns:
            logits: 类别logits，形状为 (N, num_classes)
        """
        return self.classifier(x)

import torch
import torch.nn as nn

class DomainDiscriminator(nn.Module):
    """
    域判别器，用于对抗训练
    输入为CDAN中特征与类别预测的外积
    
    Args:
        in_features: 输入特征维度 (2048 * num_classes)
        hidden_size: 隐层维度，默认1024
        dropout_rate: dropout比率，默认0.5
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
        前向传播
        Args:
            x: 输入特征，形状为 (N, in_features)
        Returns:
            domain_pred: 域预测logits，形状为 (N, 1)
        """
        return self.discriminator(x)

    def get_parameters(self):
        """获取可训练参数"""
        return [{"params": self.parameters(), "lr_mult": 1.}]

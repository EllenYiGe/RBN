import torch
import torch.nn as nn
from .backbone import get_resnet50_backbone
from .classifier import ClassifierHead

class CDAN(nn.Module):
    """
    CDAN (Conditional Domain Adversarial Networks) 模型
    包含特征提取器(ResNet-50)和分类器头部
    可选使用RBN替换特定层的BN
    
    Args:
        num_classes: 类别数
        use_rbn: 是否使用RBN
        replace_layer: 从第几层开始替换BN为RBN
        pretrained: 是否使用预训练的backbone
    """
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
        前向传播
        Args:
            x: 输入图像，形状为 (N, C, H, W)
        Returns:
            features: 提取的特征，形状为 (N, 2048)
            logits: 类别预测logits，形状为 (N, num_classes)
        """
        # 特征提取
        feat = self.feature_extractor(x)
        feat = feat.view(feat.size(0), -1)  # (N, 2048)
        
        # 分类预测
        logits = self.classifier(feat)
        
        return feat, logits

    def get_parameters(self, base_lr=1.0):
        """
        获取模型参数，用于优化器
        Args:
            base_lr: 基础学习率
        Returns:
            参数列表，包含学习率倍率
        """
        params = [
            {"params": self.feature_extractor.parameters(), "lr_mult": 0.1},
            {"params": self.classifier.parameters(), "lr_mult": 1.},
        ]
        return params

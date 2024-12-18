import torch
import torch.nn as nn

class RBN(nn.Module):
    """
    Refined Batch Normalization (RBN)
    使用GroupNorm替换BN，以解决域适应中BN统计量估计偏移问题
    
    Args:
        num_channels: 输入特征通道数
        num_groups: 分组数，默认32
        eps: 数值稳定性参数
        affine: 是否使用可学习的仿射参数
    """
    def __init__(self, num_channels, num_groups=32, eps=1e-5, affine=True):
        super(RBN, self).__init__()
        # 确保分组数不超过通道数
        num_groups = min(num_groups, num_channels)
        self.gn = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征图，形状为 (N, C, H, W)
        Returns:
            归一化后的特征图，形状与输入相同
        """
        return self.gn(x)

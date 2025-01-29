import torch
import torch.nn as nn

class RBN(nn.Module):
    """
    Refined Batch Normalization (RBN)
    Replaces BatchNorm with GroupNorm to address batch statistic estimation bias in domain adaptation.
    
    Args:
        num_channels: Number of input feature channels
        num_groups: Number of groups, default is 32
        eps: Numerical stability parameter
        affine: Whether to use learnable affine parameters
    """
    def __init__(self, num_channels, num_groups=32, eps=1e-5, affine=True):
        super(RBN, self).__init__()
        # Ensure the number of groups does not exceed the number of channels
        num_groups = min(num_groups, num_channels)
        self.gn = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine
        )

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input feature map, shape (N, C, H, W)
        Returns:
            Normalized feature map, same shape as input
        """
        return self.gn(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class NonLocalBlock(nn.Module):
    """Non-local Neural Networks Block"""
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(self.in_channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, 1)
        self.W = nn.Conv2d(self.inter_channels, self.in_channels, 1)
        
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class ResNetBackbone(nn.Module):
    """Enhanced ResNet Backbone"""
    def __init__(self, arch='resnet50', pretrained=True):
        super(ResNetBackbone, self).__init__()
        
        # Load the pre-trained model
        model = getattr(models, arch)(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(model.children())[:-2])
        
        # Get the feature dimension
        self.feature_dim = model.fc.in_features
        
        # Add attention modules
        self.se_block = SEBlock(self.feature_dim)
        self.non_local = NonLocalBlock(self.feature_dim)
        
        # Add global pooling layers
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        # Dropout regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Extract features
        x = self.features(x)
        
        # Apply attention mechanisms
        x = self.non_local(x)
        x = self.se_block(x)
        
        # Global pooling
        avg_x = self.gap(x)
        max_x = self.gmp(x)
        x = avg_x + max_x
        
        # Flatten features
        x = x.view(x.size(0), -1)
        
        # Dropout regularization
        x = self.dropout(x)
        
        return x

    def output_dim(self):
        return self.feature_dim

def get_resnet_backbone(arch='resnet50', pretrained=True):
    """
    Get ResNet Backbone
    
    Args:
        arch: ResNet model type
        pretrained: Whether to use pre-trained weights
    Returns:
        ResNet backbone model
    """
    return ResNetBackbone(arch, pretrained)

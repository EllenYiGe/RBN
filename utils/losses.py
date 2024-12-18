import torch
import torch.nn.functional as F

def domain_adversarial_loss(predictions, domain_labels, reduction='mean'):
    """
    计算域对抗损失
    使用二元交叉熵，源域标签为1，目标域标签为0
    
    Args:
        predictions: 域判别器输出，形状为 (N, 1)
        domain_labels: 域标签，形状为 (N,)，值为0或1
        reduction: 损失计算方式，'mean'或'sum'
    Returns:
        loss: 标量损失值
    """
    predictions = predictions.view(-1)
    domain_labels = domain_labels.float()
    
    loss = F.binary_cross_entropy_with_logits(
        predictions,
        domain_labels,
        reduction=reduction
    )
    return loss

def entropy_loss(predictions, reduction='mean'):
    """
    计算预测的熵损失
    用于目标域样本的不确定性度量
    
    Args:
        predictions: 分类预测logits，形状为 (N, C)
        reduction: 损失计算方式，'mean'或'sum'
    Returns:
        entropy: 熵损失值
    """
    probs = F.softmax(predictions, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    
    if reduction == 'mean':
        return entropy.mean()
    elif reduction == 'sum':
        return entropy.sum()
    else:
        return entropy

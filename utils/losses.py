import torch
import torch.nn.functional as F

def domain_adversarial_loss(predictions, domain_labels, reduction='mean'):
    """
    Compute domain adversarial loss.
    Uses binary cross-entropy where source domain labels are 1 and target domain labels are 0.
    
    Args:
        predictions: Output from the domain discriminator, shape (N, 1).
        domain_labels: Domain labels, shape (N,), values are 0 or 1.
        reduction: Method for loss computation, either 'mean' or 'sum'.
    Returns:
        loss: Scalar loss value.
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
    Compute entropy loss of predictions.
    Used to measure the uncertainty of target domain samples.
    
    Args:
        predictions: Classification prediction logits, shape (N, C).
        reduction: Method for loss computation, either 'mean' or 'sum'.
    Returns:
        entropy: Entropy loss value.
    """
    probs = F.softmax(predictions, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    
    if reduction == 'mean':
        return entropy.mean()
    elif reduction == 'sum':
        return entropy.sum()
    else:
        return entropy

import torch
import torch.nn.functional as F
import datetime
import os

class SimpleLogger:
    """
    Simple logging class.
    Supports output to both console and file.
    
    Args:
        log_file: Path to the log file, None means output to console only.
        append: Whether to open the log file in append mode.
    """
    def __init__(self, log_file=None, append=True):
        self.log_file = log_file
        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self.f = open(log_file, 'a' if append else 'w')
        else:
            self.f = None

    def log(self, msg, with_time=True):
        """
        Record a log message.
        
        Args:
            msg: Log message.
            with_time: Whether to add a timestamp.
        """
        if with_time:
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{time_str}] {msg}"
        else:
            line = msg
            
        print(line)
        if self.f is not None:
            self.f.write(line + '\n')
            self.f.flush()

    def close(self):
        """Close the log file."""
        if self.f is not None:
            self.f.close()
            self.f = None

    def __del__(self):
        self.close()


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

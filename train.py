import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import numpy as np
from models.cdan import CDAN
from models.discriminator import DomainDiscriminator
from utils.utils import load_data, save_model
from utils.losses import domain_adversarial_loss, entropy_loss
from utils.logger import SimpleLogger
from torch.cuda.amp import autocast, GradScaler
import logging
import wandb

class EMA:
    """Exponential Moving Average (EMA)"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

def get_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train CDAN+RBN for Domain Adaptation")
    
    # Data-related arguments
    parser.add_argument('--source_data', type=str, default='data/office31/amazon',
                        help='Path to source domain data')
    parser.add_argument('--target_data', type=str, default='data/office31/webcam',
                        help='Path to target domain data')
    parser.add_argument('--dataset_type', type=str, default='office31',
                        choices=['office31', 'imageclef', 'officehome', 'visda'],
                        help='Dataset type')
    parser.add_argument('--label_file_source', type=str, default=None,
                        help='Path to source domain label file (required for some datasets)')
    parser.add_argument('--label_file_target', type=str, default=None,
                        help='Path to target domain label file (required for some datasets)')
    
    # Model-related arguments
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--use_rbn', action='store_true', help='Use RBN')
    parser.add_argument('--replace_layer', type=int, default=3,
                        help='Replace BN with RBN from this layer')
    
    # Training-related arguments
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Base learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--lr_gamma', type=float, default=0.001,
                        help='Learning rate decay gamma')
    parser.add_argument('--lr_decay', type=float, default=0.75,
                        help='Learning rate decay factor')
    
    # Saving-related arguments
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Checkpoint save interval (epochs)')
    
    # Advanced training techniques
    parser.add_argument('--use_ema', action='store_true', help='Use EMA')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    parser.add_argument('--use_mixup', action='store_true', help='Use MixUp')
    parser.add_argument('--mixup_alpha', type=float, default=0.5, help='MixUp coefficient')
    parser.add_argument('--use_tta', action='store_true', help='Use Test-Time Augmentation (TTA)')
    parser.add_argument('--clip_grad', action='store_true', help='Apply gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Gradient clipping threshold')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases (WandB)')
    parser.add_argument('--project_name', type=str, default='CDAN+RBN', help='WandB project name')
    
    args = parser.parse_args()
    return args

def train():
    args = get_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger = SimpleLogger(os.path.join(args.output_dir, 'train.log'))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")
    
    # Load data
    source_loader, target_loader = load_data(
        args.source_data, args.target_data,
        batch_size=args.batch_size,
        dataset_type=args.dataset_type,
        label_file_source=args.label_file_source,
        label_file_target=args.label_file_target
    )
    logger.log("Data loaded successfully")
    
    # Build model
    model = CDAN(
        num_classes=args.num_classes,
        use_rbn=args.use_rbn,
        replace_layer=args.replace_layer
    ).to(device)
    
    domain_disc = DomainDiscriminator(
        in_features=2048 * args.num_classes
    ).to(device)
    
    # Create EMA
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    
    # Create optimizer
    optimizer = optim.SGD(
        model.get_parameters(args.lr),
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optimizer_disc = optim.SGD(
        domain_disc.get_parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Create loss function
    cls_criterion = nn.CrossEntropyLoss()
    
    # Create mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        domain_disc.train()
        
        # Adjust learning rate
        lr = args.lr * (1 + args.lr_gamma * epoch) ** (-args.lr_decay)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
        for param_group in optimizer_disc.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
        
        total_loss = 0
        cls_loss_total = 0
        transfer_loss_total = 0
        n_batches = min(len(source_loader), len(target_loader))
        
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        for batch_idx in range(n_batches):
            # Load source and target data
            source_data, source_label = next(source_iter)
            target_data, _ = next(target_iter)
            source_data, source_label, target_data = source_data.to(device), source_label.to(device), target_data.to(device)
            
            # Forward pass
            source_feat, source_pred = model(source_data)
            target_feat, target_pred = model(target_data)
            
            # Classification loss
            cls_loss = cls_criterion(source_pred, source_label)
            
            # Domain adversarial loss
            source_softmax = torch.softmax(source_pred, dim=1)
            target_softmax = torch.softmax(target_pred, dim=1)
            
            cdan_input = torch.cat((
                torch.bmm(source_softmax.unsqueeze(2), source_feat.unsqueeze(1)).view(-1, source_feat.size(1) * args.num_classes),
                torch.bmm(target_softmax.unsqueeze(2), target_feat.unsqueeze(1)).view(-1, target_feat.size(1) * args.num_classes)
            ), dim=0)
            
            domain_label = torch.cat((
                torch.ones(source_data.size(0)), torch.zeros(target_data.size(0))
            )).to(device)
            
            transfer_loss = domain_adversarial_loss(domain_disc(cdan_input), domain_label)
            
            # Total loss
            loss = cls_loss + transfer_loss
            
            # Backward pass
            optimizer.zero_grad()
            optimizer_disc.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(optimizer_disc)
            scaler.update()
            
            if args.use_ema:
                ema.update()
            
            scheduler.step(epoch + batch_idx / n_batches)
        
    logger.log("Training completed")

if __name__ == "__main__":
    train()

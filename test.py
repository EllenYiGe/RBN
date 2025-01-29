import argparse
import os
import torch
from tqdm import tqdm
import numpy as np

from models.cdan import CDAN
from utils.utils import load_test_data
from utils.logger import SimpleLogger

def get_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Test CDAN+RBN Model")
    
    # Data-related arguments
    parser.add_argument('--test_data', type=str, default='data/office31/webcam',
                        help='Path to test data')
    parser.add_argument('--dataset_type', type=str, default='office31',
                        choices=['office31', 'imageclef', 'officehome', 'visda'],
                        help='Dataset type')
    parser.add_argument('--label_file_test', type=str, default=None,
                        help='Path to test set label file (required for some datasets)')
    
    # Model-related arguments
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of classes')
    parser.add_argument('--use_rbn', action='store_true',
                        help='Use RBN (must match training configuration)')
    parser.add_argument('--replace_layer', type=int, default=3,
                        help='RBN replacement layer (must match training configuration)')
    parser.add_argument('--model_path', type=str, default='output/model_final.pth',
                        help='Path to model file')
    
    # Testing-related arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    
    args = parser.parse_args()
    return args

def test():
    args = get_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger = SimpleLogger(os.path.join(args.output_dir, 'test.log'))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")
    
    # Load test data
    test_loader = load_test_data(
        args.test_data,
        batch_size=args.batch_size,
        dataset_type=args.dataset_type,
        label_file_test=args.label_file_test
    )
    logger.log("Test data loaded successfully")
    
    # Build and load model
    model = CDAN(
        num_classes=args.num_classes,
        use_rbn=args.use_rbn,
        replace_layer=args.replace_layer
    ).to(device)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file {args.model_path} does not exist")
    
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.log("Model loaded successfully")
    
    # Testing
    correct = 0
    total = 0
    class_correct = np.zeros(args.num_classes, dtype=np.int64)
    class_total = np.zeros(args.num_classes, dtype=np.int64)
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            _, outputs = model(data)
            predictions = outputs.argmax(dim=1)
            
            # Compute overall accuracy
            correct += (predictions == target).sum().item()
            total += target.size(0)
            
            # Compute per-class accuracy
            for i in range(args.num_classes):
                idx = (target == i)
                class_correct[i] += (predictions[idx] == target[idx]).sum().item()
                class_total[i] += idx.sum().item()
    
    # Compute and log results
    overall_acc = 100. * correct / total
    logger.log(f"\nOverall Accuracy: {overall_acc:.2f}%")
    
    # Log per-class accuracy
    logger.log("\nPer-Class Accuracy:")
    for i in range(args.num_classes):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            logger.log(f"Class {i:2d}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    # Compute mean class accuracy
    valid_classes = class_total > 0
    mean_class_acc = 100. * class_correct[valid_classes].sum() / class_total[valid_classes].sum()
    logger.log(f"\nMean Class Accuracy: {mean_class_acc:.2f}%")
    
    logger.log("Testing complete")

if __name__ == "__main__":
    test()

import torch
import torch.nn as nn
from models.backbone import ResNetBackbone
from datasets.transforms import get_train_transform, get_test_transform, MixUp
import yaml
from PIL import Image
import numpy as np

def test_model():
    # Load configuration
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetBackbone(
        arch=config['model']['arch'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    # Create test data
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    
    # Test forward pass
    try:
        output = model(dummy_input)
        print(f"Model forward pass successful. Output shape: {output.shape}")
    except Exception as e:
        print(f"Error in model forward pass: {str(e)}")
        return False
    
    # Test data augmentation
    try:
        train_transform = get_train_transform()
        test_transform = get_test_transform()
        
        # Create test image
        dummy_image = Image.fromarray(
            (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        )
        
        # Test training data augmentation
        transformed_train = train_transform(dummy_image)
        print(f"Train transform successful. Output shape: {transformed_train.shape}")
        
        # Test testing data augmentation
        transformed_test = test_transform(dummy_image)
        print(f"Test transform successful. Output shape: {transformed_test.shape}")
    except Exception as e:
        print(f"Error in data augmentation: {str(e)}")
        return False
    
    # Test MixUp
    try:
        mixup = MixUp(alpha=1.0)
        batch = (
            torch.randn(4, 3, 224, 224).to(device),
            torch.randint(0, 10, (4,)).float().to(device)
        )
        mixed_images, mixed_labels = mixup(batch)
        print(f"MixUp successful. Output shapes: {mixed_images.shape}, {mixed_labels.shape}")
    except Exception as e:
        print(f"Error in MixUp: {str(e)}")
        return False
    
    print("All tests passed successfully!")
    return True

if __name__ == "__main__":
    test_model()

import os
import torch
from torch.utils.data import DataLoader

from datasets.office31 import Office31Dataset
from datasets.imageclef import ImageClefDataset
from datasets.officehome import OfficeHomeDataset
from datasets.visda2017 import Visda2017Dataset
from datasets.transforms import get_train_transform, get_test_transform

def load_data(source_path, target_path, batch_size=32, dataset_type='office31',
              label_file_source=None, label_file_target=None, num_workers=4):
    """
    Load source and target domain data.
    
    Args:
        source_path: Path to the source domain data.
        target_path: Path to the target domain data.
        batch_size: Batch size.
        dataset_type: Type of dataset.
        label_file_source: Path to the source domain label file (required for some datasets).
        label_file_target: Path to the target domain label file (required for some datasets).
        num_workers: Number of data loading threads.
    Returns:
        source_loader: Data loader for the source domain.
        target_loader: Data loader for the target domain.
    """
    train_transform = get_train_transform()
    
    # Create dataset instances
    if dataset_type == 'office31':
        source_dataset = Office31Dataset(root_dir=source_path, transform=train_transform)
        target_dataset = Office31Dataset(root_dir=target_path, transform=train_transform)
    elif dataset_type == 'imageclef':
        source_dataset = ImageClefDataset(root_dir=source_path, transform=train_transform,
                                          label_file=label_file_source)
        target_dataset = ImageClefDataset(root_dir=target_path, transform=train_transform,
                                          label_file=label_file_target)
    elif dataset_type == 'officehome':
        source_dataset = OfficeHomeDataset(root_dir=source_path, transform=train_transform)
        target_dataset = OfficeHomeDataset(root_dir=target_path, transform=train_transform)
    elif dataset_type == 'visda':
        if not (label_file_source and label_file_target):
            raise ValueError("VisDA dataset requires label_file to be provided.")
        source_dataset = Visda2017Dataset(root_dir=source_path, transform=train_transform,
                                          label_file=label_file_source)
        target_dataset = Visda2017Dataset(root_dir=target_path, transform=train_transform,
                                          label_file=label_file_target)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create data loaders
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    return source_loader, target_loader

def load_test_data(test_path, batch_size=32, dataset_type='office31',
                   label_file_test=None, num_workers=4):
    """
    Load test data.
    
    Args:
        test_path: Path to the test data.
        batch_size: Batch size.
        dataset_type: Type of dataset.
        label_file_test: Path to the test set label file (required for some datasets).
        num_workers: Number of data loading threads.
    Returns:
        test_loader: Data loader for the test set.
    """
    test_transform = get_test_transform()
    
    if dataset_type == 'office31':
        test_dataset = Office31Dataset(root_dir=test_path, transform=test_transform)
    elif dataset_type == 'imageclef':
        test_dataset = ImageClefDataset(root_dir=test_path, transform=test_transform,
                                        label_file=label_file_test)
    elif dataset_type == 'officehome':
        test_dataset = OfficeHomeDataset(root_dir=test_path, transform=test_transform)
    elif dataset_type == 'visda':
        if not label_file_test:
            raise ValueError("VisDA test requires label_file to be provided.")
        test_dataset = Visda2017Dataset(root_dir=test_path, transform=test_transform,
                                        label_file=label_file_test)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader

def save_model(model, save_path, epoch=None):
    """
    Save model.
    
    Args:
        model: Model to be saved.
        save_path: Path to save the model.
        epoch: Current epoch, used for filename.
    """
    if epoch is not None:
        save_path = f"{os.path.splitext(save_path)[0]}_epoch{epoch}.pth"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

def load_model(model, model_path, device='cpu'):
    """
    Load model.
    
    Args:
        model: Model instance.
        model_path: Path to the model file.
        device: Device.
    Returns:
        Model with loaded weights.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
        
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model

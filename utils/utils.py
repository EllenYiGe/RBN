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
    加载源域和目标域数据
    
    Args:
        source_path: 源域数据路径
        target_path: 目标域数据路径
        batch_size: batch大小
        dataset_type: 数据集类型
        label_file_source: 源域label文件路径（部分数据集需要）
        label_file_target: 目标域label文件路径（部分数据集需要）
        num_workers: 数据加载线程数
    Returns:
        source_loader: 源域数据加载器
        target_loader: 目标域数据加载器
    """
    train_transform = get_train_transform()
    
    # 创建数据集实例
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
            raise ValueError("VisDA数据集需要提供label_file")
        source_dataset = Visda2017Dataset(root_dir=source_path, transform=train_transform,
                                        label_file=label_file_source)
        target_dataset = Visda2017Dataset(root_dir=target_path, transform=train_transform,
                                        label_file=label_file_target)
    else:
        raise ValueError(f"未知数据集类型: {dataset_type}")
    
    # 创建数据加载器
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
    加载测试数据
    
    Args:
        test_path: 测试数据路径
        batch_size: batch大小
        dataset_type: 数据集类型
        label_file_test: 测试集label文件路径（部分数据集需要）
        num_workers: 数据加载线程数
    Returns:
        test_loader: 测试数据加载器
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
            raise ValueError("VisDA测试需要提供label_file")
        test_dataset = Visda2017Dataset(root_dir=test_path, transform=test_transform,
                                      label_file=label_file_test)
    else:
        raise ValueError(f"未知数据集类型: {dataset_type}")
    
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
    保存模型
    
    Args:
        model: 要保存的模型
        save_path: 保存路径
        epoch: 当前轮数，用于文件名
    """
    if epoch is not None:
        save_path = f"{os.path.splitext(save_path)[0]}_epoch{epoch}.pth"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

def load_model(model, model_path, device='cpu'):
    """
    加载模型
    
    Args:
        model: 模型实例
        model_path: 模型文件路径
        device: 设备
    Returns:
        加载权重后的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model

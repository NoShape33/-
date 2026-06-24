"""
MNIST数据加载、预处理与可视化
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def get_transforms(train=True):
    """获取数据变换：训练集含数据增强，测试集仅归一化"""
    if train:
        return transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


def get_dataloaders(batch_size=64, val_split=0.2):
    """
    加载MNIST数据集，返回训练/验证/测试DataLoader
    """
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    full_train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform
    )

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_test_loader_for_eval(batch_size=1000):
    """获取用于最终评估的测试集DataLoader（不打乱，方便按顺序分析）"""
    test_transform = get_transforms(train=False)
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def plot_dataset_samples(save_path=None):
    """
    可视化MNIST数据集样本（每类展示一例），用于论��"数据介绍"部分
    """
    dataset = datasets.MNIST(root='./data', train=True, download=True,
                              transform=transforms.ToTensor())

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()
    for digit in range(10):
        for img, label in dataset:
            if label == digit:
                axes[digit].imshow(img.squeeze(), cmap='gray')
                axes[digit].set_title(f'Digit: {digit}')
                axes[digit].axis('off')
                break

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

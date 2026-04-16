"""CIFAR-10 数据加载"""
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import CIFAR10_ROOT, BATCH_SIZE


def get_cifar10_loaders(
    root=None,
    batch_size=None,
    val_ratio=0.1,
    num_workers=8,
    random_erasing_p=0.0,
):
    """返回 CIFAR-10 train/val DataLoader。

    random_erasing_p: RandomErasing 概率（在 ToTensor 之后），常用 0.2~0.5 提升泛化。
    """
    root = root or CIFAR10_ROOT
    batch_size = batch_size or BATCH_SIZE

    # CIFAR-10 标准变换
    train_t = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ]
    if random_erasing_p and random_erasing_p > 0:
        train_t.append(transforms.RandomErasing(p=float(random_erasing_p)))
    train_transform = transforms.Compose(train_t)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    # 使用 cifar-10-batches-py 父目录
    data_dir = os.path.dirname(root) if os.path.basename(root) == "cifar-10-batches-py" else root
    full_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=not os.path.exists(root),
        transform=train_transform,
    )
    n_val = int(len(full_train) * val_ratio)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def get_cifar10_test_loader(root=None, batch_size=None, num_workers=4):
    """返回 CIFAR-10 test DataLoader。"""
    root = root or CIFAR10_ROOT
    batch_size = batch_size or BATCH_SIZE
    data_dir = os.path.dirname(root) if os.path.basename(root) == "cifar-10-batches-py" else root

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])
    test_ds = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=not os.path.exists(root),
        transform=test_transform,
    )
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

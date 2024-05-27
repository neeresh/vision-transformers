from typing import List, Union

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torchvision.transforms.v2 import ToTensor, Normalize


def get_train_test_loaders(dataset_name="cifar100", batch_size=128, num_workers=8, val_split=None):
    train_dataset = None
    test_dataset = None

    train_transforms, test_transforms = _get_transformations(dataset_name)

    if dataset_name.lower() == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transforms)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transforms)
    elif dataset_name.lower() == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
    else:
        print(f"Dataset {dataset_name} is not supported.")

    if val_split:
        train_set_len = len(train_dataset)
        val_set_len = int(train_set_len * val_split)
        train_set_len = train_set_len - val_set_len

        train_dataset, val_dataset = random_split(train_dataset, [train_set_len, val_set_len])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader

    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader


def _get_transformations(dataset_name):

    if dataset_name.lower() == "cifar100":
        common_transforms = [v2.ToTensor(), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        train_transforms = [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip(), v2.ColorJitter(brightness=63 / 255)]

        train_transforms = v2.Compose([*train_transforms, *common_transforms])
        test_transforms = common_transforms

        return train_transforms, test_transforms

    elif dataset_name.lower() == "cifar10":
        common_transforms = [v2.ToTensor(), v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]
        train_transforms = [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip(p=0.5), v2.ColorJitter(brightness=63 / 255),]

        train_transforms = v2.Compose([*train_transforms, *common_transforms])
        test_transforms = common_transforms

        return train_transforms, test_transforms

    elif dataset_name.lower() == "imagenet100":
        common_transforms = [v2.ToTensor(), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
        train_transforms = [v2.RandomResizedCrop(224), v2.RandomHorizontalFlip()]
        test_transforms = [v2.Resize(256), v2.CenterCrop(224)]

        train_transforms = v2.Compose([*train_transforms, *common_transforms])
        test_transforms = v2.Compose([*test_transforms, *common_transforms])

        return train_transforms, test_transforms

    elif dataset_name.lower() == "imagenet1000":
        common_transforms = [v2.ToTensor(), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        train_transforms = [v2.RandomResizedCrop(224), v2.RandomHorizontalFlip(), v2.ColorJitter(brightness=63 / 255)]
        test_transforms = [v2.Resize(256), v2.CenterCrop(224),]

        train_transforms = v2.Compose([*train_transforms, *common_transforms])
        test_transforms = v2.Compose([*test_transforms, *common_transforms])

        return train_transforms, test_transforms

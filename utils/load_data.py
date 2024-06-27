import os

import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2


def get_train_test_loaders(dataset_name="cifar100", batch_size=128, num_workers=8, val_split=None,
                           root_dir="../../data"):
    train_dataset = None
    test_dataset = None

    train_transforms, test_transforms = _get_transformations(dataset_name)

    if dataset_name.lower() == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(root=root_dir, train=True, download=True, transform=train_transforms)
        test_dataset = torchvision.datasets.CIFAR100(root=root_dir, train=False, download=True, transform=test_transforms)
    elif dataset_name.lower() == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=train_transforms)
        test_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=test_transforms)
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
        common_transforms = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        train_transforms = [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip(), v2.ColorJitter(brightness=63 / 255)]

        train_transforms = v2.Compose([*train_transforms, *common_transforms])
        test_transforms = v2.Compose(*[common_transforms])

        return train_transforms, test_transforms

    elif dataset_name.lower() == "cifar10":
        common_transforms = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]
        train_transforms = [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip(p=0.5), v2.ColorJitter(brightness=63 / 255),]

        train_transforms = v2.Compose([*train_transforms, *common_transforms])
        test_transforms = v2.Compose(*[common_transforms])

        return train_transforms, test_transforms

    elif dataset_name.lower() == "imagenet100":
        common_transforms = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
        train_transforms = [v2.RandomResizedCrop(224), v2.RandomHorizontalFlip()]
        test_transforms = [v2.Resize(256), v2.CenterCrop(224)]

        train_transforms = v2.Compose([*train_transforms, *common_transforms])
        test_transforms = v2.Compose([*test_transforms, *common_transforms])

        return train_transforms, test_transforms

    elif dataset_name.lower() == "imagenet1000":
        common_transforms = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        train_transforms = [v2.RandomResizedCrop(224), v2.RandomHorizontalFlip(), v2.ColorJitter(brightness=63 / 255)]
        test_transforms = [v2.Resize(256), v2.CenterCrop(224),]

        train_transforms = v2.Compose([*train_transforms, *common_transforms])
        test_transforms = v2.Compose([*test_transforms, *common_transforms])

        return train_transforms, test_transforms


class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, load_data, transform=None, target_transform=None):

        if load_data == 'train':
            root = './data/coco/images/train2017'
            annFile = './data/coco/annotations/instances_train2017.json'
        elif load_data == 'val':
            root = './data/coco/images/val2017'
            annFile = './data/coco/annotations/instances_val2017.json'
        elif load_data == 'test':
            root = './data/coco/images/test2017'
        else:
            ValueError(f"Load data options: 'train', 'val', and 'test'")

        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



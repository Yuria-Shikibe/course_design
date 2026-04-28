import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np


def get_dataset(name, data_dir='./data'):
    if name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    elif name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        train_set = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    elif name == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    elif name == 'cifar100':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_set = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_transform)

    elif name == 'tiny-imagenet-200':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_set = datasets.ImageFolder(f'{data_dir}/tiny-imagenet-200/train', transform=transform)
        test_set = datasets.ImageFolder(f'{data_dir}/tiny-imagenet-200/val', transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_set, test_set


def partition_data(dataset, num_clients, seed=42, non_iid=True, alpha=0.5):
    np.random.seed(seed)
    num_samples = len(dataset)
    targets = np.array([y for _, y in dataset])
    client_indices = {i: [] for i in range(num_clients)}

    if non_iid:
        num_classes = len(np.unique(targets))
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_k) < num_clients / num_classes)
                                    for p in proportions])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            split = np.split(idx_k, proportions)
            for i, s in enumerate(split):
                client_indices[i].extend(s.tolist())
    else:
        indices = np.random.permutation(num_samples)
        samples_per_client = num_samples // num_clients
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client if i < num_clients - 1 else num_samples
            client_indices[i] = indices[start:end].tolist()

    return client_indices


def create_client_loaders(dataset, client_indices, batch_size):
    loaders = {}
    for client_id, indices in client_indices.items():
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        loaders[client_id] = loader
    return loaders


def create_test_loader(test_set, batch_size):
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)

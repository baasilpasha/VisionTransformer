# Dowloads CIFAR-10 Data and prepares data loaders
#CIFAT -10 32x32 Image size resized to 224x224 and rescale pixel values between -1 & 1 to train on Vision Transformer architecture by creating patch size of 16x16 pixels

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def get_loaders_cifar(dataset_type="CIFAR10", img_width=224, img_height=224, batch_size=16):

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((img_width, img_height), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
    ]) # Rescale train data between -1 & 1
    transform_test = transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
    ]) # Rescale test data between -1 & 1

    if dataset_type == "CIFAR10": #instantiates CIFAR-10 dataset
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    if dataset_type == "CIFAR100": #instantiates CIFAR-100 dataset
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)

    train_loader = DataLoader(trainset, sampler=train_sampler, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, sampler=test_sampler, batch_size=batch_size, num_workers=2, pin_memory=True)

    return train_loader, test_loader, testset

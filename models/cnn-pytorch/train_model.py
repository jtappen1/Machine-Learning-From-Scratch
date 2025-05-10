import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cpu')

model_transforms = transforms([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./cnn-pytorch/data",
    train=True,
    transform=model_transforms,
    download=True
)

validation_dataset = torchvision.datasets.CIFAR10(
    root="./cnn-pytorch/data",
    train=False,
    transform=model_transforms,
    download=True
)

train_loader =  
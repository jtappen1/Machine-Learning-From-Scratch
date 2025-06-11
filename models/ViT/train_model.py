import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from ViT import VisionTransformer
from tqdm import tqdm

def train_model(model: VisionTransformer, train_dataloader: DataLoader,  test_dataloader: DataLoader, epochs: int=15, lr: float=1e-4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        test_acc = evaluate(model, test_dataloader, device)
        print(f"Test Accuracy: {test_acc:.2f}%")


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

def main():
    device = torch.device('cpu')
    batch_size = 64

    model_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 mean
                         (0.247, 0.243, 0.261))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data/ViT/data",
        train=True,
        transform=model_transforms,
        download=True
    )

    validation_dataset = torchvision.datasets.CIFAR10(
        root="./data/ViT/data",
        train=False,
        transform=model_transforms,
        download=True
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    model = VisionTransformer()
    train_model(
        model=model, 
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        device=device
    )

if __name__ == "__main__":
    main()
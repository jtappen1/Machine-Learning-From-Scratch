import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from cnn import BasicCNN
from torch.utils.data.dataloader import DataLoader

def main():
    device = torch.device('cpu')

    batch_size = 64
    learning_rate= 0.001
    weight_decay=  0.005
    num_epochs = 10

    model_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data/cnn-pytorch/data",
        train=True,
        transform=model_transforms,
        download=True
    )

    validation_dataset = torchvision.datasets.CIFAR10(
        root="./data/cnn-pytorch/data",
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
        shuffle=True
    )
    # Another fancy name for a loss function
    criterion = nn.CrossEntropyLoss()

    model = BasicCNN(num_classes=10)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    print("Starting training")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move the tensors to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))        
    
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))

if __name__ == "__main__":
    main()

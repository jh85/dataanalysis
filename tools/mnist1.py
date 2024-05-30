import torch
import torch.nn as nn
import torchvision
import numpy as np
import os

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self,x):
        x = self.layer(x)
        return x

class Net2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer0 = nn.Flatten()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        return x

def accuracy(prediction, label):
    return torch.sum(torch.argmax(prediction, dim=1) == label) / len(label)

def main():
    mnist_location = "./MNIST"
    batch_size = 100
    num_workers = 0
    device = "cuda"

    train_dataset = torchvision.datasets.MNIST(
        root=mnist_location,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=mnist_location,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    input_size = 28*28
    hidden_size = 100
    output_size = 10
    model = Net(input_size, hidden_size, output_size)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    CHECKPOINT_FILE = "/tmp/checkpoint.pth"
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    num_epoch = 100
    for epoch in range(num_epoch):
        model.train()
        for img,label in train_data_loader:
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            prediction = model(img)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, CHECKPOINT_FILE)

        model.eval()
        acc = 0
        num = 0
        with torch.no_grad():
            for img,label in test_data_loader:
                img = img.to(device)
                label = label.to(device)
                prediction = model(img)
                loss = criterion(prediction,label)
                num += len(img)
                acc += accuracy(prediction,label) * len(img)
        print(f"test accuracy = {acc / num}")            
            
main()

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
import numpy as np
from random import randrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 20
batch_size = 4
learning_rate = 0.001

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
     transforms.RandomRotation(90),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip()
     ])


trainset = datasets.ImageFolder(root='W:/python/class_root/', transform=transform) 

trainloader = DataLoader(trainset, batch_size=32, shuffle=True) 

classes = ('discard','viable')

#Conv Net

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv3 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24*47*47, 2400)
        self.fc2 = nn.Linear(2400, 240)
        self.fc3 = nn.Linear(240, 60)
        self.fc4 = nn.Linear(60, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 24*47*47)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.3f}')



with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(2)]
    n_class_samples = [0 for i in range(2)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct/n_samples
    print(f'accuracy = {acc}')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'accuracy of {classes[i]} = {acc} %')





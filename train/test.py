# %%
import os
import shutil
import glob
import random
import pickle
from unicodedata import category
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torchvision.models as models
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

# %%
epoch = 90
batch_size = 64
model = models.resnet50(weights=True)
model.fc = nn.Linear(model.fc.in_features, 100)
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)
train_transform = transforms.Compose([
    transforms.Resize(128, interpolation=2),
    transforms.RandomCrop(112),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
# Validation settings
test_transform = transforms.Compose([
    transforms.Resize(128, interpolation=2),
    transforms.RandomCrop(112),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
train_dataset = CIFAR100("../CIFAR100_data", train=True, transform=train_transform, download=True)
test_dataset = CIFAR100("../CIFAR100_data", train=False, transform=test_transform)
print(len(train_dataset))
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
optimizer = optim.SGD(model.parameters(), lr=1e-1)
criterion = nn.CrossEntropyLoss().to(device)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
acc_list = [0]
for i in range(epoch):
    for x, t in tqdm((train_dataloader)):
        x, t = x.to(device), t.to(device)
        y = model(x)
        loss = criterion(y, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    correct_sum = 0
    for x, t in test_dataloader:
        x, t = x.to(device), t.to(device)
        y = model(x)
        correct_sum += (torch.argmax(y, dim=1) == t).sum()
    scheduler.step()
    acc_list.append(correct_sum/len(test_dataset))
    print(i, acc_list[-1])

# %%

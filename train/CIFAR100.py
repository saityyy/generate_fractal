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

normalize = transforms.Normalize(mean=[0.2, 0.2, 0.2], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.ToTensor(),
])


epoch = 20
batch_size = 256
model_name = "resnet50"
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
for i in os.scandir("./weight"):
    weight_path = i.path
    print(weight_path)
    train_dataset = CIFAR100("./CIFAR100_data", train=True, transform=transform, download=True)
    test_dataset = CIFAR100("./CIFAR100_data", train=False, transform=transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    print(len(train_dataset))
    model = models.resnet18(pretrained=False)
    weight = torch.load(weight_path)
    model.load_state_dict(weight)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss().to(device)
    acc_list = [0]
    print(device)
    for i in range(epoch):
        for x, t in (train_dataloader):
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
        acc_list.append(correct_sum/len(test_dataset))
        print(acc_list[-1])

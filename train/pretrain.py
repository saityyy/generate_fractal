import os
import shutil
import glob
import random
import pickle
import datetime
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

parser = argparse.ArgumentParser()
parser.add_argument('--db_path', type=str)
parser.add_argument('--weight_path', type=str)
args = parser.parse_args()
DB_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", args.db_path)
WEIGHT_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", args.weight_path)
normalize = transforms.Normalize(mean=[0.2, 0.2, 0.2], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.ToTensor(), normalize
])


class Model(nn.Module):
    def __init__(self, pretrained_model, category_num):
        super(Model, self).__init__()
        self.pretrained_model = pretrained_model
        self.layer1 = nn.Linear(self.pretrained_model.fc.out_features, category_num)

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.layer1(x)
        return x


epoch = 90
batch_size = 128
model_name = "resnet50"
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
for i in os.scandir(DB_PATH):
    db_path = i.path
    train_dataset = ImageFolder(db_path, transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    print(len(os.listdir(db_path)))
    if model_name == "resnet18":
        model = Model(models.resnet18(pretrained=False), len(os.listdir(db_path))).to(device)
    elif model_name == "resnet50":
        model = Model(models.resnet50(pretrained=False), len(os.listdir(db_path))).to(device)
    #weight = torch.load("weight/FractalDB+gaussian_resnet18_0.25.pth")
    # model.pretrained_model.load_state_dict(weight)
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)
    acc=0
    print(db_path)
    for it in range(epoch):
        for x, t in train_dataloader:
            x, t = x.to(device), t.to(device)
            y = model(x)
            loss = criterion(y, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct_sum = (torch.argmax(y, dim=1) == t).sum()
        scheduler.step()
        acc=correct_sum/batch_size
        print("epoch: {} accuracy: {} time :{}".format(it,acc,datetime.datetime.now()))
    torch.save(model.pretrained_model.cpu().state_dict(), os.path.join(WEIGHT_PATH,"{}_{}_{}.pth".format(i.name, model_name, acc)))

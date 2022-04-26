# %%
import os
import shutil
import glob
import random
import pickle
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
args = parser.parse_args()
DB_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", args.db_path)
normalize = transforms.Normalize(mean=[0.2, 0.2, 0.2], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.ToTensor(),
])


class Model(nn.Module):
    def __init__(self, pretrained_model):
        super(Model, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000, 1000),
        )
        self.pretrained_model = pretrained_model

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.fc(x)
        return x


epoch = 10
batch_size = 64
model_name = "resnet18"
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
train_dataset = ImageFolder(DB_PATH, transform)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
print(len(train_dataset))
if model_name == "resnet50":
    model = Model(models.resnet50(pretrained=False)).to(device)
elif model_name == "resnet18":
    model = Model(models.resnet18(pretrained=False)).to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().to(device)
acc_list = [0]
print(device)
for _ in range(epoch):
    for x, t in tqdm(train_dataloader):
        x, t = x.to(device), t.to(device)
        y = model(x)
        loss = criterion(y, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct_sum = (torch.argmax(y, dim=1) == t).sum()
        acc_list.append(correct_sum/batch_size)
        print(acc_list[-1])
    break
plt.plot(acc_list)
plt.show()

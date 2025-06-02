import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.depthwise(x))
        x = self.relu(self.pointwise(x))
        return x

class RescueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.b1 = DepthwiseSeparableConv(16, 32, stride=2)
        self.b2 = DepthwiseSeparableConv(32, 64, stride=2)
        self.b3 = DepthwiseSeparableConv(64, 128, stride=2)
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 5, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        out = self.head(x)
        return out


GRID_SIZE = 16
IMG_SIZE = 256
CELL_SIZE = IMG_SIZE // GRID_SIZE

def create_target(bboxes):
    target = torch.zeros(5, GRID_SIZE, GRID_SIZE)

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        xmin = xmin * (IMG_SIZE / 500)
        xmax = xmax * (IMG_SIZE / 500)
        ymin = ymin * (IMG_SIZE / 375)
        ymax = ymax * (IMG_SIZE / 375)

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin

        i = int(cy // CELL_SIZE)
        j = int(cx // CELL_SIZE)
        if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE:
            target[0, i, j] = 1
            target[1, i, j] = (cx / CELL_SIZE) - j
            target[2, i, j] = (cy / CELL_SIZE) - i
            target[3, i, j] = w / IMG_SIZE
            target[4, i, j] = h / IMG_SIZE

    return target


class VOCPeopleDataset(torch.utils.data.Dataset):
    def __init__(self, root, year='2007', image_set='train', transform=None):
        self.voc = VOCDetection(root=root, year=year, image_set=image_set, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, target = self.voc[idx]
        # Извлекаем bbox для класса "person"
        objs = target['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]

        person_bboxes = []
        for obj in objs:
            if obj['name'] == 'person':
                bbox = obj['bndbox']
                xmin = float(bbox['xmin'])
                ymin = float(bbox['ymin'])
                xmax = float(bbox['xmax'])
                ymax = float(bbox['ymax'])
                person_bboxes.append([xmin, ymin, xmax, ymax])

        if self.transform:
            img = self.transform(img)

        target_tensor = create_target(person_bboxes)

        return img, target_tensor

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

dataset = VOCPeopleDataset(root='./data', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

def custom_loss(pred, target):
    obj_loss = F.binary_cross_entropy_with_logits(pred[:,0], target[:,0])
    dx_dy_loss = F.mse_loss(torch.tanh(pred[:,1:3]), target[:,1:3])
    size_loss = F.mse_loss(F.relu(pred[:,3:5]), target[:,3:5])
    return obj_loss + dx_dy_loss + size_loss

model = RescueNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5
max_batches = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(f"Epoch {epoch+1} started")

    for batch_idx, (imgs, targets) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = custom_loss(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Batch {batch_idx+1}/{max_batches} - Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} finished, total loss on {max_batches} batches: {total_loss:.4f}")

torch.save(model.state_dict(), "rescuenet_voc.pth")
print("Модель сохранена в rescuenet_voc.pth")

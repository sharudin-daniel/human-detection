import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCDetection
from tqdm import tqdm

from main import CustomDetector

class CustomModelTrainer:
    def __init__(self,
                 data_dir,
                 model_save_path="custom_detector.pth",
                 batch_size=16,
                 lr=1e-3,
                 num_epochs=10,
                 image_size=(256, 256),
                 device=None):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.model_save_path = model_save_path
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CustomDetector().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def load_dataset(self):
        train_dataset = VOCDetection(root=self.data_dir, year='2007', image_set='train', download=True,
                                     transform=self.transforms)
        val_dataset = VOCDetection(root=self.data_dir, year='2007', image_set='val', download=True,
                                   transform=self.transforms)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

    def train(self):
        print(f"Starting training on device: {self.device}")
        self.model.train()

        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}"):
                images, _ = batch
                images = images.to(self.device)

                # Заглушка: создаем искусственные "таргеты" того же размера, что и вывод
                targets = torch.zeros((images.size(0), 5, 16, 16)).to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch} Loss: {epoch_loss:.4f}")
            self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        self.model.to(self.device).eval()
        print(f"Model loaded from {self.model_save_path}")

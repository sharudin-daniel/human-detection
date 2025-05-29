import torch
import torch.nn as nn
import torch.nn.functional as F

# Depthwise separable convolution блок
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(x)
        return x

# Backbone + Head модель
class HumanDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1: обычная свертка
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        # Backbone blocks
        self.b1 = DepthwiseSeparableConv(16, 32)   # 128x128 -> 64x64
        self.b2 = DepthwiseSeparableConv(32, 64)   # 64x64 -> 32x32
        self.b3 = DepthwiseSeparableConv(64, 128)  # 32x32 -> 16x16
        # Опциональный блок B4 (не используется, т.к. выбран выход после B3)
        self.b4 = DepthwiseSeparableConv(128, 128) # 16x16 -> 8x8 (опционально)

        # Head
        self.head_h1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.head_h2 = nn.Conv2d(64, 5, kernel_size=1, stride=1, padding=0)  # output 5x16x16

    def forward(self, x):
        # Backbone
        x = self.conv1(x)  # 16x128x128
        x = self.b1(x)     # 32x64x64
        x = self.b2(x)     # 64x32x32
        x = self.b3(x)     # 128x16x16

        # Head
        x = self.head_h1(x)  # 64x16x16
        x = self.head_h2(x)  # 5x16x16

        # Активируем выходные каналы по смыслу:
        # objectness score — sigmoid
        # dx, dy — tanh
        # w, h — exp (для положительности)
        objectness = torch.sigmoid(x[:, 0:1, :, :])
        dx = torch.tanh(x[:, 1:2, :, :])
        dy = torch.tanh(x[:, 2:3, :, :])
        w = torch.exp(x[:, 3:4, :, :])
        h = torch.exp(x[:, 4:5, :, :])

        out = torch.cat([objectness, dx, dy, w, h], dim=1)
        return out


# Функция предобработки (нормализация + стандартизация по ImageNet)
from torchvision import transforms

def preprocess_image(image):
    """
    image: PIL.Image или numpy array RGB, размер может быть любым
    Возвращает tensor [3,256,256] с нормализацией и стандартизацией
    """
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # перевод в [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # стандартизация ImageNet
                             std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)


if __name__ == "__main__":
    # Проверка модели
    model = HumanDetectionModel()
    x = torch.randn(1, 3, 256, 256)  # пример батча
    y = model(x)
    print(y.shape)  # должно быть [1, 5, 16, 16]

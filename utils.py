
import os, json, time, cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path

class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pointwise = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)

class CustomDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True)
        )
        self.b1 = DepthwiseSeparableConv(16, 32, stride=2)
        self.b2 = DepthwiseSeparableConv(32, 64, stride=2)
        self.b3 = DepthwiseSeparableConv(64, 128, stride=2)
        self.head_h1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True)
        )
        self.head_h2 = torch.nn.Conv2d(64, 5, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.head_h1(x)
        x = self.head_h2(x)
        return x

class Preprocessor:
    def __init__(self):
        self.pipeline = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        return self.pipeline(img)

class VideoFrameExtractor:
    def __init__(self, path, every_nth=10):
        self.cap = cv2.VideoCapture(path)
        self.every_nth = every_nth
        self.frame_id = 0

    def __iter__(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            self.frame_id += 1
            if self.frame_id % self.every_nth == 0:
                yield self.frame_id, frame
        self.cap.release()

class PredictionDecoder:
    def __init__(self, threshold=0.5, grid_size=16, input_size=256):
        self.th = threshold
        self.grid = grid_size
        self.inp = input_size
        self.cell = input_size / grid_size

    def decode(self, output):
        output = output.squeeze(0).cpu()
        obj = torch.sigmoid(output[0])
        dx = torch.tanh(output[1])
        dy = torch.tanh(output[2])
        w = F.relu(output[3])
        h = F.relu(output[4])
        indices = (obj > self.th).nonzero(as_tuple=False)

        boxes = []
        for i, j in indices:
            cx = (j + 0.5 + dx[i, j].item()) * self.cell
            cy = (i + 0.5 + dy[i, j].item()) * self.cell
            bw = w[i, j].item() * self.cell
            bh = h[i, j].item() * self.cell
            boxes.append([cx, cy, bw, bh])
        return boxes

class BoundingBoxDrawer:
    def __init__(self, input_size=256):
        self.input_size = input_size

    def draw(self, frame, boxes):
        h, w = frame.shape[:2]
        sx, sy = w / self.input_size, h / self.input_size
        for cx, cy, bw, bh in boxes:
            x1, y1 = int((cx - bw / 2) * sx), int((cy - bh / 2) * sy)
            x2, y2 = int((cx + bw / 2) * sx), int((cy + bh / 2) * sy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame, len(boxes)

class DetectionLogger:
    def __init__(self):
        self.results = []

    def log(self, frame_id, count, path=None):
        entry = {"frame": frame_id, "people_detected": count}
        if path:
            entry["image_path"] = path
        self.results.append(entry)
        return entry

class ImageSaver:
    def __init__(self, folder="detected_frames"):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def save(self, frame, frame_id):
        filename = f"frame_{frame_id}.jpg"
        path = self.folder / filename
        cv2.imwrite(str(path), frame)
        return f"/images/{filename}"

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import torch
import json
import time
import os
from PIL import Image
from torchvision import transforms

app = FastAPI()

# Кастомная модель (оставлена, но не используется)
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

# Используем YOLOv5 вместо CustomDetector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
model.to(device).eval()

# Препроцессинг (оставлен, но не используется с YOLOv5 напрямую)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

os.makedirs("detected_frames", exist_ok=True)

app.mount("/images", StaticFiles(directory="detected_frames"), name="images")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def start():
    with open("static/index.html") as f:
        return f.read()

@app.get("/start", response_class=HTMLResponse)
async def root():
    with open("static/start.html") as f:
        return f.read()

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        video_data = await file.read()
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_data)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        results = []

        def generate():
            nonlocal frame_count
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 10 != 0:
                    continue

                # YOLO expects RGB
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Inference
                results_yolo = model(img_rgb)

                # Filter: только "person" (class 0)
                detections = results_yolo.xyxy[0].cpu().numpy()
                people_detected = 0
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    if int(cls) == 0:  # class 0 == person
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        people_detected += 1

                entry = {"frame": frame_count, "people_detected": people_detected}
                if people_detected > 0:
                    filename = f"frame_{frame_count}.jpg"
                    filepath = f"detected_frames/{filename}"
                    cv2.imwrite(filepath, frame)
                    entry["image_path"] = f"/images/{filename}"

                results.append(entry)

                progress = int((frame_count / total_frames) * 100)
                yield f"data:{json.dumps({'progress': progress, 'latest': entry})}\n\n"
                time.sleep(0.05)

            cap.release()
            yield f"data:{json.dumps({'done': True, 'results': results})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return {"error": str(e)}

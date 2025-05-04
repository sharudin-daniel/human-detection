from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import torch
import numpy as np
from PIL import Image
import os
import json
import io
import time

app = FastAPI()

# Загружаем YOLOv5s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Папка для результатов
os.makedirs("detected_frames", exist_ok=True)

app.mount("/images", StaticFiles(directory="detected_frames"), name="images")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        video_data = await file.read()

        # Сохраняем видео во временный файл
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

                # Детекция
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                detection_results = model(image)
                detections = detection_results.xyxy[0]

                people_detected = 0
                for *xyxy, conf, cls in detections:
                    if model.names[int(cls)] == 'person' and conf > 0.5:
                        people_detected += 1
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                entry = {"frame": frame_count, "people_detected": people_detected}
                if people_detected > 0:
                    filename = f"frame_{frame_count}.jpg"
                    filepath = f"detected_frames/{filename}"
                    cv2.imwrite(filepath, frame)
                    entry["image_path"] = f"/images/{filename}"

                results.append(entry)

                # Прогресс-обновление (в формате NDJSON — newline-delimited JSON)
                progress = int((frame_count / total_frames) * 100)
                yield f"data:{json.dumps({'progress': progress, 'latest': entry})}\n\n"
                time.sleep(0.05)  # для эффекта

            cap.release()
            yield f"data:{json.dumps({'done': True, 'results': results})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return {"error": str(e)}

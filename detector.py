import torch
import cv2
import os
import tempfile

# Загрузка модели YOLOv5s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_people_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_results = []
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Каждые 10 кадров (для ускорения)
        if frame_index % 10 == 0:
            results = model(frame)
            detections = results.pandas().xyxy[0]
            people = detections[detections['name'] == 'person']
            frame_results.append({
                "frame": frame_index,
                "people_detected": len(people)
            })

        frame_index += 1

    cap.release()
    return frame_results

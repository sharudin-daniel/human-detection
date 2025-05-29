from ultralytics import YOLO
import cv2

# Загрузка самой лёгкой модели YOLOv8n
model = YOLO('yolov8n.pt')

def detect_people_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_results = []
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % 10 == 0:
            results = model(frame)
            people_count = 0
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if result.names[int(box.cls)] == 'person':
                        people_count += 1
            frame_results.append({
                "frame": frame_index,
                "people_detected": people_count
            })

        frame_index += 1

    cap.release()
    return frame_results

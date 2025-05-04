from fastapi import FastAPI, File, UploadFile
import cv2
import torch
import numpy as np
from io import BytesIO
from PIL import Image
import base64
import os

app = FastAPI()

# Загружаем модель YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def encode_image(image: np.array) -> str:
    """Преобразует изображение в формат base64 для отправки через API"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def save_image(image: np.array, frame_number: int) -> None:
    """Сохраняет изображение на диск"""
    output_filename = f"frame_{frame_number}_detected_people.jpg"
    cv2.imwrite(output_filename, image)
    print(f"Image saved as {output_filename}")

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        # Чтение видеофайла
        video_data = await file.read()
        
        # Создаем временный файл из байтов
        video_path = "/tmp/temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_data)
        
        # Открываем видео с помощью OpenCV
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        results = []
        
        # Чтение видео по кадрам
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Обрабатываем только каждый 10-й кадр
            if frame_count % 200 != 0:
                continue  # Пропускаем кадры, которые не являются 10-м
            
            # Преобразуем кадр в изображение PIL
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Прогоняем кадр через модель YOLOv5
            detection_results = model(image)
            
            # Получаем результаты детекции
            detections = detection_results.xywh[0]
            
            people_detected = 0

            # Для каждого объекта в кадре проверяем, является ли он человеком
            for *xywh, conf, cls in detections:
                if model.names[int(cls)] == 'person' and conf > 0.5:
                    people_detected += 1
                    
                    # Рисуем bounding box для каждого человека
                    x1, y1, x2, y2 = [int(coord) for coord in xywh]  # Преобразуем координаты в целые числа
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Если люди найдены, сохраняем кадр
            if people_detected > 0:
                encoded_image = encode_image(frame)
                
                # Сохраняем изображение на диск
                save_image(frame, frame_count)

                results.append({
                    "frame": frame_count,
                    "people_detected": people_detected
                    # "image": encoded_image  # Добавляем картинку в формате base64
                })
            else:
                results.append({
                    "frame": frame_count,
                    "people_detected": people_detected
                })
        
        cap.release()
        
        return {"results": results}

    except Exception as e:
        return {"error": str(e)}

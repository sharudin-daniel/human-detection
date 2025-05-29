# Stage 1: builder
FROM python:3.9-slim AS builder

WORKDIR /app

# Установка системных библиотек (минимально необходимый набор)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем проект и зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Stage 2: финальный образ
FROM python:3.9-slim

WORKDIR /app

# Повторно устанавливаем необходимые системные библиотеки
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем только нужное из builder (уже установленное)
COPY --from=builder /usr/local /usr/local

# Копируем остальной код проекта
COPY . .

# Открываем порт для FastAPI
EXPOSE 8000

# Запуск FastAPI через uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

# Системные зависимости
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости
COPY requirements_anpr.txt .

# Ставим в правильном порядке чтобы избежать конфликтов
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
    fastapi==0.135.2 \
    uvicorn==0.42.0 \
    python-multipart==0.0.22 \
    ultralytics==8.4.27 \
    paddlepaddle==2.6.2 \
    paddleocr==2.7.3 \
    easyocr==1.7.2 \
    pytesseract==0.3.13 \
    opencv-python-headless==4.13.0.92 \
    pydantic==2.12.5


COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
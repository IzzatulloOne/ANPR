FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /train

# Ставим обучающий стек
RUN pip install --no-cache-dir numpy==1.24.4
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
    lmdb==2.1.1 \
    opencv-python==4.11.0.86 \
    pillow==12.1.1 \
    natsort==8.4.0 \
    nltk==3.9.4 \
    tqdm==4.67.3 \
    fire==0.7.1

# Клонируем deep-text-recognition-benchmark
RUN git clone https://github.com/clovaai/deep-text-recognition-benchmark .

# Датасет монтируется снаружи
VOLUME ["/train/dataset"]
VOLUME ["/train/saved_models"]
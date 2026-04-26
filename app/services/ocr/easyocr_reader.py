import cv2
import easyocr
import re

# Инициализируем один раз — загрузка модели тяжёлая
READER = easyocr.Reader(
    ["en"],
    gpu=False,          # True если есть CUDA
    verbose=False,
)

def read_easyocr(image):
    if image is None:
        return None, 0.0

    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # allowlist — только символы номера
    results = READER.readtext(
        rgb,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        detail=1,
        paragraph=False,
    )

    if not results:
        return None, 0.0

    # Берём результат с наибольшим confidence
    best_text, best_conf = None, 0.0
    for (_, text, conf) in results:
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        if text and conf > best_conf:
            best_text = text
            best_conf = conf

    return best_text, float(best_conf)
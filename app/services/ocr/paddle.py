import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=False,  # номера всегда горизонтальные
    lang="en",
    show_log=False,
    det=False,           # детекцию уже сделал YOLO, не нужна повторная
    rec=True,
)

def read_paddle(image):
    if image is None:
        return None, 0.0

    # Конвертируем в RGB
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Апскейл если изображение маленькое (paddle любит минимум ~100px высоты)
    h, w = rgb.shape[:2]
    if h < 64:
        scale = 64 / h
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    result = ocr.ocr(rgb, cls=False)
    if not result or result[0] is None:
        return None, 0.0

    best_text, best_conf = None, 0.0
    for line in result[0]:
        text = line[1][0]
        conf = float(line[1][1])
        if conf > best_conf and text:
            best_conf = conf
            best_text = text

    if not best_text:
        return None, 0.0

    return best_text, best_conf
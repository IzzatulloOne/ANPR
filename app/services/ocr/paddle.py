import cv2
import re
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="en")

# Мусор который не является частью номера
NOISE_WORDS = {"UZ", "UZB", "RUS", "KAZ", "KG", "TJ"}


def read_paddle(image):
    if image is None:
        return None, 0.0

    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    if h < 64:
        scale = 64 / h
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    result = ocr.ocr(rgb, cls=False)
    if not result or result[0] is None:
        return None, 0.0

    # Собираем все блоки с координатами
    blocks = []
    for line in result[0]:
        bbox  = line[0]
        text  = re.sub(r"[^A-Z0-9]", "", line[1][0].upper())
        conf  = float(line[1][1])
        x_center = (bbox[0][0] + bbox[2][0]) / 2

        # Фильтруем мусор
        if text in NOISE_WORDS:
            continue
        if len(text) < 2:
            continue

        blocks.append((x_center, text, conf))

    if not blocks:
        return None, 0.0

    # Сортируем слева направо по x координате
    blocks.sort(key=lambda b: b[0])

    # Склеиваем все блоки
    full_text = "".join(b[1] for b in blocks)
    avg_conf  = sum(b[2] for b in blocks) / len(blocks)

    return full_text, avg_conf
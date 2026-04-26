import os
import cv2
import re
import pytesseract

TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Разные psm для ensemble
PSM_CONFIGS = [
    "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
]

def _run_tesseract(gray, config):
    text = pytesseract.image_to_string(gray, config=config)
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not text:
        return None, 0.0

    data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
    confs = []
    for c in data.get("conf", []):
        try:
            c = float(c)
            if c >= 0:
                confs.append(c)
        except Exception:
            pass
    avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    return text, avg_conf


def read_tesseract(image):
    if image is None:
        return None, 0.0

    # Только grayscale — НЕ бинаризуем, НЕ апскейлим (variants уже сделал)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Лёгкий denoise без потери тонких символов
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    best_text, best_conf = None, 0.0

    for config in PSM_CONFIGS:
        text, conf = _run_tesseract(gray, config)
        if text and conf > best_conf:
            best_text = text
            best_conf = conf

    return best_text, best_conf
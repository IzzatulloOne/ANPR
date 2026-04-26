from pathlib import Path
import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[3]
PLATE_MODEL_PATH = BASE_DIR / "models" / "license_plate_detector.pt"
PLATE_MODEL = YOLO(str(PLATE_MODEL_PATH))

def detect_plate(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    results = PLATE_MODEL(img)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None, None

    best_i = int(results.boxes.conf.argmax().item())
    box = results.boxes.xyxy[best_i].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box.tolist()

    h, w = img.shape[:2]

    # В кость — минимальный padding только чтобы не обрезать символы
    pad_x = int((x2 - x1) * 0.02)
    pad_y = int((y2 - y1) * 0.02)
    x1 = max(0, x1 - pad_x)
    x2 = min(w, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None, None

    plate_crop = img[y1:y2, x1:x2].copy()
    pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return plate_crop, pts
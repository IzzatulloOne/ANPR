from pathlib import Path
import cv2
from ultralytics import YOLO

# === CONFIG ===
INPUT_DIR = Path("dataset/uz")          # где твои фотки
OUTPUT_DIR = Path("dataset/crops")          # куда сохранять кропы
MODEL_PATH = Path("models/license_plate_detector.pt")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))


def detect_and_crop(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] Не читается: {image_path}")
        return None

    results = model(img)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return None

    # берем самый уверенный bbox
    best_i = int(results.boxes.conf.argmax().item())
    box = results.boxes.xyxy[best_i].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box.tolist()

    h, w = img.shape[:2]

    # маленький паддинг
    pad_x = int((x2 - x1) * 0.02)
    pad_y = int((y2 - y1) * 0.02)

    x1 = max(0, x1 - pad_x)
    x2 = min(w, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2].copy()
    return crop


def process_folder():
    images = list(INPUT_DIR.glob("*.*"))

    if not images:
        print("Папка пустая. Поздравляю, ты кропишь воздух.")
        return

    for img_path in images:
        crop = detect_and_crop(img_path)

        if crop is None:
            print(f"[SKIP] Нет номера: {img_path.name}")
            continue

        save_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(save_path), crop)

        print(f"[OK] {img_path.name}")


if __name__ == "__main__":
    process_folder()
import cv2
import tempfile
import logging
import uuid
from pathlib import Path

from app.services.detector.yolo_detector import detect_plate
from app.services.preprocess.variants import generate_variants
from app.services.ocr.ensemble import read_plate_ensemble
from app.services.postprocess.plate_rules import normalize_plate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEBUG_DIR = Path("dataset/collected")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


async def run_anpr(upload_file):
    temp_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await upload_file.read())
        temp_path = Path(tmp.name)

    try:
        plate, pts = detect_plate(temp_path)
        if plate is None:
            logger.info("No plate detected")
            return {"plate": None}
        
        cv2.imwrite("debug_plate.jpg", plate)

        logger.info(f"Plate detected: shape={plate.shape}")
        cv2.imwrite("debug_plate.jpg", plate)

        variants = generate_variants(plate)
        if not variants:
            logger.info("No preprocess variants generated")
            return {"plate": None}

        raw_text = read_plate_ensemble(variants)
        logger.info(f"OCR raw text: {raw_text!r}")

        final_text = normalize_plate(raw_text)
        logger.info(f"Normalized plate: {final_text!r}")

        # Сохраняем ПОСЛЕ того как final_text вычислен
        if final_text:
            img_path = DEBUG_DIR / f"{final_text}_{uuid.uuid4().hex[:6]}.jpg"
            cv2.imwrite(str(img_path), plate)
            logger.info(f"Saved training sample: {img_path}")

        return {"plate": final_text}

    finally:
        try:
            if temp_path:
                temp_path.unlink(missing_ok=True)
        except Exception:
            pass
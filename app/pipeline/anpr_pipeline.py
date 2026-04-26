import cv2
import tempfile
import uuid
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

from app.services.detector.yolo_detector import detect_plate
from app.services.postprocess.plate_rules import normalize_plate
from app.services.ocr.paddle import read_paddle

DEBUG_DIR = Path("dataset/collected")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    
async def run_anpr(upload_file):
    temp_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await upload_file.read())
        temp_path = Path(tmp.name)

    try:
        plate, pts = detect_plate(temp_path)
        # после detect_plate
        cv2.imwrite("debug_plate.jpg", plate)
        logger.info(f"Debug plate saved: {plate.shape}")
        if plate is None:
            return {"plate": None}

        # Только апскейл — никаких вариантов
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        up   = cv2.resize(gray, None, fx=2, fy=2,
                          interpolation=cv2.INTER_CUBIC)

        raw_text, conf = read_paddle(up)
        if not raw_text:
            return {"plate": None}

        final_text = normalize_plate(raw_text)

        if final_text:
            fname = DEBUG_DIR / f"{final_text}_{uuid.uuid4().hex[:6]}.jpg"
            cv2.imwrite(str(fname), plate)

        return {"plate": final_text, "conf": round(conf, 2)}

    finally:
        try:
            if temp_path:
                temp_path.unlink(missing_ok=True)
        except Exception:
            pass
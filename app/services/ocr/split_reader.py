import cv2
import re
import logging
import easyocr

logger = logging.getLogger(__name__)
READER = easyocr.Reader(["en"], gpu=False, verbose=False)


def read_plate_split(image):
    if image is None:
        return None, 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape[:2]

    left     = gray[:, :int(w * 0.20)]
    right    = gray[:, int(w * 0.20):]
    left_inv = cv2.bitwise_not(left)

    left_up  = cv2.resize(left_inv, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    right_up = cv2.resize(right,    None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    def ocr_part(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        results = READER.readtext(
            rgb,
            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            detail=1,
            paragraph=False,
        )
        if not results:
            return "", 0.0
        best_text, best_conf = "", 0.0
        for (_, text, conf) in results:
            text = re.sub(r"[^A-Z0-9]", "", text.upper())
            if text and conf > best_conf:
                best_text = text
                best_conf = conf
        return best_text, best_conf

    left_text,  left_conf  = ocr_part(left_up)
    right_text, right_conf = ocr_part(right_up)

    logger.info(f"split: left={left_text!r}({left_conf:.2f}) right={right_text!r}({right_conf:.2f})")

    if not right_text:
        return None, 0.0

    full_text = left_text + right_text
    avg_conf  = (left_conf + right_conf) / 2 if left_text else right_conf
    return full_text, avg_conf
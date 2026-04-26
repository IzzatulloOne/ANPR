# app/services/preprocess/perspective.py
import cv2
import numpy as np


def correct_perspective(img):
    """
    Пытается исправить перспективное искажение номерного знака.
    Если не удаётся — возвращает оригинал.
    """
    if img is None:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # Размываем и ищем края
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    # Морфология чтобы замкнуть контуры
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    # Берём самый большой контур
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    h, w = img.shape[:2]

    if len(approx) == 4:
        # Нашли 4 угла — делаем warp
        pts = approx.reshape(4, 2).astype(np.float32)
        pts = _order_points(pts)

        # Целевой размер
        target_w = 400
        target_h = 120

        dst = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (target_w, target_h))
        return warped

    else:
        # Не нашли 4 угла — пробуем shear correction по горизонтали
        return _correct_shear(img)


def _order_points(pts):
    """Сортирует точки: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _correct_shear(img):
    """
    Простая коррекция горизонтального сдвига через моменты.
    Помогает когда номер снят под углом но контур не найден.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    moments = cv2.moments(thresh)
    if abs(moments["mu02"]) < 1e-2:
        return img

    skew = moments["mu11"] / moments["mu02"]
    h, w = img.shape[:2]

    # Ограничиваем коррекцию разумным диапазоном
    if abs(skew) > 2:
        return img

    M = np.float32([[1, skew, -0.5 * h * skew], [0, 1, 0]])
    corrected = cv2.warpAffine(img, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return corrected
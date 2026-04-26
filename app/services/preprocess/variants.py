import cv2
import numpy as np


def generate_variants(img):
    variants = []
    if img is None:
        return variants

    variants.append(img)  # оригинал всегда первый

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Апскейл x2 — самый полезный вариант для мелких символов
    up2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants.append(up2)

    # Апскейл x3 — для очень маленьких crop
    up3 = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    variants.append(up3)

    # CLAHE на апскейленном — лучше чем на оригинале
    
# Разделяем номер на две части — синий блок слева и белый блок справа
    h_img, w_img = gray.shape[:2]
    left_part = gray[:, :int(w_img * 0.25)]   # левые 25% — синий блок
    right_part = gray[:, int(w_img * 0.25):]  # правые 75% — белый блок

    # Инвертируем только левую часть и склеиваем
    left_inverted = cv2.bitwise_not(left_part)
    merged = np.hstack([left_inverted, right_part])
    merged_up = cv2.resize(merged, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants.append(merged_up)

# Усиление контраста по всему изображению через CLAHE x3
    clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    clahe_strong_img = clahe_strong.apply(gray)
    variants.append(clahe_strong_img)

    # Инвертированный — иногда тёмные символы на светлом фоне читаются лучше наоборот
    inverted = cv2.bitwise_not(gray)
    variants.append(inverted)

    # Gamma correction — осветляет тёмные области
    gamma = 1.8
    look_up = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255)) 
                        for i in range(256)], dtype=np.uint8)
    gamma_img = cv2.LUT(up2, look_up)
    variants.append(gamma_img)

    # Adaptive threshold на апскейле
    adapt = cv2.adaptiveThreshold(
        up2, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )
    variants.append(adapt)

    return variants
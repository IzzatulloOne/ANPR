from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  
def read_plate_ensemble(versions):
    texts = []
    for img in versions:
        result = ocr.ocr(img, cls=True)
        if result and result[0]:
            text = " ".join([line[1][0] for line in result[0]])
            texts.append(text)
    return " ".join(texts)
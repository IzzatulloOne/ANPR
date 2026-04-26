from collections import Counter
from app.services.ocr.paddle import read_paddle
from app.services.ocr.tesseract import read_tesseract
from app.services.ocr.easyocr_reader import read_easyocr
from app.services.ocr.split_reader import read_plate_split
from app.services.postprocess.plate_rules import normalize_plate, plate_score
import logging
import re

logger = logging.getLogger(__name__)

PATTERN = re.compile(r"^\d{2}[A-Z]\d{3}[A-Z]{2}$")


def _char_vote(texts):
    texts = [t for t in texts if t]
    if not texts:
        return None
    by_len = {}
    for t in texts:
        by_len.setdefault(len(t), []).append(t)
    group = by_len.get(8) or max(by_len.values(), key=len)
    if len(group) == 1:
        return group[0]
    result = []
    for i in range(len(group[0])):
        chars = [t[i] for t in group if i < len(t)]
        result.append(Counter(chars).most_common(1)[0][0])
    return "".join(result)


def read_plate_ensemble(images):
    candidates = []

    for img in images:
        p_text, p_conf = read_paddle(img)
        t_text, t_conf = read_tesseract(img)
        e_text, e_conf = read_easyocr(img)

        try:
            s_text, s_conf = read_plate_split(img)
            logger.info(f"  [split] {s_text!r} conf={s_conf:.2f}")
        except Exception as ex:
            logger.error(f"  [split] error: {ex}")
            s_text, s_conf = None, 0.0

        for text, conf, source in [
            (p_text, p_conf, "paddle"),
            (t_text, t_conf, "tesseract"),
            (e_text, e_conf, "easyocr"),
            (s_text, s_conf, "split"),
        ]:
            if text:
                norm = normalize_plate(text)
                candidates.append({
                    "text": text,
                    "norm": norm,
                    "source": source,
                    "conf": float(conf),
                })
                logger.info(f"  [{source}] {text!r} → {norm!r} conf={conf:.2f}")

    if not candidates:
        return None

    norms_8 = [c["norm"] for c in candidates if c["norm"] and len(c["norm"]) == 8]
    if norms_8:
        voted_norm = _char_vote(norms_8)
        if voted_norm:
            candidates.append({
                "text": voted_norm,
                "norm": normalize_plate(voted_norm),
                "source": "vote_norm",
                "conf": 0.80,
            })

    def final_score(c):
        norm = c["norm"] or c["text"] or ""
        score = c["conf"] * 2.0
        score += plate_score(norm)
        if len(norm) == 8:
            score += 1.0
        if PATTERN.match(norm):
            score += 2.0
        return score

    best = max(candidates, key=final_score)
    logger.info(f"  BEST: {best['norm'] or best['text']!r} from {best['source']}")
    return best["norm"] or best["text"]
import re
import logging

logger = logging.getLogger(__name__)

PLATE_PATTERN = re.compile(r"^\d{2}[A-Z]\d{3}[A-Z]{2}$")

DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}
LETTER_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5",
                   "B": "8", "G": "6", "Q": "0"}

digit_positions = {0, 1, 3, 4, 5}
letter_positions = {2, 6, 7}


UZB_REGION_LETTERS = {
    "01": "A", "02": "A",
    "10": "B", "11": "B", "12": "B",
    "20": "D", "21": "D", "22": "D",
    "25": "E", "26": "E",
    "30": "F", "31": "F",
    "35": "G", "36": "G",
    "40": "H", "41": "H", "42": "H",
    "50": "J", "51": "J", "52": "J",
    "55": "K", "56": "K",
    "60": "M", "61": "M", "62": "M",
    "65": "N", "66": "N",
    "70": "O", "71": "O", "72": "O",
    "75": "P", "76": "P",
    "80": "R", "81": "R", "82": "R",
    "85": "S", "86": "S",
    "90": "T", "91": "T", "92": "T",
    "95": "U", "96": "U",
}

# OCR часто путает 0↔O, 1↔I — добавь алиасы
REGION_ALIASES = {
    "O0": "01", "O1": "01", "OO": "00",
    "I0": "10", "0I": "01", "II": "11",
    "00": "01",  # 00 скорее всего 01 — OCR дропнул засечку у 1
}

def _apply_position_rules(chars: list) -> list:
    chars = chars[:]
    for i in digit_positions:
        if i < len(chars) and chars[i] in LETTER_TO_DIGIT:
            chars[i] = LETTER_TO_DIGIT[chars[i]]
    for i in letter_positions:
        if i < len(chars) and chars[i] in DIGIT_TO_LETTER:
            chars[i] = DIGIT_TO_LETTER[chars[i]]
    return chars


def _detect_shift(chars: list) -> bool:
    """pos 0-4 цифры, pos 5-6 буквы → буква на pos 2 была дропнута."""
    if len(chars) < 7:
        return False
    digit_like = set("0123456789OIZBSGQ")
    return (
        chars[0] in digit_like and
        chars[1] in digit_like and
        chars[2] in digit_like and 
        chars[3] in digit_like and
        chars[4] in digit_like and
        chars[5].isalpha() and
        chars[6].isalpha()
    )


def _recover_shifted(chars: list) -> list | None:
    if not _detect_shift(chars):
        return None

    region_code = chars[0] + chars[1]

    # Пробуем алиас если точного совпадения нет
    if region_code not in UZB_REGION_LETTERS:
        region_code = REGION_ALIASES.get(region_code, region_code)

    letter = UZB_REGION_LETTERS.get(region_code)

    if letter:
        recovered = chars[:2] + [letter] + chars[2:7]
        logger.info(f"Recovered: {''.join(chars)!r} → {''.join(recovered)!r} (region {region_code}→{letter})")
        return recovered

    logger.warning(f"Unknown region code: {region_code!r}, cannot recover letter at pos 2")
    return None

def normalize_plate(text: str) -> str | None:
    if not text:
        return None

    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    chars = list(text)

    # 9+ символов — обрезаем мусор справа
    if len(chars) >= 9:
        chars = chars[:8]

    # Пробуем восстановить дропнутую букву на pos 2
    recovered = _recover_shifted(chars)
    if recovered:
        chars = recovered

    if len(chars) < 8:
        return "".join(_apply_position_rules(chars))

    chars = chars[:8]
    chars = _apply_position_rules(chars)
    result = "".join(chars)

    if not PLATE_PATTERN.match(result):
        logger.warning(f"Pattern mismatch after normalize: {result!r}")

    return result


def plate_score(text: str) -> float:
    if not text:
        return 0.0
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    score = 0.0
    if len(text) == 8:
        score += 1.0
    if PLATE_PATTERN.match(text):
        score += 3.0
    if sum(c.isdigit() for c in text) >= 5:
        score += 0.5
    if any(c.isalpha() for c in text):
        score += 0.5
    return score

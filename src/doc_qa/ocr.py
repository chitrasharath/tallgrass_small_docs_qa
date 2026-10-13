from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pytesseract
from PIL import Image, ImageFilter


# Tesseract reads a capital I as a lowercase l in short words ("lf", "lt").
_L_FOR_I = re.compile(r"\bl([fnst])\b")
_SENTENCE_OPEN = re.compile(r"[.!?:)\"']$")


def fix_ocr_text(text: str) -> str:
    return _L_FOR_I.sub(r"I\1", text)


def merge_broken_paragraphs(paragraphs: list[str]) -> list[str]:
    """Rejoin paragraphs that OCR split in the middle of a sentence."""
    merged: list[str] = []
    for paragraph in paragraphs:
        if (
            merged
            and "\n" not in merged[-1]
            and not _SENTENCE_OPEN.search(merged[-1])
            and paragraph[:1].islower()
        ):
            merged[-1] += " " + paragraph
        else:
            merged.append(paragraph)
    return merged


@dataclass
class OcrResult:
    paragraphs: list[list[str]]  # each paragraph is a list of lines
    confidence: float  # mean word confidence, 0-100


def _otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    cumulative = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256))
    global_mean = cumulative_mean[-1]
    background = cumulative[:-1]
    foreground = total - background
    valid = (background > 0) & (foreground > 0)
    between = np.zeros(255)
    between[valid] = (
        (global_mean * background[valid] / total - cumulative_mean[:-1][valid]) ** 2
        / (background[valid] * foreground[valid])
    )
    return int(np.argmax(between))


def _skew_angle(gray: Image.Image, max_angle: float = 5.0, step: float = 0.25) -> float:
    """Angle that makes text rows most horizontal (sharpest row-sum profile)."""
    small = gray.resize((max(gray.width // 4, 1), max(gray.height // 4, 1)))
    array = np.asarray(small)
    ink = Image.fromarray(((array < _otsu_threshold(array)) * 255).astype(np.uint8))
    angles = np.arange(-max_angle, max_angle + step, step)
    scores = [np.var(np.asarray(ink.rotate(a, fillcolor=0)).sum(axis=1)) for a in angles]
    return float(angles[int(np.argmax(scores))])


def prepare_image(image: Image.Image) -> Image.Image:
    """Despeckle, deskew and binarize a scanned page for OCR."""
    gray = image.convert("L").filter(ImageFilter.MedianFilter(3))
    angle = _skew_angle(gray)
    if angle:
        gray = gray.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    array = np.asarray(gray)
    return Image.fromarray(((array > _otsu_threshold(array)) * 255).astype(np.uint8))


def ocr_image(image: Image.Image) -> OcrResult:
    data = pytesseract.image_to_data(
        prepare_image(image), config="--psm 3", output_type=pytesseract.Output.DICT
    )
    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        confidence = float(data["conf"][i])
        if not word or confidence < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confidences.append(confidence)

    paragraphs: dict[tuple[int, int], list[str]] = {}
    for (block, par, _), words in lines.items():
        paragraphs.setdefault((block, par), []).append(" ".join(words))
    return OcrResult(
        paragraphs=list(paragraphs.values()),
        confidence=round(sum(confidences) / len(confidences), 1) if confidences else 0.0,
    )

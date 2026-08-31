"""PaddleOCR wrapper for Egyptian ID card text fields (Arabic + digit-like fields)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# Paddle 3.3.x + PP-OCRv5 crashes on CPU oneDNN/PIR; force off before importing paddleocr.
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

MIN_CROP_HEIGHT = 32
LTR_FIELD_NAMES = frozenset({"ExpDate", "IssueDate", "Serial_Num"})
LINE_Y_TOLERANCE = 0.5

_ocr_readers: dict[str, FieldOcrReader] = {}


@dataclass(frozen=True)
class FieldOcrResult:
    text: str
    ocr_conf: float | None
    n_lines: int


def _poly_center(poly: np.ndarray) -> tuple[float, float]:
    points = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def _line_key(y_center: float, height: float) -> int:
    if height <= 0:
        return int(round(y_center))
    return int(round(y_center / max(height * LINE_Y_TOLERANCE, 1.0)))


def stitch_recognized_text(
    texts: list[str],
    scores: list[float],
    polys: list[np.ndarray],
    *,
    right_to_left: bool,
) -> tuple[str, float | None]:
    if not texts:
        return "", None

    height = 0.0
    if polys:
        ys = [point[1] for poly in polys for point in np.asarray(poly).reshape(-1, 2)]
        height = float(max(ys) - min(ys)) if ys else 0.0

    rows: dict[int, list[tuple[float, str, float]]] = {}
    for text, score, poly in zip(texts, scores, polys):
        if not str(text).strip():
            continue
        x_center, y_center = _poly_center(poly)
        bucket = _line_key(y_center, height)
        rows.setdefault(bucket, []).append((x_center, str(text).strip(), float(score)))

    if not rows:
        return "", None

    lines: list[str] = []
    line_scores: list[float] = []
    for bucket in sorted(rows):
        items = rows[bucket]
        items.sort(key=lambda item: item[0], reverse=right_to_left)
        line_text = " ".join(item[1] for item in items)
        lines.append(line_text)
        line_scores.append(sum(item[2] for item in items) / len(items))

    stitched = "\n".join(lines)
    mean_conf = sum(line_scores) / len(line_scores) if line_scores else None
    return stitched, mean_conf


def upsample_tiny_crop(image: np.ndarray, min_height: int = MIN_CROP_HEIGHT) -> np.ndarray:
    height, width = image.shape[:2]
    if height >= min_height or height == 0:
        return image
    scale = min_height / height
    new_width = max(1, int(round(width * scale)))
    return cv2.resize(image, (new_width, min_height), interpolation=cv2.INTER_CUBIC)


def _extract_ocr_payload(result: Any) -> tuple[list[str], list[float], list[np.ndarray]]:
    if result is None:
        return [], [], []
    if isinstance(result, dict):
        payload = result
    elif hasattr(result, "get"):
        payload = result
    else:
        return [], [], []

    texts = list(payload.get("rec_texts") or [])
    scores = [float(score) for score in (payload.get("rec_scores") or [])]
    polys = list(payload.get("rec_polys") or payload.get("dt_polys") or [])
    if len(polys) < len(texts):
        polys = polys + [np.zeros((4, 2), dtype=np.float32)] * (len(texts) - len(polys))
    return texts, scores, polys[: len(texts)]


class FieldOcrReader:
    def __init__(self, lang: str = "ar") -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang=lang,
            device="cpu",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def read_crop(self, image: np.ndarray, *, field_name: str) -> FieldOcrResult:
        if image is None or image.size == 0:
            return FieldOcrResult(text="", ocr_conf=None, n_lines=0)

        prepared = upsample_tiny_crop(image)
        results = self._ocr.predict(prepared)
        payload = results[0] if results else None
        texts, scores, polys = _extract_ocr_payload(payload)
        right_to_left = field_name not in LTR_FIELD_NAMES
        stitched, mean_conf = stitch_recognized_text(
            texts,
            scores,
            polys,
            right_to_left=right_to_left,
        )
        n_lines = 0 if not stitched else len(stitched.splitlines())
        return FieldOcrResult(text=stitched, ocr_conf=mean_conf, n_lines=n_lines)


def get_field_ocr_reader(lang: str = "ar") -> FieldOcrReader:
    reader = _ocr_readers.get(lang)
    if reader is None:
        reader = FieldOcrReader(lang=lang)
        _ocr_readers[lang] = reader
    return reader


def read_field_crop(
    image: np.ndarray,
    *,
    field_name: str,
    lang: str = "ar",
) -> FieldOcrResult:
    return get_field_ocr_reader(lang).read_crop(image, field_name=field_name)

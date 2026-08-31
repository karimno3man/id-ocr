"""Persist web extraction uploads, annotations, and field crops."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from typing import Any

import cv2
import numpy as np

WEB_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = WEB_DIR / "uploads"
YOLO_OCR = WEB_DIR.parent / "yolo+ocr"
if str(YOLO_OCR) not in sys.path:
    sys.path.insert(0, str(YOLO_OCR))

from card_extractor import FieldResult, annotate_fields, save_field_crops  # noqa: E402


def create_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{stamp}_{uuid4().hex[:8]}"
    run_dir = UPLOADS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _url_for(run_dir: Path, relative: str) -> str:
    run_id = run_dir.name
    return f"/uploads/{run_id}/{relative}"


def save_side_artifacts(
    run_dir: Path,
    side: str,
    image: np.ndarray,
    fields: dict[str, FieldResult],
) -> dict[str, str | dict[str, str]]:
    original_path = run_dir / f"{side}.jpg"
    annotated_path = run_dir / f"{side}_annotated.jpg"
    cv2.imwrite(str(original_path), image)
    cv2.imwrite(str(annotated_path), annotate_fields(image, fields))

    crop_paths = save_field_crops(image, fields, run_dir / "crops" / side)
    crop_urls = {
        class_name: _url_for(run_dir, f"crops/{side}/{Path(path).name}")
        for class_name, path in crop_paths.items()
    }

    return {
        "original": _url_for(run_dir, original_path.name),
        "annotated": _url_for(run_dir, annotated_path.name),
        "crops": crop_urls,
    }


def save_extraction_run(
    front_image: np.ndarray,
    back_image: np.ndarray | None,
    front_fields: dict[str, FieldResult],
    back_fields: dict[str, FieldResult],
) -> dict[str, Any]:
    run_dir = create_run_dir()
    artifacts: dict[str, Any] = {
        "run_id": run_dir.name,
        "directory": str(run_dir),
        "front": save_side_artifacts(run_dir, "front", front_image, front_fields),
    }
    if back_image is not None:
        artifacts["back"] = save_side_artifacts(run_dir, "back", back_image, back_fields)
    return artifacts

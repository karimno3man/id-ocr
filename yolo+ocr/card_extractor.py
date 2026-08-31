"""Shared card field extraction for CLI and web API."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from digit_nid import (
    NID_LEN,
    load_digit_model,
    read_nid_from_crop,
    resolve_digit_weights,
    snapshot_digit_weights,
)
from field_ocr import get_field_ocr_reader, read_field_crop

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_YAML = ROOT / "nid_localization.yaml"
DEFAULT_WEIGHTS = ROOT / "runs" / "nid_localize" / "weights" / "best.pt"
WEIGHT_CANDIDATES = (
    DEFAULT_WEIGHTS,
    ROOT / "runs" / "nid_localize" / "weights" / "best_snapshot.pt",
    ROOT / "runs" / "nid_localize" / "weights" / "last.pt",
    HERE / "weights" / "best_snapshot.pt",
)
ID_CLASS_NAME = "ID"
SIDE_MARKER_CLASSES = {
    "Front": "Detected (front)",
    "Back": "Detected (back)",
}


@dataclass(frozen=True)
class ExtractorConfig:
    data_yaml: Path = DEFAULT_YAML
    weights: Path | None = None
    digit_weights: Path | None = None
    imgsz: int = 1280
    digit_imgsz: int = 640
    conf: float = 0.2
    digit_conf: float = 0.25
    pad: float = 0.05
    right_pad: float = 1.0
    retry_pad: float = 0.07
    field_pad: float = 0.0
    ocr_lang: str = "ar"
    device: str | int = "cpu"
    snapshot_weights: bool = True


@dataclass
class FieldResult:
    text: str = ""
    det_conf: float = 0.0
    ocr_conf: float | None = None
    nid: dict[str, Any] | None = None
    source: str = ""
    box_xyxy: list[int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_class_names(data_yaml: Path) -> list[str]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = payload["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    return list(names)


def resolve_weights(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit
    for path in WEIGHT_CANDIDATES:
        if path.exists():
            return path
    searched = "\n".join(f"  {path}" for path in WEIGHT_CANDIDATES)
    raise FileNotFoundError(f"No localization YOLO weights found. Looked in:\n{searched}")


def _snapshot_localization_weights(src: Path) -> Path:
    dest = HERE / "weights" / "best_snapshot.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def clip_xyxy(
    xyxy: np.ndarray,
    width: int,
    height: int,
    pad: float = 0.0,
    pad_x: float | None = None,
    pad_y: float | None = None,
    pad_left: float | None = None,
    pad_right: float | None = None,
    clip_right: int | None = None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(v) for v in xyxy.tolist())
    bw, bh = x2 - x1, y2 - y1
    px = pad if pad_x is None else pad_x
    py = pad if pad_y is None else pad_y
    pl = px if pad_left is None else pad_left
    pr = px if pad_right is None else pad_right
    max_x = width if clip_right is None else min(width, clip_right)
    x1 = max(0, int(round(x1 - bw * pl)))
    y1 = max(0, int(round(y1 - bh * py)))
    x2 = min(max_x, int(round(x2 + bw * pr)))
    y2 = min(height, int(round(y2 + bh * py)))
    x1 = min(x1, width - 1)
    y1 = min(y1, height - 1)
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)
    return x1, y1, x2, y2


def crop_box(
    image: np.ndarray,
    xyxy: np.ndarray,
    pad: float = 0.0,
    pad_x: float | None = None,
    pad_y: float | None = None,
    pad_left: float | None = None,
    pad_right: float | None = None,
    clip_right: int | None = None,
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    height, width = image.shape[:2]
    box = clip_xyxy(
        xyxy,
        width,
        height,
        pad=pad,
        pad_x=pad_x,
        pad_y=pad_y,
        pad_left=pad_left,
        pad_right=pad_right,
        clip_right=clip_right,
    )
    x1, y1, x2, y2 = box
    return box, image[y1:y2, x1:x2]


def best_boxes_per_class(result, class_names: list[str]) -> dict[str, tuple[np.ndarray, float]]:
    if result.boxes is None or len(result.boxes) == 0:
        return {}

    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()
    best: dict[str, tuple[np.ndarray, float]] = {}

    for class_id, class_name in enumerate(class_names):
        keep = np.where(cls == class_id)[0]
        if keep.size == 0:
            continue
        best_idx = keep[np.argmax(conf[keep])]
        best[class_name] = (xyxy[best_idx], float(conf[best_idx]))
    return best


def card_right_edge(
    detections: dict[str, tuple[np.ndarray, float]],
    source: str,
    image_width: int,
) -> int:
    """Right x-limit for ID crops: end of the card (Front/Back box), not the full image."""
    preferred = "Front" if source == "front" else "Back" if source == "back" else None
    if preferred and preferred in detections:
        return min(image_width, int(round(float(detections[preferred][0][2]))))
    for class_name in ("Front", "Back"):
        if class_name in detections:
            return min(image_width, int(round(float(detections[class_name][0][2]))))
    return image_width


def _field_rank(result: FieldResult) -> tuple[int, float, float]:
    has_text = 1 if (result.text or "").strip() else 0
    return (has_text, result.det_conf, result.ocr_conf or 0.0)


def merge_field_maps(
    front: dict[str, FieldResult],
    back: dict[str, FieldResult],
    class_names: list[str],
) -> dict[str, FieldResult]:
    merged: dict[str, FieldResult] = {}
    for class_name in class_names:
        candidates = [item for item in (front.get(class_name), back.get(class_name)) if item is not None]
        if not candidates:
            merged[class_name] = FieldResult(text="")
            continue
        merged[class_name] = max(candidates, key=_field_rank)
    return merged


def merged_fields_to_response(merged: dict[str, FieldResult]) -> tuple[dict[str, str], dict[str, Any]]:
    fields: dict[str, str] = {}
    meta: dict[str, Any] = {}
    for class_name, result in merged.items():
        fields[class_name] = result.text or ""
        entry: dict[str, Any] = {
            "source": result.source or None,
            "det_conf": result.det_conf if result.det_conf else None,
        }
        if result.ocr_conf is not None:
            entry["ocr_conf"] = result.ocr_conf
        if result.nid is not None:
            entry["nid"] = result.nid
        meta[class_name] = entry
    return fields, meta


def join_field_text(fields: dict[str, Any], *names: str, sep: str = " ") -> str | None:
    parts = []
    for name in names:
        value = fields.get(name)
        if value is None:
            continue
        text = value.get("text") if isinstance(value, dict) else getattr(value, "text", "")
        if text:
            parts.append(text)
    if not parts:
        return None
    return sep.join(parts)


_ANNOTATION_COLORS_BGR = (
    (0, 102, 255),
    (0, 204, 255),
    (0, 128, 255),
    (0, 165, 255),
    (0, 200, 255),
    (0, 255, 255),
)


def _color_for_class(class_name: str) -> tuple[int, int, int]:
    return _ANNOTATION_COLORS_BGR[hash(class_name) % len(_ANNOTATION_COLORS_BGR)]


def annotate_fields(image: np.ndarray, fields: dict[str, FieldResult]) -> np.ndarray:
    vis = image.copy()
    for class_name, result in fields.items():
        if result.box_xyxy is None:
            continue
        x1, y1, x2, y2 = result.box_xyxy
        color = _color_for_class(class_name)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {result.det_conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
        label_y = max(y1 - 6, text_h + 4)
        cv2.rectangle(
            vis,
            (x1, label_y - text_h - 4),
            (x1 + text_w + 4, label_y + baseline),
            color,
            -1,
        )
        cv2.putText(vis, label, (x1 + 2, label_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return vis


def save_field_crops(
    image: np.ndarray,
    fields: dict[str, FieldResult],
    crops_dir: Path,
) -> dict[str, str]:
    crops_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for class_name, result in fields.items():
        if result.box_xyxy is None:
            continue
        x1, y1, x2, y2 = result.box_xyxy
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_path = crops_dir / f"{class_name}.jpg"
        cv2.imwrite(str(crop_path), crop)
        saved[class_name] = str(crop_path)
    return saved


class CardExtractor:
    def __init__(self, config: ExtractorConfig | None = None) -> None:
        self.config = config or ExtractorConfig()
        weights = resolve_weights(self.config.weights)
        digit_weights = resolve_digit_weights(self.config.digit_weights)
        if self.config.snapshot_weights:
            try:
                weights = _snapshot_localization_weights(weights)
            except OSError:
                pass
            try:
                digit_weights = snapshot_digit_weights(digit_weights)
            except OSError:
                pass

        self.class_names = load_class_names(self.config.data_yaml)
        self.detector = YOLO(str(weights))
        self.digit_model = load_digit_model(digit_weights)
        get_field_ocr_reader(self.config.ocr_lang)

    def warm_ocr(self) -> None:
        get_field_ocr_reader(self.config.ocr_lang)

    def extract_from_image(
        self,
        image: np.ndarray,
        *,
        source: str = "",
    ) -> dict[str, FieldResult]:
        if image is None or image.size == 0:
            return {}

        result = self.detector.predict(
            source=image,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            device=self.config.device,
            verbose=False,
        )[0]
        detections = best_boxes_per_class(result, self.class_names)
        id_clip_right = card_right_edge(detections, source, image.shape[1])
        fields: dict[str, FieldResult] = {}

        for class_name, (xyxy, det_conf) in detections.items():
            if class_name in SIDE_MARKER_CLASSES:
                fields[class_name] = FieldResult(
                    text=SIDE_MARKER_CLASSES[class_name],
                    det_conf=round(det_conf, 4),
                    source=source,
                    box_xyxy=list(map(int, clip_xyxy(xyxy, image.shape[1], image.shape[0]))),
                )
                continue

            if class_name == ID_CLASS_NAME:
                field_data = self._process_id_field(image, xyxy, det_conf, clip_right=id_clip_right)
            else:
                field_data = self._process_text_field(image, class_name, xyxy, det_conf)

            field_data.source = source
            fields[class_name] = field_data

        return fields

    def _process_id_field(
        self,
        image: np.ndarray,
        xyxy: np.ndarray,
        det_conf: float,
        *,
        clip_right: int | None = None,
    ) -> FieldResult:
        (x1, y1, x2, y2), crop = crop_box(
            image,
            xyxy,
            pad=self.config.pad,
            pad_left=self.config.pad,
            pad_right=self.config.right_pad,
            pad_y=self.config.pad,
            clip_right=clip_right,
        )

        digit_rows, stitched, decoded, trimmed = read_nid_from_crop(
            self.digit_model,
            crop,
            conf=self.config.digit_conf,
            imgsz=self.config.digit_imgsz,
            device=self.config.device,
        )
        n_retries = 0
        steps = 0
        while len(digit_rows) < NID_LEN and self.config.retry_pad > 0:
            steps += NID_LEN - len(digit_rows)
            retry_box, retry_crop = crop_box(
                image,
                xyxy,
                pad=self.config.pad,
                pad_left=self.config.pad + steps * self.config.retry_pad,
                pad_right=self.config.right_pad,
                pad_y=self.config.pad,
                clip_right=clip_right,
            )
            if retry_box == (x1, y1, x2, y2):
                break
            n_retries += 1
            (x1, y1, x2, y2), crop = retry_box, retry_crop
            digit_rows, stitched, decoded, trimmed = read_nid_from_crop(
                self.digit_model,
                crop,
                conf=self.config.digit_conf,
                imgsz=self.config.digit_imgsz,
                device=self.config.device,
            )

        display_text = decoded.nid if decoded is not None else stitched
        return FieldResult(
            text=display_text or "",
            det_conf=round(det_conf, 4),
            box_xyxy=[x1, y1, x2, y2],
            nid=None if decoded is None else decoded.to_dict(),
            extra={
                "n_boxes": len(digit_rows),
                "retried": n_retries > 0,
                "n_retries": n_retries,
                "trimmed": trimmed,
                "digits": digit_rows,
                "raw_digits": stitched,
            },
        )

    def _process_text_field(
        self,
        image: np.ndarray,
        class_name: str,
        xyxy: np.ndarray,
        det_conf: float,
    ) -> FieldResult:
        (x1, y1, x2, y2), crop = crop_box(image, xyxy, pad=self.config.field_pad)
        ocr_result = read_field_crop(crop, field_name=class_name, lang=self.config.ocr_lang)
        return FieldResult(
            text=ocr_result.text or "",
            det_conf=round(det_conf, 4),
            ocr_conf=None if ocr_result.ocr_conf is None else round(ocr_result.ocr_conf, 4),
            box_xyxy=[x1, y1, x2, y2],
            extra={"n_lines": ocr_result.n_lines},
        )

    def extract_front_back(
        self,
        front: np.ndarray,
        back: np.ndarray | None = None,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, FieldResult], dict[str, FieldResult]]:
        front_fields = self.extract_from_image(front, source="front")
        back_fields = self.extract_from_image(back, source="back") if back is not None else {}
        merged = merge_field_maps(front_fields, back_fields, self.class_names)
        fields, meta = merged_fields_to_response(merged)
        return fields, meta, front_fields, back_fields

    def field_result_to_dict(self, result: FieldResult, crop_path: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": result.text,
            "det_conf": result.det_conf,
            "box_xyxy": result.box_xyxy,
            "source": result.source,
        }
        if result.ocr_conf is not None:
            payload["ocr_conf"] = result.ocr_conf
        if result.nid is not None:
            payload["nid"] = result.nid
        if crop_path is not None:
            payload["crop"] = crop_path
        payload.update(result.extra)
        return payload

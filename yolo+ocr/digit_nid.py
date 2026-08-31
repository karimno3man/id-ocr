"""Run digit YOLO on ID-number crops and decode a 14-digit Egyptian NID."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from decode_nid import NidDecode, decode_nid

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
NID_LEN = 14

DEFAULT_INPUT = HERE / "crops" / "pred" / "ID"
DEFAULT_OUT = HERE / "digit_results.jsonl"

DIGIT_WEIGHT_CANDIDATES = (
    ROOT / "runs" / "nid_digits" / "weights" / "best.pt",
    ROOT / "runs" / "nid_digits" / "weights" / "last.pt",
    HERE / "weights" / "digit_best.pt",
)


def resolve_digit_weights(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit
    for path in DIGIT_WEIGHT_CANDIDATES:
        if path.exists():
            return path
    searched = "\n".join(f"  {path}" for path in DIGIT_WEIGHT_CANDIDATES)
    raise FileNotFoundError(f"No digit YOLO weights found. Looked in:\n{searched}")


def snapshot_digit_weights(src: Path) -> Path:
    dest = HERE / "weights" / "digit_best.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def load_digit_model(weights: Path) -> YOLO:
    return YOLO(str(weights))


def pred_nid_from_result(
    result: Any,
    conf: float = 0.25,
) -> tuple[str | None, int, list[dict[str, Any]]]:
    if result.boxes is None or len(result.boxes) == 0:
        return None, 0, []

    cls = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()
    keep = scores >= conf
    if not keep.any():
        return None, 0, []

    cls = cls[keep]
    scores = scores[keep]
    xyxy = xyxy[keep]
    order = np.argsort([(box[0] + box[2]) / 2 for box in xyxy])

    digits: list[dict[str, Any]] = []
    for idx in order:
        box = xyxy[idx].tolist()
        digits.append(
            {
                "digit": str(int(cls[idx])),
                "score": round(float(scores[idx]), 4),
                "box": [round(v, 1) for v in box],
            }
        )

    stitched = "".join(item["digit"] for item in digits)
    return stitched or None, len(digits), digits


def decode_digit_rows(
    digit_rows: list[dict[str, Any]],
) -> tuple[str, NidDecode | None, bool]:
    stitched = "".join(item["digit"] for item in digit_rows)
    n_boxes = len(digit_rows)
    if n_boxes < NID_LEN:
        return stitched, None, False

    trimmed = n_boxes > NID_LEN
    decoded = decode_nid(stitched[-NID_LEN:])
    if not decoded.is_valid_structure:
        return stitched, None, trimmed
    return stitched, decoded, trimmed


def read_nid_from_crop(
    model: YOLO,
    image: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | int = 0,
) -> tuple[list[dict[str, Any]], str, NidDecode | None, bool]:
    result = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )[0]
    _, _, digit_rows = pred_nid_from_result(result, conf=conf)
    stitched, decoded, trimmed = decode_digit_rows(digit_rows)
    return digit_rows, stitched, decoded, trimmed


def iter_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read Egyptian NID digits from ID crops with digit YOLO.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Crop image or folder")
    parser.add_argument("--weights", type=Path, default=None, help="Digit YOLO weights")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=0)
    parser.add_argument("--no-snapshot", action="store_true", help="Read weights in place; skip copy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    images = iter_images(args.input)
    if not images:
        raise FileNotFoundError(f"No images under {args.input}")

    weights = resolve_digit_weights(args.weights)
    if not args.no_snapshot:
        try:
            weights = snapshot_digit_weights(weights)
            print(f"Snapshot digit weights -> {weights}")
        except OSError as exc:
            print(f"Could not snapshot {weights} ({exc}); reading in place")

    model = load_digit_model(weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                record = {"path": str(path), "error": "unreadable image"}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"{path.name}\tUNREADABLE")
                continue

            digit_rows, stitched, decoded, trimmed = read_nid_from_crop(
                model,
                image,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
            )
            if decoded is not None:
                n_ok += 1
            record = {
                "path": str(path),
                "nid_digits": stitched,
                "n_boxes": len(digit_rows),
                "trimmed": trimmed,
                "digits": digit_rows,
                "nid": None if decoded is None else decoded.to_dict(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            nid_value = "-" if decoded is None else decoded.nid
            trim_flag = "\ttrim" if trimmed else ""
            print(f"{path.name}\tboxes={len(digit_rows)}{trim_flag}\t{nid_value}\t{stitched}")

    print(f"Decoded {n_ok}/{len(images)} NIDs -> {args.out}")


if __name__ == "__main__":
    main()

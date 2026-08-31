"""YOLO card localization -> crop -> digit YOLO (ID) / PaddleOCR (other fields)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from card_extractor import (
    CardExtractor,
    ExtractorConfig,
    ID_CLASS_NAME,
    join_field_text,
)
from digit_nid import IMAGE_SUFFIXES

DEFAULT_SOURCE = HERE / "real-samples"
DEFAULT_CROPS_ROOT = HERE / "crops" / "pred"
DEFAULT_OUT = HERE / "results.jsonl"
DEFAULT_YAML = ROOT / "nid_localization.yaml"
DEFAULT_WEIGHTS = ROOT / "runs" / "nid_localize" / "weights" / "best.pt"


def iter_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect all card fields, read ID digits with YOLO, and OCR other fields with PaddleOCR."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Card localization YOLO weights")
    parser.add_argument("--digit-weights", type=Path, default=None, help="Digit YOLO weights")
    parser.add_argument("--data", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--crops-root", type=Path, default=DEFAULT_CROPS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--imgsz", type=int, default=1280, help="Localization inference size")
    parser.add_argument("--digit-imgsz", type=int, default=640, help="Digit YOLO inference size")
    parser.add_argument("--conf", type=float, default=0.2, help="Localization confidence")
    parser.add_argument("--digit-conf", type=float, default=0.25, help="Digit detection confidence")
    parser.add_argument(
        "--pad",
        type=float,
        default=0.05,
        help="Grow the ID box by this fraction on the left/top/bottom (0.05 ≈ half a digit)",
    )
    parser.add_argument(
        "--right-pad",
        type=float,
        default=1.0,
        help="Right-side pad on the ID crop (fraction of ID box width; clamped to the card edge, or image edge if no Front/Back box)",
    )
    parser.add_argument(
        "--retry-pad",
        type=float,
        default=0.07,
        help="Extra left-side pad per missing digit; retries until 14 boxes or the crop cannot grow",
    )
    parser.add_argument(
        "--field-pad",
        type=float,
        default=0.0,
        help="Optional pad fraction for non-ID field crops (default: no extra padding)",
    )
    parser.add_argument("--ocr-lang", default="ar", help="PaddleOCR language for text fields")
    parser.add_argument("--device", default="cpu", help="Inference device for YOLO models")
    parser.add_argument("--no-snapshot", action="store_true", help="Read weights in place; skip copy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)
    images = iter_images(args.source)
    if not images:
        raise FileNotFoundError(f"No images under {args.source}")

    config = ExtractorConfig(
        data_yaml=args.data,
        weights=args.weights,
        digit_weights=args.digit_weights,
        imgsz=args.imgsz,
        digit_imgsz=args.digit_imgsz,
        conf=args.conf,
        digit_conf=args.digit_conf,
        pad=args.pad,
        right_pad=args.right_pad,
        retry_pad=args.retry_pad,
        field_pad=args.field_pad,
        ocr_lang=args.ocr_lang,
        device=args.device,
        snapshot_weights=not args.no_snapshot,
    )
    extractor = CardExtractor(config)
    args.crops_root.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_cards = 0
    n_nid = 0
    n_fields = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                record = {"path": str(path), "error": "unreadable image"}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"{path.name}\tUNREADABLE")
                continue

            raw_fields = extractor.extract_from_image(image, source=path.stem)
            if not raw_fields:
                record = {"path": str(path), "error": "no detections"}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"{path.name}\tNO_DETECTIONS")
                continue

            n_cards += 1
            fields: dict[str, Any] = {}
            for class_name, field_result in raw_fields.items():
                crop_dir = args.crops_root / class_name
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_path = crop_dir / f"{path.stem}.jpg"
                if field_result.box_xyxy is not None:
                    x1, y1, x2, y2 = field_result.box_xyxy
                    crop = image[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(str(crop_path), crop)
                fields[class_name] = extractor.field_result_to_dict(field_result, str(crop_path))
                n_fields += 1

            id_field = fields.get(ID_CLASS_NAME)
            decoded = None if id_field is None else id_field.get("nid")
            if decoded is not None:
                n_nid += 1

            record: dict[str, Any] = {
                "path": str(path),
                "fields": fields,
                "name": join_field_text(fields, "First_Name", "Last_Name"),
                "address": join_field_text(fields, "Add1", "Add2"),
                "job": join_field_text(fields, "Job1", "Job2"),
            }
            if id_field is not None:
                record.update(
                    {
                        "crop": id_field.get("crop"),
                        "det_conf": id_field.get("det_conf"),
                        "box_xyxy": id_field.get("box_xyxy"),
                        "ocr_text": id_field.get("raw_digits", id_field.get("text")),
                        "n_boxes": id_field.get("n_boxes"),
                        "retried": id_field.get("retried"),
                        "n_retries": id_field.get("n_retries"),
                        "trimmed": id_field.get("trimmed"),
                        "digits": id_field.get("digits"),
                        "nid": id_field.get("nid"),
                    }
                )

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            nid_value = "-" if decoded is None else decoded.get("nid", "-")
            print(f"{path.name}\tfields={len(fields)}\tnid={nid_value}")

    print(
        f"Cards {n_cards}/{len(images)} | fields {n_fields} | decoded NIDs {n_nid} | {args.out}"
    )


if __name__ == "__main__":
    main()

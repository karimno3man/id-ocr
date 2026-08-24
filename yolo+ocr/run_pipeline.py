"""YOLO ID localization -> crop -> digit YOLO -> Egyptian NID decode."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from digit_nid import (
    IMAGE_SUFFIXES,
    load_digit_model,
    read_nid_from_crop,
    resolve_digit_weights,
    snapshot_digit_weights,
)

DEFAULT_SOURCE = HERE / "real-samples"
DEFAULT_CROPS = HERE / "crops" / "pred" / "ID"
DEFAULT_OUT = HERE / "results.jsonl"
DEFAULT_YAML = ROOT / "nid_localization.yaml"
WEIGHT_CANDIDATES = (
    ROOT / "runs" / "nid_localize" / "weights" / "best.pt",
    ROOT / "runs" / "nid_localize" / "weights" / "best_snapshot.pt",
    ROOT / "runs" / "nid_localize" / "weights" / "last.pt",
    HERE / "weights" / "best_snapshot.pt",
)


def resolve_id_class(data_yaml: Path) -> int:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = payload["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    return names.index("ID")


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


def snapshot_weights(src: Path) -> Path:
    dest = HERE / "weights" / "best_snapshot.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def pad_xyxy(xyxy: np.ndarray, width: int, height: int, pad: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy.tolist()
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(width, int(x2 + bw * pad))
    y2 = min(height, int(y2 + bh * pad))
    return x1, y1, x2, y2


def iter_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def best_id_box(result, class_id: int) -> tuple[np.ndarray, float] | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()
    keep = np.where(cls == class_id)[0]
    if keep.size == 0:
        return None
    best = keep[np.argmax(conf[keep])]
    return xyxy[best], float(conf[best])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect the NID box, read digits with YOLO, and decode 14 digits.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--weights", type=Path, default=None, help="Card localization YOLO weights")
    parser.add_argument("--digit-weights", type=Path, default=None, help="Digit YOLO weights")
    parser.add_argument("--data", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--crops", type=Path, default=DEFAULT_CROPS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--imgsz", type=int, default=1280, help="Localization inference size")
    parser.add_argument("--digit-imgsz", type=int, default=640, help="Digit YOLO inference size")
    parser.add_argument("--conf", type=float, default=0.2, help="Localization confidence")
    parser.add_argument("--digit-conf", type=float, default=0.25, help="Digit detection confidence")
    parser.add_argument("--pad", type=float, default=0.15)
    parser.add_argument("--device", default="cpu", help="Inference device for both YOLO models")
    parser.add_argument("--no-snapshot", action="store_true", help="Read weights in place; skip copy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)
    images = iter_images(args.source)
    if not images:
        raise FileNotFoundError(f"No images under {args.source}")

    weights = resolve_weights(args.weights)
    digit_weights = resolve_digit_weights(args.digit_weights)
    if not args.no_snapshot:
        try:
            weights = snapshot_weights(weights)
            print(f"Snapshot localization weights -> {weights}")
        except OSError as exc:
            print(f"Could not snapshot {weights} ({exc}); reading in place")
        try:
            digit_weights = snapshot_digit_weights(digit_weights)
            print(f"Snapshot digit weights -> {digit_weights}")
        except OSError as exc:
            print(f"Could not snapshot {digit_weights} ({exc}); reading in place")

    class_id = resolve_id_class(args.data)
    detector = YOLO(str(weights))
    digit_model = load_digit_model(digit_weights)
    args.crops.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_det = 0
    n_nid = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                record = {"path": str(path), "error": "unreadable image"}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"{path.name}\tUNREADABLE")
                continue

            result = detector.predict(
                source=image,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=[class_id],
                device=args.device,
                verbose=False,
            )[0]
            box = best_id_box(result, class_id)
            if box is None:
                record = {"path": str(path), "error": "no ID box"}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"{path.name}\tNO_ID")
                continue

            xyxy, conf = box
            h, w = image.shape[:2]
            pad = args.pad
            if float(xyxy[3] - xyxy[1]) < 80:
                pad = max(pad, 0.4)
            x1, y1, x2, y2 = pad_xyxy(xyxy, w, h, pad)
            crop = image[y1:y2, x1:x2]
            crop_path = args.crops / f"{path.stem}.jpg"
            cv2.imwrite(str(crop_path), crop)
            n_det += 1

            digit_rows, stitched, decoded = read_nid_from_crop(
                digit_model,
                crop,
                conf=args.digit_conf,
                imgsz=args.digit_imgsz,
                device=args.device,
            )
            if decoded is not None:
                n_nid += 1
            record = {
                "path": str(path),
                "crop": str(crop_path),
                "det_conf": round(conf, 4),
                "box_xyxy": [x1, y1, x2, y2],
                "ocr_text": stitched,
                "n_boxes": len(digit_rows),
                "digits": digit_rows,
                "nid": None if decoded is None else decoded.to_dict(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            nid_value = "-" if decoded is None else decoded.nid
            print(f"{path.name}\tconf={conf:.3f}\tboxes={len(digit_rows)}\t{nid_value}\t{stitched}")

    print(f"ID crops {n_det}/{len(images)} | decoded {n_nid} | {args.out}")


if __name__ == "__main__":
    main()

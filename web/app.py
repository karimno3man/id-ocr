"""Localhost FastAPI server for iSchool National ID OCR."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
YOLO_OCR = ROOT / "yolo+ocr"
if str(YOLO_OCR) not in sys.path:
    sys.path.insert(0, str(YOLO_OCR))

from card_extractor import CardExtractor, ExtractorConfig  # noqa: E402
from .storage import UPLOADS_DIR, save_extraction_run

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"

_extractor: CardExtractor | None = None


def _default_device() -> str | int:
    return 0 if torch.cuda.is_available() else "cpu"


def _decode_upload(file_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _extractor
    config = ExtractorConfig(device=_default_device(), snapshot_weights=True)
    _extractor = CardExtractor(config)
    _extractor.warm_ocr()
    yield
    _extractor = None


app = FastAPI(title="iSchool National ID OCR", lifespan=lifespan)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ready": _extractor is not None}


@app.post("/api/extract")
async def extract(
    front: UploadFile = File(...),
    back: UploadFile | None = File(None),
) -> dict:
    if _extractor is None:
        raise HTTPException(status_code=503, detail="Extractor not ready")

    front_bytes = await front.read()
    front_image = _decode_upload(front_bytes)
    if front_image is None:
        raise HTTPException(status_code=400, detail="Could not read front image")

    back_image = None
    if back is not None and back.filename:
        back_bytes = await back.read()
        if back_bytes:
            back_image = _decode_upload(back_bytes)
            if back_image is None:
                raise HTTPException(status_code=400, detail="Could not read back image")

    fields, meta, front_fields, back_fields = _extractor.extract_front_back(front_image, back_image)
    artifacts = save_extraction_run(front_image, back_image, front_fields, back_fields)
    return {"fields": fields, "meta": meta, "artifacts": artifacts}

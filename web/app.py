"""FastAPI server for iSchool National ID OCR (local + ECS Express Mode)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

YOLO_OCR = ROOT / "yolo+ocr"
if str(YOLO_OCR) not in sys.path:
    sys.path.insert(0, str(YOLO_OCR))

from card_extractor import CardExtractor, ExtractorConfig  # noqa: E402
from .storage import UPLOADS_DIR, save_extraction_run

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"

logger = logging.getLogger(__name__)

SHEET_FIELD_COLUMNS = [
    "First_Name",
    "Last_Name",
    "HusbandName",
    "Gender",
    "Religion",
    "Status",
    "ID",
    "IssueDate",
    "ExpDate",
    "Serial_Num",
    "Add1",
    "Add2",
    "Job1",
    "Job2",
    "Front",
    "Back",
]

_extractor: CardExtractor | None = None
_extractor_lock = threading.Lock()
_load_error: str | None = None


def _default_device() -> str | int:
    return 0 if torch.cuda.is_available() else "cpu"


def _decode_upload(file_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _load_extractor_background() -> None:
    global _extractor, _load_error
    try:
        config = ExtractorConfig(device=_default_device(), snapshot_weights=True)
        extractor = CardExtractor(config)
        extractor.warm_ocr()
        with _extractor_lock:
            _extractor = extractor
        logger.info("CardExtractor ready (device=%s)", _default_device())
    except Exception as exc:
        logger.exception("Failed to load CardExtractor")
        with _extractor_lock:
            _load_error = str(exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    thread = threading.Thread(target=_load_extractor_background, daemon=True, name="extractor-load")
    thread.start()
    yield
    with _extractor_lock:
        global _extractor
        _extractor = None


app = FastAPI(title="iSchool National ID OCR", lifespan=lifespan)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _health_payload() -> dict[str, bool | str | None]:
    with _extractor_lock:
        ready = _extractor is not None
        error = _load_error
    payload: dict[str, bool | str | None] = {"ready": ready}
    if error and not ready:
        payload["error"] = error
    return payload


@app.get("/health")
def health() -> dict[str, bool | str | None]:
    return _health_payload()


@app.get("/ping")
def ping() -> dict[str, bool | str | None]:
    return _health_payload()


class SubmitBody(BaseModel):
    fields: dict[str, str]


def normalize_sheet_fields(raw_fields: dict[str, str]) -> dict[str, str]:
    return {name: str(raw_fields.get(name, "") or "") for name in SHEET_FIELD_COLUMNS}


async def post_to_google_sheets(payload: dict[str, Any]) -> dict[str, Any]:
    webhook_url = os.environ.get("GOOGLE_SHEETS_WEBHOOK_URL", "").strip()
    if not webhook_url:
        raise HTTPException(
            status_code=503,
            detail="Google Sheets webhook is not configured (set GOOGLE_SHEETS_WEBHOOK_URL).",
        )

    body = dict(payload)
    token = os.environ.get("GOOGLE_SHEETS_TOKEN", "").strip()
    if token:
        body["token"] = token

    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def send_request() -> dict[str, Any]:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        return await asyncio.to_thread(send_request)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        logger.error("Google Sheets webhook HTTP %s: %s", exc.code, error_body)
        raise HTTPException(
            status_code=502,
            detail="Google Sheets rejected the submission.",
        ) from exc
    except urllib.error.URLError as exc:
        logger.error("Google Sheets webhook unreachable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Could not reach Google Sheets.",
        ) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail="Invalid response from Google Sheets.",
        ) from exc


@app.post("/api/submit")
async def submit_fields(body: SubmitBody) -> dict[str, Any]:
    submitted_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "fields": normalize_sheet_fields(body.fields),
        "submitted_at": submitted_at,
    }
    result = await post_to_google_sheets(payload)

    if not result.get("ok"):
        raise HTTPException(
            status_code=502,
            detail=result.get("error", "Google Sheets submission failed."),
        )

    return {"ok": True, "submitted_at": submitted_at}


@app.post("/api/extract")
async def extract(
    front: UploadFile = File(...),
    back: UploadFile | None = File(None),
) -> dict:
    with _extractor_lock:
        extractor = _extractor
        error = _load_error
    if extractor is None:
        detail = "Extractor not ready"
        if error:
            detail = f"{detail}: {error}"
        raise HTTPException(status_code=503, detail=detail)

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

    fields, meta, front_fields, back_fields = extractor.extract_front_back(front_image, back_image)
    artifacts = save_extraction_run(front_image, back_image, front_fields, back_fields)
    return {"fields": fields, "meta": meta, "artifacts": artifacts}

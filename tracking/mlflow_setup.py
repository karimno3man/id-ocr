"""Configure Ultralytics + MLflow experiment tracking for YOLO training."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from ultralytics import settings


def configure_mlflow_tracking(
    root: Path,
    experiment_name: str,
    run_prefix: str,
    run_suffix: str | None = None,
) -> str:
    """Enable MLflow in Ultralytics and return a unique YOLO run folder name."""
    mlflow_dir = root / "mlruns"
    mlflow_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MLFLOW_TRACKING_URI", str(mlflow_dir.resolve()))
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_prefix}_{suffix}"
    os.environ["MLFLOW_RUN"] = run_name

    settings.update({"mlflow": True})

    print("MLflow store:", os.environ["MLFLOW_TRACKING_URI"])
    print("MLflow experiment:", experiment_name)
    print("MLflow run:", run_name)
    return run_name

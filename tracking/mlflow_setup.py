"""Configure Ultralytics + MLflow experiment tracking for YOLO training."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from ultralytics import settings


def sqlite_tracking_uri(project_root: Path) -> str:
    """Build a SQLite tracking URI for MLflow 3+ (file store is deprecated)."""
    db_path = project_root.resolve() / "mlflow.db"
    return f"sqlite:///{db_path.as_posix()}"


def configure_mlflow_tracking(
    root: Path,
    experiment_name: str,
    run_prefix: str,
    run_suffix: str | None = None,
) -> str:
    """Enable MLflow in Ultralytics and return a unique YOLO run folder name."""
    root = root.resolve()
    artifacts_dir = root / "mlartifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = sqlite_tracking_uri(root)
    artifact_uri = artifacts_dir.resolve().as_uri()

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
    except MlflowException:
        pass
    mlflow.set_experiment(experiment_name)

    suffix = run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_prefix}_{suffix}"
    os.environ["MLFLOW_RUN"] = run_name

    settings.update({"mlflow": True})

    print("MLflow tracking URI:", tracking_uri)
    print("MLflow artifacts:", artifact_uri)
    print("MLflow experiment:", experiment_name)
    print("MLflow run:", run_name)
    return run_name

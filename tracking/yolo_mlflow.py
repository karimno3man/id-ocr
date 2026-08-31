"""MLflow logging helpers for Ultralytics YOLO training runs."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.exceptions import MlflowException

from tracking.mlflow_setup import sqlite_tracking_uri

LOGGER = logging.getLogger(__name__)

MLFLOW_PARAM_MAX_LENGTH = 500

# Ultralytics MLflow callback already logs train losses, val losses, LR, and mAP each epoch
# at step=trainer.epoch (0-based). Only log CSV fields that are not covered there.
SUPPLEMENTAL_CSV_METRICS = frozenset({"time"})


def sanitize_metric_key(key: str) -> str:
    """Mirror Ultralytics MLflow callback key sanitization."""
    return key.replace("(", "").replace(")", "")


def _stringify_param(value: Any) -> str:
    text = str(value)
    if len(text) > MLFLOW_PARAM_MAX_LENGTH:
        return text[:MLFLOW_PARAM_MAX_LENGTH - 3] + "..."
    return text


def load_run_params(args_yaml: Path) -> dict[str, str]:
    """Load training args from Ultralytics args.yaml for MLflow params."""
    if not args_yaml.is_file():
        LOGGER.warning("No args.yaml at %s", args_yaml)
        return {}

    with args_yaml.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    params: dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        params[str(key)] = _stringify_param(value)
    return params


def log_metrics_from_results_csv(csv_path: Path, step_col: str = "epoch") -> int:
    """Log every numeric column from results.csv, one MLflow step per row."""
    if not csv_path.is_file():
        LOGGER.warning("No results.csv at %s", csv_path)
        return 0

    logged_epochs = 0
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get(step_col):
                continue
            try:
                step = int(float(row[step_col]))
            except (TypeError, ValueError):
                continue

            metrics: dict[str, float] = {}
            for key, value in row.items():
                if key == step_col or not value:
                    continue
                try:
                    metrics[sanitize_metric_key(key)] = float(value)
                except (TypeError, ValueError):
                    continue

            if metrics:
                mlflow.log_metrics(metrics=metrics, step=step)
                logged_epochs += 1

    return logged_epochs


def log_run_artifacts(run_dir: Path) -> None:
    """Upload the full YOLO run directory tree to MLflow artifacts."""
    if not run_dir.is_dir():
        LOGGER.warning("Run directory does not exist: %s", run_dir)
        return
    mlflow.log_artifacts(str(run_dir))


def ensure_experiment(root: Path, experiment_name: str) -> None:
    """Create experiment if missing and set it as active."""
    root = root.resolve()
    artifacts_dir = root / "mlartifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_uri = artifacts_dir.resolve().as_uri()

    tracking_uri = sqlite_tracking_uri(root)
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
    except MlflowException:
        pass
    mlflow.set_experiment(experiment_name)


def find_run_by_name(experiment_name: str, run_name: str) -> str | None:
    """Return run_id if a run with the given name already exists in the experiment."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        max_results=1,
    )
    if runs.empty:
        return None
    return str(runs.iloc[0]["run_id"])


def log_yolo_run_from_disk(
    root: Path,
    run_dir: Path,
    experiment_name: str,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Backfill a completed YOLO run folder into MLflow (params, metrics, artifacts)."""
    root = root.resolve()
    run_dir = run_dir.resolve()
    resolved_run_name = run_name or run_dir.name

    ensure_experiment(root, experiment_name)

    with mlflow.start_run(run_name=resolved_run_name):
        if tags:
            mlflow.set_tags(tags)

        params = load_run_params(run_dir / "args.yaml")
        if params:
            mlflow.log_params(params)

        epochs_logged = log_metrics_from_results_csv(run_dir / "results.csv")
        mlflow.set_tag("epochs_logged", str(epochs_logged))

        log_run_artifacts(run_dir)

        run_id = mlflow.active_run().info.run_id
        LOGGER.info(
            "Logged run %s (%s) — %d epochs, experiment %s",
            resolved_run_name,
            run_id,
            epochs_logged,
            experiment_name,
        )
        return run_id


def _mlflow_is_active(trainer: Any) -> bool:
    return bool(getattr(trainer, "_mlflow_active", False))


def _trainer_epoch_step(trainer: Any) -> int | None:
    """MLflow step aligned with Ultralytics callback (0-based trainer.epoch)."""
    epoch = getattr(trainer, "epoch", None)
    if epoch is None:
        return None
    try:
        return int(epoch)
    except (TypeError, ValueError):
        return None


def _log_supplemental_csv_metrics(trainer: Any) -> None:
    """Log CSV metrics missing from the Ultralytics MLflow callback (e.g. time)."""
    if not _mlflow_is_active(trainer):
        return

    step = _trainer_epoch_step(trainer)
    if step is None:
        return

    csv_path = Path(trainer.save_dir) / "results.csv"
    if not csv_path.is_file():
        return

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    row = rows[-1]
    metrics: dict[str, float] = {}
    for key, value in row.items():
        if key == "epoch" or not value:
            continue
        metric_key = sanitize_metric_key(key)
        if metric_key not in SUPPLEMENTAL_CSV_METRICS:
            continue
        try:
            metrics[metric_key] = float(value)
        except (TypeError, ValueError):
            continue

    if metrics:
        mlflow.log_metrics(metrics=metrics, step=step)


def _log_all_run_artifacts(trainer: Any) -> None:
    """Recursively upload the full save_dir after training."""
    if not _mlflow_is_active(trainer):
        return

    save_dir = Path(trainer.save_dir)
    if save_dir.is_dir():
        mlflow.log_artifacts(str(save_dir))


def on_supplemental_fit_epoch_end(trainer: Any) -> None:
    """Log CSV metrics not already emitted by Ultralytics (time), at trainer.epoch step."""
    _log_supplemental_csv_metrics(trainer)


def on_supplemental_train_end(trainer: Any) -> None:
    """Upload the full run directory tree to MLflow artifacts."""
    _log_all_run_artifacts(trainer)


def register_supplemental_mlflow_callbacks(model: Any) -> None:
    """Attach callbacks for non-overlapping metrics (time) and full run artifacts."""
    model.add_callback("on_fit_epoch_end", on_supplemental_fit_epoch_end)
    model.add_callback("on_train_end", on_supplemental_train_end)

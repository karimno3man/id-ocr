"""Backfill existing YOLO run folders into MLflow experiments."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from tracking.yolo_mlflow import find_run_by_name, log_yolo_run_from_disk

LOGGER = logging.getLogger(__name__)

BACKFILL_RUNS = [
    ("nid-localization", "runs/nid_localize", "nid_localize", "train_nid_yolo.ipynb"),
    ("nid-digits", "runs/nid_digits", "nid_digits", "train_nid_digits.ipynb"),
]


def count_epochs(results_csv: Path) -> int:
    if not results_csv.is_file():
        return 0
    with results_csv.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def backfill_all(root: Path, force: bool = False) -> list[str]:
    """Backfill all configured historical runs. Returns list of logged run_ids."""
    root = root.resolve()
    logged_run_ids: list[str] = []

    for experiment_name, run_rel, run_name, notebook in BACKFILL_RUNS:
        run_dir = root / run_rel
        if not run_dir.is_dir():
            LOGGER.warning("Skipping %s — directory not found: %s", run_name, run_dir)
            continue

        existing_run_id = find_run_by_name(experiment_name, run_name)
        if existing_run_id and not force:
            LOGGER.info(
                "Skipping %s — run already exists (run_id=%s). Use --force to re-log.",
                run_name,
                existing_run_id,
            )
            continue

        epochs = count_epochs(run_dir / "results.csv")
        tags = {
            "source": "backfill",
            "notebook": notebook,
            "epochs_completed": str(epochs),
        }

        run_id = log_yolo_run_from_disk(
            root=root,
            run_dir=run_dir,
            experiment_name=experiment_name,
            run_name=run_name,
            tags=tags,
        )
        logged_run_ids.append(run_id)
        print(f"Backfilled {run_name} -> {experiment_name} (run_id={run_id}, epochs={epochs})")

    return logged_run_ids


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Backfill YOLO runs into MLflow.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-log runs even if a run with the same name already exists",
    )
    args = parser.parse_args()

    run_ids = backfill_all(args.root, force=args.force)
    if not run_ids:
        print("No runs were backfilled.")
    else:
        print(f"Backfilled {len(run_ids)} run(s).")


if __name__ == "__main__":
    main()

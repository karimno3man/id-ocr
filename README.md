# Egyptian National ID OCR

Two-stage YOLO pipeline for reading the 14-digit national ID number on Egyptian ID cards.

1. **Localization** — find the ID number field on the card (`train_nid_yolo.ipynb`)
2. **Digit detection** — detect each Eastern Arabic digit `٠–٩` in the crop (`train_nid_digits.ipynb`)
3. **Decode** — parse birth date, governorate, gender from the 14-digit string (`yolo+ocr/decode_nid.py`)

## Layout

```
ID-OCR/
  train_nid_yolo.ipynb       # card → ID box
  train_nid_digits.ipynb     # digit boxes on cro4 strips
  nid_localization.yaml
  nid_digits.yaml
  yolo+ocr/
    run_pipeline.py          # full card inference
    digit_nid.py             # crops-only inference
    decode_nid.py
  Thndr-National-Card.v4-v4.yolov8/   # local, gitignored
  cro4.v1-8.yolov8/                  # local, gitignored
  runs/                               # local, gitignored (YOLO outputs per run)
  mlruns/                             # local, gitignored (MLflow experiment store)
  tracking/mlflow_setup.py            # MLflow + Ultralytics wiring
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

In Cursor/VS Code, select the `.venv` kernel before running notebooks.

## Datasets (local only)

Download from Roboflow Universe and place in the repo root (not committed to git):

- **[Thndr National Card](https://universe.roboflow.com/thndr-ovgh9/thndr-national-card/browse?queryText=class%3AID&pageSize=50&startingIndex=0&browseQuery=true)** → folder `Thndr-National-Card.v4-v4.yolov8` — card field localization (16 classes; use the **ID** class for the number field)
- **[cro4](https://universe.roboflow.com/re8/cro4)** → folder `cro4.v1-8.yolov8` — per-digit boxes on ID number strips (10 classes `0–9`, 80/10/10 split)

Export each dataset in **YOLOv8** format from Roboflow before training.

Trained weights land in `runs/<run_name>/weights/best.pt` after each notebook training run. Each train creates a timestamped run folder and logs to MLflow.

## Experiment tracking (MLflow)

Training notebooks call [`tracking/mlflow_setup.py`](tracking/mlflow_setup.py) before `model.train()`. Ultralytics logs parameters, per-epoch metrics, and artifacts (weights, plots, `results.csv`) to `mlruns/`.

Experiments:

- `nid-localization` — [`train_nid_yolo.ipynb`](train_nid_yolo.ipynb)
- `nid-digits` — [`train_nid_digits.ipynb`](train_nid_digits.ipynb)

View and compare runs:

```powershell
mlflow ui --backend-store-uri mlruns
```

Open http://127.0.0.1:5000 in your browser. `mlruns/` is gitignored (large artifacts stay local).

## Train

1. `train_nid_yolo.ipynb` — localization on Thndr → `runs/nid_localize_<timestamp>/weights/best.pt`
2. `train_nid_digits.ipynb` — digits on cro4 → `runs/nid_digits_<timestamp>/weights/best.pt`

Digit training uses `fliplr=0` (digits are left-to-right).

## Infer

Full card (detect ID box, crop, read digits, decode):

```powershell
python yolo+ocr/run_pipeline.py --source yolo+ocr/real-samples --device 0
```

ID crops only:

```powershell
python yolo+ocr/digit_nid.py --input yolo+ocr/crops/pred/ID --device 0
```

Outputs are JSONL files under `yolo+ocr/` (`results.jsonl`, `digit_results.jsonl`).

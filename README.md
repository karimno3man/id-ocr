# Egyptian National ID OCR

YOLO localization + field OCR for Egyptian national ID cards.

1. **Localization** — find all 16 card fields (`train_nid_yolo.ipynb`)
2. **ID digits** — detect each Eastern Arabic digit `٠–٩` in the ID crop (`train_nid_digits.ipynb`)
3. **Text fields** — read the other 15 fields with PaddleOCR Arabic (`yolo+ocr/field_ocr.py`)
4. **Decode** — parse birth date, governorate, gender from the 14-digit string (`yolo+ocr/decode_nid.py`)

## Layout

```
ID-OCR/
  train_nid_yolo.ipynb       # card → ID box
  train_nid_digits.ipynb     # digit boxes on cro4 strips
  nid_localization.yaml
  nid_digits.yaml
  yolo+ocr/
    run_pipeline.py          # full card inference (all 16 fields)
    card_extractor.py        # shared extractor for CLI + web
    field_ocr.py             # PaddleOCR wrapper for text fields
    digit_nid.py             # ID crops-only digit inference
    decode_nid.py
  web/
    app.py                   # localhost FastAPI UI server
    static/                  # iSchool-branded front/back upload UI
  Thndr-National-Card.v4-v4.yolov8/   # local, gitignored
  cro4.v1-8.yolov8/                  # local, gitignored
  runs/                               # local, gitignored (YOLO outputs per run)
  mlflow.db                           # local, gitignored (MLflow SQLite store)
  mlartifacts/                        # local, gitignored (MLflow run artifacts)
  tracking/mlflow_setup.py            # MLflow + Ultralytics wiring
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install paddlepaddle==3.2.2
pip install "paddlex[ocr-core]"
pip install -r requirements.txt
```

In Cursor/VS Code, select the `.venv` kernel before running notebooks.

If PaddleOCR crashes with `ConvertPirAttribute2RuntimeAttribute` / `onednn_instruction.cc`, reinstall the pinned CPU stack (PaddlePaddle 3.3.x breaks PP-OCRv5 on CPU):

```powershell
pip install paddlepaddle==3.2.2
pip install "paddlex[ocr-core]"
pip install "paddleocr>=3.0,<3.4"
```

## Datasets (local only)

Download from Roboflow Universe and place in the repo root (not committed to git):

- **[Thndr National Card](https://universe.roboflow.com/thndr-ovgh9/thndr-national-card/browse?queryText=class%3AID&pageSize=50&startingIndex=0&browseQuery=true)** → folder `Thndr-National-Card.v4-v4.yolov8` — card field localization (16 classes; use the **ID** class for the number field)
- **[cro4](https://universe.roboflow.com/re8/cro4)** → folder `cro4.v1-8.yolov8` — per-digit boxes on ID number strips (10 classes `0–9`, 80/10/10 split)

Export each dataset in **YOLOv8** format from Roboflow before training.

Trained weights land in `runs/<run_name>/weights/best.pt` after each notebook training run. Each train creates a timestamped run folder and logs to MLflow.

## Experiment tracking (MLflow)

Training notebooks call [`tracking/mlflow_setup.py`](tracking/mlflow_setup.py) before `model.train()` and register supplemental callbacks from [`tracking/yolo_mlflow.py`](tracking/yolo_mlflow.py). Each live run logs:

- Ultralytics MLflow callback: params, train/val losses, LR, mAP (step = `trainer.epoch`, 0-based)
- Supplemental callback: `time` only (avoids duplicate metrics that caused stair-step charts)
- Full `runs/<run_name>/` tree at end of training

Backfill (`backfill_mlflow_runs`) logs every `results.csv` column once from disk.

Experiments:

- `nid-localization` — [`train_nid_yolo.ipynb`](train_nid_yolo.ipynb)
- `nid-digits` — [`train_nid_digits.ipynb`](train_nid_digits.ipynb)

Backfill historical runs that were trained before MLflow was enabled:

```powershell
.\.venv\Scripts\python -m tracking.backfill_mlflow_runs
```

This logs `runs/nid_localize` (47 epochs) and `runs/nid_digits` (5 epochs, interrupted) into the experiments above. Use `--force` to re-log if a run name already exists.

View and compare runs (use the project venv — MLflow 3 no longer supports the old `mlruns/` file store):

```powershell
.\.venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlartifacts
```

Open http://127.0.0.1:5000 in your browser. `mlflow.db` and `mlartifacts/` are gitignored.

## Train

1. `train_nid_yolo.ipynb` — localization on Thndr → `runs/nid_localize_<timestamp>/weights/best.pt`
2. `train_nid_digits.ipynb` — digits on cro4 → `runs/nid_digits_<timestamp>/weights/best.pt`

Digit training uses `fliplr=0` (digits are left-to-right).

## Infer

Full card (detect all 16 fields, digit-read `ID`, PaddleOCR the other 15):

```powershell
python yolo+ocr/run_pipeline.py --source yolo+ocr/real-samples --device 0
```

ID crops only (digit YOLO, unchanged):

```powershell
python yolo+ocr/digit_nid.py --input yolo+ocr/crops/pred/ID --device 0
```

`run_pipeline.py` writes `yolo+ocr/results.jsonl` with a `fields` map (one entry per detected class), plus top-level ID keys for backward compatibility. Crops land under `yolo+ocr/crops/pred/<ClassName>/`. `digit_nid.py` still writes `yolo+ocr/digit_results.jsonl`.

## Web UI

Local iSchool-branded UI with front/back upload and 16 editable fields:

```powershell
uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 — upload the front (required) and back (optional), click **Extract fields**, then review or fill in any missing values. After extraction, click **Submit to spreadsheet** to append one row of field values to Google Sheets (no images).

Each extraction saves artifacts under `web/uploads/<run_id>/`:

- `front.jpg` / `back.jpg` — original uploads
- `front_annotated.jpg` / `back_annotated.jpg` — detections drawn on the card
- `crops/front/<Class>.jpg` and `crops/back/<Class>.jpg` — per-field crops

## Submit to Google Sheets

After **Extract fields**, users can edit the form and click **Submit to spreadsheet**. The FastAPI server forwards the current field values to a Google Apps Script web app, which appends one row to your Sheet. Card images are not sent.

### 1. Create the Sheet and Apps Script

1. Create a Google Sheet with row 1 headers (exact names):

   `First_Name`, `Last_Name`, `HusbandName`, `Gender`, `Religion`, `Status`, `ID`, `IssueDate`, `ExpDate`, `Serial_Num`, `Add1`, `Add2`, `Job1`, `Job2`, `Front`, `Back`, `submitted_at`

2. **Extensions → Apps Script**, paste the contents of [web/sheets_apps_script.js](web/sheets_apps_script.js), and save.

3. Optionally set `WEBHOOK_TOKEN` at the top of the script to match the server token below.

4. **Deploy → New deployment → Web app**
   - Execute as: **Me**
   - Who has access: **Anyone** (required so the server can POST without Google login)

5. Copy the deployment URL ending in `/exec`.

### 2. Configure the server

Set environment variables (never commit these to git):

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_SHEETS_WEBHOOK_URL` | Yes | Apps Script web app URL (`…/exec`) |
| `GOOGLE_SHEETS_TOKEN` | No | Shared secret; must match `WEBHOOK_TOKEN` in the script |

**Local (PowerShell):**

```powershell
$env:GOOGLE_SHEETS_WEBHOOK_URL = "https://script.google.com/macros/s/.../exec"
$env:GOOGLE_SHEETS_TOKEN = "your-secret"   # optional
uvicorn web.app:app --reload --port 8000
```

**ECS Express Mode:** In the ECS console, update the service environment variables with the same names, then redeploy or restart the task.

If `GOOGLE_SHEETS_WEBHOOK_URL` is missing, `POST /api/submit` returns **503** with a clear message.

## Public demo (Amazon ECS Express Mode)

Deploy the web UI to AWS as a public HTTPS URL using [ECS Express Mode](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-overview.html). Express Mode runs on **Fargate (CPU only)** — inference is slower than a local GPU. **Delete the service when you are not demoing** to avoid ongoing Fargate + Application Load Balancer charges (~$0.20/hour compute at 4 vCPU / 8 GB plus ~$16/month for the ALB while running).

### 1. Build and push the Docker image

From the repo root (Linux amd64 image; required on Windows with Docker Desktop):

```powershell
docker build --platform linux/amd64 -t id-ocr .
```

Create an ECR repository and push (replace `REGION` and `ACCOUNT`):

```powershell
aws ecr create-repository --repository-name id-ocr --region REGION
aws ecr get-login-password --region REGION | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker tag id-ocr:latest ACCOUNT.dkr.ecr.REGION.amazonaws.com/id-ocr:latest
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/id-ocr:latest
```

The [Dockerfile](Dockerfile) installs CPU PyTorch and PaddleOCR, copies weights from `yolo+ocr/weights/`, and pre-downloads Paddle models at build time.

### 2. IAM roles

Express Mode needs two roles. Follow the [Express Mode first-run guide](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-first-run.html):

- **Task execution role** — e.g. `ecsTaskExecutionRole` (pull ECR image, write CloudWatch logs)
- **Infrastructure role** — e.g. `ecsInfrastructureRoleForExpressServices` (create ALB, target groups, etc.)

### 3. Create the Express Mode service

```bash
aws ecs create-express-gateway-service \
  --service-name id-ocr \
  --execution-role-arn arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole \
  --infrastructure-role-arn arn:aws:iam::ACCOUNT:role/ecsInfrastructureRoleForExpressServices \
  --cpu 4096 \
  --memory 8192 \
  --health-check-path /health \
  --scaling-target minTaskCount=1,maxTaskCount=1 \
  --primary-container image=ACCOUNT.dkr.ecr.REGION.amazonaws.com/id-ocr:latest,containerPort=8080
```

The CLI returns a public HTTPS URL (`*.ecs.<region>.on.aws`). The container listens on **8080** and exposes `/health` and `/ping` immediately while models load in the background (1–2 minutes on CPU). `/api/extract` returns 503 until `ready` is true.

### 4. Update an existing Express service

After code changes (e.g. new UI or API routes), rebuild and push the image, set any new environment variables, then roll the task:

```powershell
docker build --platform linux/amd64 -t id-ocr .
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.eu-west-1.amazonaws.com
docker tag id-ocr:latest ACCOUNT.dkr.ecr.eu-west-1.amazonaws.com/id-ocr:latest
docker push ACCOUNT.dkr.ecr.eu-west-1.amazonaws.com/id-ocr:latest
```

Add or update container environment variables in the ECS console (**Update service**), or via CLI:

```powershell
aws ecs update-express-gateway-service `
  --service-arn arn:aws:ecs:REGION:ACCOUNT:service/default/SERVICE_NAME `
  --primary-container "environment=[{name=GOOGLE_SHEETS_WEBHOOK_URL,value=YOUR_APPS_SCRIPT_EXEC_URL}]" `
  --region REGION
```

Updating the service starts a rolling deployment that pulls the new `:latest` image. Wait for `/health` → `{"ready":true}` before demoing.

Secrets such as `GOOGLE_SHEETS_WEBHOOK_URL` belong in ECS environment variables or Secrets Manager — not in the Docker image or git.

### 5. Stop billing

Delete the Express Mode service from the ECS console (or CLI) when the demo is over. Uploaded artifacts live on ephemeral task disk and are lost when the task stops.

If the 8 GB task runs out of memory, deploy the same image as a standard Fargate service with **4 vCPU / 16 GB** instead of Express Mode.


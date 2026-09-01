# CPU inference image for Amazon ECS Express Mode (Fargate has no GPU).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    FLAGS_use_mkldnn=0 \
    FLAGS_enable_pir_api=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir paddlepaddle==3.2.2 \
    && pip install --no-cache-dir "paddlex[ocr-core]"

RUN pip install --no-cache-dir \
    "ultralytics>=8.3.0" \
    "opencv-python-headless>=4.10.0" \
    "matplotlib>=3.9.0" \
    "pyyaml>=6.0" \
    "pillow>=10.0.0" \
    "tqdm>=4.66.0" \
    "paddleocr>=3.0,<3.4" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "python-multipart>=0.0.12" \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall "opencv-python-headless>=4.10.0"

COPY nid_localization.yaml nid_digits.yaml ./
COPY yolo+ocr/ yolo+ocr/
COPY web/ web/

RUN python -c "\
import sys; sys.path.insert(0, 'yolo+ocr'); \
from field_ocr import get_field_ocr_reader; \
get_field_ocr_reader('ar'); \
print('PaddleOCR models cached')"

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-8080}"]

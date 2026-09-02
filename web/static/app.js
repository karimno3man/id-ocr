const FIELD_NAMES = [
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
  "Governorate",
  "Job1",
  "Job2",
  "Front",
  "Back",
];

const frontInput = document.getElementById("front-input");
const backInput = document.getElementById("back-input");
const frontDrop = document.getElementById("front-drop");
const backDrop = document.getElementById("back-drop");
const frontPreviewWrap = document.getElementById("front-preview-wrap");
const backPreviewWrap = document.getElementById("back-preview-wrap");
const extractBtn = document.getElementById("extract-btn");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const nidDecodePanel = document.getElementById("nid-decode-panel");
const artifactsSection = document.getElementById("artifacts-section");
const artifactsRunEl = document.getElementById("artifacts-run");
const frontAnnotatedImg = document.getElementById("front-annotated");
const backAnnotatedImg = document.getElementById("back-annotated");
const frontAnnotatedLink = document.getElementById("front-annotated-link");
const backAnnotatedLink = document.getElementById("back-annotated-link");
const backArtifactCard = document.getElementById("back-artifact-card");

let frontFile = null;
let backFile = null;
let canSubmit = false;

const NID_DECODE_FIELDS = [
  { key: "birth_date", label: "Birth date" },
  { key: "birth_year", label: "Birth year" },
  { key: "birth_month", label: "Birth month" },
  { key: "birth_day", label: "Birth day" },
  { key: "governorate", label: "Governorate" },
  { key: "governorate_code", label: "Governorate code" },
  { key: "century_digit", label: "Century digit" },
  { key: "serial", label: "Serial" },
  { key: "gender", label: "Gender (from ID)" },
  { key: "check_digit", label: "Check digit" },
  {
    key: "is_valid_structure",
    label: "Valid structure",
    format: (value) => (value ? "Yes" : "No"),
  },
  {
    key: "issues",
    label: "Issues",
    format: (value) => (Array.isArray(value) && value.length ? value.join("; ") : "—"),
  },
];

function setStatus(message, kind = "") {
  if (!statusEl) {
    return;
  }
  statusEl.textContent = message;
  statusEl.className = `status${kind ? ` ${kind}` : ""}`;
}

function updateExtractButton() {
  if (extractBtn) {
    extractBtn.disabled = !frontFile;
  }
}

function setSubmitEnabled(isEnabled) {
  canSubmit = isEnabled;
  if (submitBtn) {
    submitBtn.disabled = !isEnabled;
  }
}

function collectFieldValues() {
  const values = {};
  FIELD_NAMES.forEach((name) => {
    const fieldWrap = document.querySelector(`.field[data-field="${name}"]`);
    const input = fieldWrap?.querySelector("input, textarea") || document.getElementById(name);
    values[name] = input?.value.trim() ?? "";
  });
  return values;
}

async function submitFields() {
  if (!canSubmit || !submitBtn) {
    return;
  }

  submitBtn.disabled = true;
  setStatus("Submitting to spreadsheet…");

  try {
    const response = await fetch("/api/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fields: collectFieldValues() }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || payload.error || "Submit failed");
    }
    setStatus("Submitted to spreadsheet.", "success");
  } catch (error) {
    setStatus(error.message || "Submit failed", "error");
  } finally {
    submitBtn.disabled = !canSubmit;
  }
}

function showPreview(wrap, file) {
  const url = URL.createObjectURL(file);
  wrap.classList.add("has-image");
  wrap.innerHTML = `<img src="${url}" alt="Card preview" />`;
}

const UPLOAD_PLACEHOLDER = {
  front: {
    title: "Upload front side of the card",
    label: "Upload front side image",
  },
  back: {
    title: "Upload back side of the card",
    label: "Upload back side image",
  },
};

function resetPreview(wrap, side) {
  const copy = UPLOAD_PLACEHOLDER[side];
  wrap.classList.remove("has-image", "dragover");
  wrap.setAttribute("aria-label", copy.label);
  wrap.innerHTML = `
    <span class="upload-icon">+</span>
    <strong>${copy.title}</strong>
    <span class="upload-hint">Click or drag an image here</span>
  `;
}

function setSideFile(file, side) {
  if (!file || !file.type.startsWith("image/")) {
    return;
  }
  const wrap = side === "front" ? frontPreviewWrap : backPreviewWrap;
  if (side === "front") {
    frontFile = file;
    setSubmitEnabled(false);
  } else {
    backFile = file;
  }
  showPreview(wrap, file);
  updateExtractButton();
}

function bindUpload(input, wrap, side) {
  const setFile = (file) => setSideFile(file, side);

  wrap.addEventListener("click", () => input.click());
  wrap.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });

  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) {
      setFile(file);
    }
  });

  const dropTarget = side === "front" ? frontDrop : backDrop;
  ["dragenter", "dragover"].forEach((eventName) => {
    dropTarget.addEventListener(eventName, (event) => {
      event.preventDefault();
      wrap.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropTarget.addEventListener(eventName, (event) => {
      event.preventDefault();
      wrap.classList.remove("dragover");
      if (eventName === "drop") {
        const file = event.dataTransfer?.files?.[0];
        if (file) {
          setFile(file);
        }
      }
    });
  });
}

function formatNidValue(nidMeta, field) {
  const raw = nidMeta?.[field.key];
  if (field.format) {
    return field.format(raw);
  }
  if (raw == null || raw === "") {
    return "—";
  }
  return String(raw);
}

function populateNidDecode(nidMeta) {
  if (!nidDecodePanel) {
    return;
  }

  NID_DECODE_FIELDS.forEach((field) => {
    const input = document.getElementById(`nid-${field.key}`);
    if (!input) {
      return;
    }
    input.value = nidMeta ? formatNidValue(nidMeta, field) : "—";
  });

  nidDecodePanel.classList.toggle("hidden", !nidMeta);
}

function showArtifacts(artifacts) {
  if (!artifactsSection) {
    return;
  }

  if (!artifacts) {
    artifactsSection.classList.add("hidden");
    return;
  }

  artifactsSection.classList.remove("hidden");
  if (artifactsRunEl) {
    artifactsRunEl.textContent = `Saved to ${artifacts.directory || artifacts.run_id}`;
  }

  const frontUrl = artifacts.front?.original;
  if (frontUrl && frontAnnotatedImg && frontAnnotatedLink) {
    frontAnnotatedImg.src = frontUrl;
    frontAnnotatedLink.href = frontUrl;
  }

  const backUrl = artifacts.back?.original;
  if (backUrl && backAnnotatedImg && backAnnotatedLink && backArtifactCard) {
    backArtifactCard.classList.remove("hidden");
    backAnnotatedImg.src = backUrl;
    backAnnotatedLink.href = backUrl;
  } else if (backArtifactCard) {
    backArtifactCard.classList.add("hidden");
  }
}

function populateFields(fields, meta) {
  FIELD_NAMES.forEach((name) => {
    const fieldWrap = document.querySelector(`.field[data-field="${name}"]`);
    const input = fieldWrap?.querySelector("input, textarea") || document.getElementById(name);
    if (!input) {
      return;
    }

    const helper = fieldWrap?.querySelector(".helper");
    const value = fields[name] ?? "";
    input.value = value;

    const fieldMeta = meta?.[name] ?? {};
    fieldWrap?.classList.toggle("missing", !value.trim());

    if (!helper) {
      return;
    }

    if (!value.trim()) {
      helper.textContent = "Not detected — please fill in";
      helper.className = "helper warn";
    } else if (fieldMeta.split_from) {
      const conf =
        fieldMeta.det_conf != null
          ? ` (conf ${Math.round(fieldMeta.det_conf * 100)}%)`
          : "";
      helper.textContent = `Split from ${fieldMeta.split_from}${conf}`;
      helper.className = "helper info";
    } else if (fieldMeta.source) {
      const conf =
        fieldMeta.det_conf != null
          ? ` (conf ${Math.round(fieldMeta.det_conf * 100)}%)`
          : "";
      helper.textContent = `From ${fieldMeta.source}${conf}`;
      helper.className = "helper info";
    } else {
      helper.textContent = "";
      helper.className = "helper";
    }
  });

  populateNidDecode(meta?.ID?.nid ?? null);
}

async function extractFields() {
  if (!frontFile) {
    return;
  }

  extractBtn.disabled = true;
  setSubmitEnabled(false);
  setStatus("Extracting fields… this may take a minute on first run.");

  const formData = new FormData();
  formData.append("front", frontFile);
  if (backFile) {
    formData.append("back", backFile);
  }

  try {
    const response = await fetch("/api/extract", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Extraction failed");
    }
    populateFields(payload.fields, payload.meta);
    showArtifacts(payload.artifacts);
    setSubmitEnabled(true);
    setStatus("Extraction complete. Review and edit any fields below.", "success");
  } catch (error) {
    setSubmitEnabled(false);
    setStatus(error.message || "Extraction failed", "error");
  } finally {
    updateExtractButton();
  }
}

bindUpload(frontInput, frontPreviewWrap, "front");
bindUpload(backInput, backPreviewWrap, "back");
extractBtn.addEventListener("click", extractFields);
submitBtn?.addEventListener("click", submitFields);

// --- ID card scanner (camera + frame crop + quality gate) ---

const CARD_ASPECT = 85.6 / 54;
const SCAN_QUALITY = {
  minSharpnessLive: 3.5,
  minSharpnessCapture: 4.0,
  minBrightness: 35,
  maxBrightness: 235,
  minCropWidth: 480,
  stableFrames: 3,
  sampleWidth: 240,
};

const SCAN_FEEDBACK = {
  align: "Align the card inside the frame",
  dark: "Too dark — move to better light",
  bright: "Too bright — reduce glare",
  blur: "Hold steady — image is blurry",
  ready: "Ready — tap to capture",
};

const scannerModal = document.getElementById("scanner-modal");
const scannerVideo = document.getElementById("scanner-video");
const scannerFrame = document.getElementById("scanner-frame");
const scannerCanvas = document.getElementById("scanner-canvas");
const scannerQualityCanvas = document.getElementById("scanner-quality-canvas");
const scannerInstruction = document.getElementById("scanner-instruction");
const scannerFeedback = document.getElementById("scanner-feedback");
const scannerLive = document.getElementById("scanner-live");
const scannerPreviewWrap = document.getElementById("scanner-preview-wrap");
const scannerPreviewImg = document.getElementById("scanner-preview");
const scannerPreviewStatus = document.getElementById("scanner-preview-status");
const scannerCloseBtn = document.getElementById("scanner-close");
const scannerCaptureBtn = document.getElementById("scanner-capture");
const scannerRetryBtn = document.getElementById("scanner-retry");
const scannerUseBtn = document.getElementById("scanner-use");

let scannerSide = "front";
let scannerStream = null;
let scannerQualityLoopId = null;
let scannerGoodFrameCount = 0;
let scannerCaptureReady = false;

function getCoverTransform(videoEl) {
  const displayW = videoEl.clientWidth;
  const displayH = videoEl.clientHeight;
  const videoW = videoEl.videoWidth;
  const videoH = videoEl.videoHeight;
  if (!videoW || !videoH || !displayW || !displayH) {
    return null;
  }
  const scale = Math.max(displayW / videoW, displayH / videoH);
  const scaledW = videoW * scale;
  const scaledH = videoH * scale;
  return {
    scale,
    offsetX: (displayW - scaledW) / 2,
    offsetY: (displayH - scaledH) / 2,
    videoW,
    videoH,
  };
}

function getFrameCropRect(videoEl, frameEl) {
  const transform = getCoverTransform(videoEl);
  if (!transform) {
    return null;
  }

  const videoRect = videoEl.getBoundingClientRect();
  const frameRect = frameEl.getBoundingClientRect();
  const relX = frameRect.left - videoRect.left - transform.offsetX;
  const relY = frameRect.top - videoRect.top - transform.offsetY;
  const srcX = Math.max(0, relX / transform.scale);
  const srcY = Math.max(0, relY / transform.scale);
  const srcW = Math.min(transform.videoW - srcX, frameRect.width / transform.scale);
  const srcH = Math.min(transform.videoH - srcY, frameRect.height / transform.scale);

  if (srcW <= 0 || srcH <= 0) {
    return null;
  }

  return {
    srcX,
    srcY,
    srcW,
    srcH,
    cropWidth: Math.round(srcW),
    cropHeight: Math.round(srcH),
  };
}

function drawFrameCropToCanvas(videoEl, frameEl, canvas, targetWidth) {
  const rect = getFrameCropRect(videoEl, frameEl);
  if (!rect || !canvas) {
    return null;
  }

  const aspect = rect.srcW / rect.srcH;
  const outW = targetWidth || Math.round(rect.srcW);
  const outH = Math.round(outW / aspect);
  canvas.width = outW;
  canvas.height = outH;

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(videoEl, rect.srcX, rect.srcY, rect.srcW, rect.srcH, 0, 0, outW, outH);
  return { width: outW, height: outH, cropWidth: rect.cropWidth };
}

function computeImageQuality(imageData) {
  const { data, width, height } = imageData;
  const step = 2;

  const lumAt = (x, y) => {
    const i = (y * width + x) * 4;
    return 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  };

  let brightnessSum = 0;
  let brightnessCount = 0;
  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      brightnessSum += lumAt(x, y);
      brightnessCount += 1;
    }
  }
  const brightness = brightnessSum / brightnessCount;

  let lapAbsSum = 0;
  let lapCount = 0;
  for (let y = step; y < height - step; y += step) {
    for (let x = step; x < width - step; x += step) {
      const lap =
        -4 * lumAt(x, y) +
        lumAt(x - step, y) +
        lumAt(x + step, y) +
        lumAt(x, y - step) +
        lumAt(x, y + step);
      lapAbsSum += Math.abs(lap);
      lapCount += 1;
    }
  }

  const sharpness = lapCount > 0 ? lapAbsSum / lapCount : 0;
  return { sharpness, brightness };
}

function evaluateScanQuality(metrics, mode = "live") {
  if (mode === "capture" && metrics.cropWidth < SCAN_QUALITY.minCropWidth) {
    return { ok: false, reason: "align" };
  }
  if (metrics.brightness < SCAN_QUALITY.minBrightness) {
    return { ok: false, reason: "dark" };
  }
  if (metrics.brightness > SCAN_QUALITY.maxBrightness) {
    return { ok: false, reason: "bright" };
  }
  const minSharpness =
    mode === "capture"
      ? SCAN_QUALITY.minSharpnessCapture
      : SCAN_QUALITY.minSharpnessLive;
  if (metrics.sharpness < minSharpness) {
    return { ok: false, reason: "blur" };
  }
  return { ok: true, reason: "ready" };
}

function sampleLiveQuality() {
  if (!scannerVideo || !scannerFrame || !scannerQualityCanvas) {
    return null;
  }
  if (scannerVideo.readyState < 2) {
    return null;
  }

  const drawn = drawFrameCropToCanvas(
    scannerVideo,
    scannerFrame,
    scannerQualityCanvas,
    SCAN_QUALITY.sampleWidth,
  );
  if (!drawn) {
    return null;
  }

  const ctx = scannerQualityCanvas.getContext("2d", { willReadFrequently: true });
  const imageData = ctx.getImageData(0, 0, drawn.width, drawn.height);
  const quality = computeImageQuality(imageData);
  return { ...quality, cropWidth: drawn.cropWidth };
}

function sampleCanvasQuality(canvas) {
  if (!canvas || canvas.width <= 0 || canvas.height <= 0) {
    return null;
  }
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const quality = computeImageQuality(imageData);
  return { ...quality, cropWidth: canvas.width };
}

function setScannerFeedback(reason) {
  const message = SCAN_FEEDBACK[reason] || SCAN_FEEDBACK.align;
  if (scannerFeedback) {
    scannerFeedback.textContent = message;
    scannerFeedback.classList.toggle("is-ready", reason === "ready");
  }
  scannerFrame?.classList.toggle("is-ready", reason === "ready");
  scannerFrame?.classList.toggle("is-waiting", reason !== "ready");
}

function setCaptureReady(isReady) {
  scannerCaptureReady = isReady;
  if (scannerCaptureBtn) {
    scannerCaptureBtn.disabled = !isReady;
    scannerCaptureBtn.setAttribute("aria-disabled", String(!isReady));
  }
}

function stopQualityLoop() {
  if (scannerQualityLoopId != null) {
    cancelAnimationFrame(scannerQualityLoopId);
    scannerQualityLoopId = null;
  }
  scannerGoodFrameCount = 0;
  setCaptureReady(false);
}

function qualityLoopTick() {
  if (!scannerLive || scannerLive.classList.contains("hidden")) {
    stopQualityLoop();
    return;
  }

  const metrics = sampleLiveQuality();
  if (!metrics) {
    setScannerFeedback("align");
    setCaptureReady(false);
    scannerGoodFrameCount = 0;
    scannerQualityLoopId = requestAnimationFrame(qualityLoopTick);
    return;
  }

  const result = evaluateScanQuality(metrics, "live");
  setScannerFeedback(result.reason);

  if (result.ok) {
    scannerGoodFrameCount += 1;
    if (scannerGoodFrameCount >= SCAN_QUALITY.stableFrames) {
      setCaptureReady(true);
    }
  } else {
    scannerGoodFrameCount = 0;
    setCaptureReady(false);
  }

  scannerQualityLoopId = requestAnimationFrame(qualityLoopTick);
}

function startQualityLoop() {
  stopQualityLoop();
  setScannerFeedback("align");
  scannerQualityLoopId = requestAnimationFrame(qualityLoopTick);
}

function captureFramedCrop() {
  if (!scannerVideo || !scannerFrame || !scannerCanvas) {
    return null;
  }

  const rect = getFrameCropRect(scannerVideo, scannerFrame);
  if (!rect) {
    return null;
  }

  const targetAspect = CARD_ASPECT;
  let outW = Math.round(rect.srcW);
  let outH = Math.round(outW / targetAspect);
  if (Math.abs(rect.srcH - outH) > 2) {
    outH = Math.round(rect.srcH);
    outW = Math.round(outH * targetAspect);
  }

  scannerCanvas.width = outW;
  scannerCanvas.height = outH;
  const ctx = scannerCanvas.getContext("2d");
  ctx.drawImage(
    scannerVideo,
    rect.srcX,
    rect.srcY,
    rect.srcW,
    rect.srcH,
    0,
    0,
    outW,
    outH,
  );
  return scannerCanvas.toDataURL("image/jpeg", 0.92);
}

function stopScannerStream() {
  stopQualityLoop();
  if (!scannerStream) {
    return;
  }
  scannerStream.getTracks().forEach((track) => track.stop());
  scannerStream = null;
  if (scannerVideo) {
    scannerVideo.srcObject = null;
  }
}

function resetPreviewActions() {
  if (scannerUseBtn) {
    scannerUseBtn.disabled = true;
  }
  if (scannerRetryBtn) {
    scannerRetryBtn.classList.remove("is-highlight");
  }
  if (scannerPreviewStatus) {
    scannerPreviewStatus.textContent = "";
    scannerPreviewStatus.className = "scanner-preview-status";
  }
}

function showScannerLive() {
  stopQualityLoop();
  scannerLive?.classList.remove("hidden");
  scannerPreviewWrap?.classList.add("hidden");
  resetPreviewActions();
  if (scannerPreviewImg) {
    scannerPreviewImg.removeAttribute("src");
  }
  setScannerFeedback("align");
}

function showScannerPreview(dataUrl, postCheck) {
  stopQualityLoop();
  scannerLive?.classList.add("hidden");
  scannerPreviewWrap?.classList.remove("hidden");
  resetPreviewActions();

  if (scannerPreviewImg) {
    scannerPreviewImg.src = dataUrl;
  }

  if (!postCheck) {
    return;
  }

  if (postCheck.ok) {
    if (scannerPreviewStatus) {
      scannerPreviewStatus.textContent = "Looks good — use this photo?";
      scannerPreviewStatus.className = "scanner-preview-status is-ok";
    }
    if (scannerUseBtn) {
      scannerUseBtn.disabled = false;
    }
    return;
  }

  const message = SCAN_FEEDBACK[postCheck.reason] || "Photo not clear enough — retry";
  if (scannerPreviewStatus) {
    scannerPreviewStatus.textContent = `Photo not clear enough — ${message.toLowerCase()}`;
    scannerPreviewStatus.className = "scanner-preview-status is-bad";
  }
  if (scannerRetryBtn) {
    scannerRetryBtn.classList.add("is-highlight");
    scannerRetryBtn.focus();
  }
}

async function startScannerCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Camera is not supported in this browser.");
  }

  stopScannerStream();
  scannerStream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      focusMode: { ideal: "continuous" },
    },
    audio: false,
  });

  if (scannerVideo) {
    scannerVideo.srcObject = scannerStream;
    await scannerVideo.play();
  }
}

async function openScanner(side) {
  scannerSide = side;
  const label = side === "front" ? "Scan the front side" : "Scan the back side";
  if (scannerInstruction) {
    scannerInstruction.textContent = label;
  }

  scannerModal?.classList.remove("hidden");
  scannerModal?.setAttribute("aria-hidden", "false");
  document.body.classList.add("scanner-open");
  showScannerLive();

  try {
    await startScannerCamera();
    startQualityLoop();
  } catch (error) {
    closeScanner();
    setStatus(error.message || "Could not access camera.", "error");
  }
}

function closeScanner() {
  stopScannerStream();
  scannerModal?.classList.add("hidden");
  scannerModal?.setAttribute("aria-hidden", "true");
  document.body.classList.remove("scanner-open");
  scannerLive?.classList.remove("hidden");
  scannerPreviewWrap?.classList.add("hidden");
  resetPreviewActions();
}

function dataUrlToFile(dataUrl, filename) {
  const [header, base64] = dataUrl.split(",");
  const mime = header.match(/:(.*?);/)?.[1] || "image/jpeg";
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new File([bytes], filename, { type: mime });
}

document.querySelectorAll("[data-scan-side]").forEach((button) => {
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    const side = button.getAttribute("data-scan-side");
    if (side === "front" || side === "back") {
      openScanner(side);
    }
  });
});

scannerCloseBtn?.addEventListener("click", closeScanner);
scannerModal?.addEventListener("click", (event) => {
  if (event.target === scannerModal) {
    closeScanner();
  }
});

scannerCaptureBtn?.addEventListener("click", () => {
  if (!scannerCaptureReady) {
    return;
  }

  const dataUrl = captureFramedCrop();
  if (!dataUrl) {
    setStatus("Could not capture image. Try again.", "error");
    return;
  }

  const postMetrics = sampleCanvasQuality(scannerCanvas);
  const postCheck = postMetrics
    ? evaluateScanQuality(postMetrics, "capture")
    : { ok: false, reason: "blur" };

  showScannerPreview(dataUrl, postCheck);
});

scannerRetryBtn?.addEventListener("click", () => {
  showScannerLive();
  startQualityLoop();
});

scannerUseBtn?.addEventListener("click", () => {
  if (scannerUseBtn?.disabled) {
    return;
  }

  const dataUrl = scannerPreviewImg?.src;
  if (!dataUrl) {
    return;
  }
  const filename = `${scannerSide}-scan-${Date.now()}.jpg`;
  const file = dataUrlToFile(dataUrl, filename);
  setSideFile(file, scannerSide);
  closeScanner();
  setStatus(`${scannerSide === "front" ? "Front" : "Back"} photo saved.`, "success");
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && scannerModal && !scannerModal.classList.contains("hidden")) {
    closeScanner();
  }
});

FIELD_NAMES.forEach((name) => {
  const fieldWrap = document.querySelector(`.field[data-field="${name}"]`);
  const input = fieldWrap?.querySelector("input, textarea") || document.getElementById(name);
  if (!input) {
    return;
  }

  input.addEventListener("input", () => {
    fieldWrap?.classList.toggle("missing", !input.value.trim());
    const helper = fieldWrap?.querySelector(".helper");
    if (helper && input.value.trim()) {
      helper.textContent = "";
      helper.className = "helper";
    }
  });
});

fetch("/health")
  .then((response) => response.json())
  .then((payload) => {
    if (!payload.ready) {
      setStatus("Loading models…");
    }
  })
  .catch(() => {
    setStatus("Waiting for server…");
  });

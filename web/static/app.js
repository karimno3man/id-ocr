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

function bindUpload(input, wrap, side) {
  const setFile = (file) => {
    if (!file || !file.type.startsWith("image/")) {
      return;
    }
    if (side === "front") {
      frontFile = file;
      setSubmitEnabled(false);
    } else {
      backFile = file;
    }
    showPreview(wrap, file);
    updateExtractButton();
  };

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

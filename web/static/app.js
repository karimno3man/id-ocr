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
const statusEl = document.getElementById("status");
const nidDecodeEl = document.getElementById("nid-decode");
const artifactsSection = document.getElementById("artifacts-section");
const artifactsRunEl = document.getElementById("artifacts-run");
const frontAnnotatedImg = document.getElementById("front-annotated");
const backAnnotatedImg = document.getElementById("back-annotated");
const frontAnnotatedLink = document.getElementById("front-annotated-link");
const backAnnotatedLink = document.getElementById("back-annotated-link");
const backArtifactCard = document.getElementById("back-artifact-card");

let frontFile = null;
let backFile = null;

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.className = `status${kind ? ` ${kind}` : ""}`;
}

function updateExtractButton() {
  extractBtn.disabled = !frontFile;
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

function formatNidSummary(nidMeta) {
  if (!nidMeta) {
    return "";
  }
  const parts = [];
  if (nidMeta.birth_date) {
    parts.push(`Birth: ${nidMeta.birth_date}`);
  }
  if (nidMeta.governorate) {
    parts.push(`Gov: ${nidMeta.governorate}`);
  }
  if (nidMeta.gender) {
    parts.push(`Gender: ${nidMeta.gender}`);
  }
  return parts.join(" · ");
}

function showArtifacts(artifacts) {
  if (!artifacts) {
    artifactsSection.classList.add("hidden");
    return;
  }

  artifactsSection.classList.remove("hidden");
  artifactsRunEl.textContent = `Saved to ${artifacts.directory || artifacts.run_id}`;

  const frontUrl = artifacts.front?.original;
  if (frontUrl) {
    frontAnnotatedImg.src = frontUrl;
    frontAnnotatedLink.href = frontUrl;
  }

  const backUrl = artifacts.back?.original;
  if (backUrl) {
    backArtifactCard.classList.remove("hidden");
    backAnnotatedImg.src = backUrl;
    backAnnotatedLink.href = backUrl;
  } else {
    backArtifactCard.classList.add("hidden");
  }
}

function populateFields(fields, meta) {
  FIELD_NAMES.forEach((name) => {
    const input = document.getElementById(name);
    const fieldWrap = document.querySelector(`.field[data-field="${name}"]`);
    const helper = fieldWrap?.querySelector(".helper");
    const value = fields[name] ?? "";
    input.value = value;

    const fieldMeta = meta?.[name] ?? {};
    fieldWrap?.classList.toggle("missing", !value.trim());

    if (helper && name !== "ID") {
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
    }
  });

  const idMeta = meta?.ID?.nid;
  if (idMeta) {
    nidDecodeEl.textContent = formatNidSummary(idMeta);
    nidDecodeEl.className = "helper info";
  } else {
    nidDecodeEl.textContent = "";
    nidDecodeEl.className = "helper";
  }
}

async function extractFields() {
  if (!frontFile) {
    return;
  }

  extractBtn.disabled = true;
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
    setStatus("Extraction complete. Review and edit any fields below.", "success");
  } catch (error) {
    setStatus(error.message || "Extraction failed", "error");
  } finally {
    updateExtractButton();
  }
}

bindUpload(frontInput, frontPreviewWrap, "front");
bindUpload(backInput, backPreviewWrap, "back");
extractBtn.addEventListener("click", extractFields);

FIELD_NAMES.forEach((name) => {
  const input = document.getElementById(name);
  input.addEventListener("input", () => {
    const fieldWrap = document.querySelector(`.field[data-field="${name}"]`);
    fieldWrap?.classList.toggle("missing", !input.value.trim());
    if (name !== "ID") {
      const helper = fieldWrap?.querySelector(".helper");
      if (helper && input.value.trim()) {
        helper.textContent = "";
        helper.className = "helper";
      }
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

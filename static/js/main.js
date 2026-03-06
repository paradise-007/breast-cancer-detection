/**
 * MammoScan AI — Frontend Logic
 *
 * Handles:
 *   - Drag-and-drop / click-to-upload image selection
 *   - Preview rendering
 *   - POST /predict with FormData
 *   - Rendering results per model
 *   - Model card switching
 *   - Comparison table
 */

"use strict";

// ─── Model Config ───────────────────────────────────
const MODEL_COLORS = {
  VGG16:       "#e84393",
  ResNet50V2:  "#00c9a7",
  InceptionV3: "#f59e0b",
};

// ─── State ──────────────────────────────────────────
let currentFile   = null;
let allResults    = null;
let activeModel   = "VGG16";
let isAnalyzing   = false;

// ─── DOM Refs ────────────────────────────────────────
const dropzone      = document.getElementById("dropzone");
const dropzoneInner = document.getElementById("dropzone-inner");
const fileInput     = document.getElementById("file-input");
const previewImg    = document.getElementById("preview-img");
const scanOverlay   = document.getElementById("scan-overlay");
const imgBadge      = document.getElementById("img-badge");
const badgeDot      = document.getElementById("badge-dot");
const badgeText     = document.getElementById("badge-text");
const changeBtn     = document.getElementById("change-btn");
const btnAnalyze    = document.getElementById("btn-analyze");
const progressWrap  = document.getElementById("progress-wrap");
const progressBar   = document.getElementById("progress-bar");
const modelCards    = document.querySelectorAll(".model-card");
const resultCard    = document.getElementById("result-card");
const howItWorks    = document.getElementById("how-it-works");
const compareTable  = document.getElementById("compare-table");
const compareTbody  = document.getElementById("compare-tbody");

// ─── Result DOM ──────────────────────────────────────
const ringFill        = document.getElementById("ring-fill");
const ringText        = document.getElementById("ring-text");
const resultLabel     = document.getElementById("result-label");
const metricPrecision = document.getElementById("metric-precision");
const metricRecall    = document.getElementById("metric-recall");
const metricF1        = document.getElementById("metric-f1");
const verdictEl       = document.getElementById("verdict");
const modeBadge       = document.getElementById("mode-badge");

// ─── File Handling ───────────────────────────────────
dropzone.addEventListener("click", (e) => {
  if (e.target === changeBtn || isAnalyzing) return;
  if (!currentFile || e.target === dropzoneInner || dropzoneInner.contains(e.target)) {
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

changeBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});

function loadFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please upload an image file (PNG, JPG, JPEG, TIFF, BMP).");
    return;
  }
  currentFile = file;
  allResults  = null;

  const url = URL.createObjectURL(file);
  previewImg.src = url;

  // Show image, hide placeholder
  dropzoneInner.classList.add("hidden");
  previewImg.classList.remove("hidden");
  imgBadge.classList.remove("hidden");
  changeBtn.classList.remove("hidden");
  dropzone.classList.add("has-image");

  badgeDot.classList.remove("pulsing");
  badgeText.textContent = "IMAGE LOADED";

  // Show analyze button, reset results
  btnAnalyze.classList.remove("hidden");
  resetResults();
  progressWrap.classList.add("hidden");
  progressBar.style.width = "0%";
}

// ─── Model Card Switching ────────────────────────────
modelCards.forEach((card) => {
  card.addEventListener("click", () => {
    activeModel = card.dataset.model;
    modelCards.forEach((c) => {
      c.classList.remove("active");
      c.style.borderColor = "";
      c.style.background  = "";
    });
    card.classList.add("active");
    const color = MODEL_COLORS[activeModel];
    card.style.borderColor = color;
    card.style.background  = `${color}18`;

    if (allResults) renderResult(activeModel);
  });
});

// ─── Analyze ─────────────────────────────────────────
btnAnalyze.addEventListener("click", analyze);

async function analyze() {
  if (!currentFile || isAnalyzing) return;
  isAnalyzing = true;

  // UI: loading state
  btnAnalyze.disabled = true;
  btnAnalyze.innerHTML = `<span class="spinner"></span> Analyzing…`;
  badgeDot.classList.add("pulsing");
  badgeText.textContent = "ANALYZING...";
  scanOverlay.classList.remove("hidden");
  progressWrap.classList.remove("hidden");

  // Fake progress animation
  let prog = 0;
  const ticker = setInterval(() => {
    prog = Math.min(prog + Math.random() * 8, 90);
    progressBar.style.width = prog + "%";
  }, 120);

  // Build form data
  const formData = new FormData();
  formData.append("file", currentFile);

  let data = null;
  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    data = await response.json();
  } catch (err) {
    clearInterval(ticker);
    stopAnalyzing();
    alert("Network error: " + err.message);
    return;
  }

  // Complete progress
  clearInterval(ticker);
  progressBar.style.width = "100%";
  await sleep(300);

  if (data.error) {
    stopAnalyzing();
    alert("Error: " + data.error);
    return;
  }

  allResults = data.results;
  stopAnalyzing();
  renderAllResults();
}

function stopAnalyzing() {
  isAnalyzing = false;
  btnAnalyze.disabled = false;
  btnAnalyze.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round">
      <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
    </svg>
    Run CNN Analysis`;
  badgeDot.classList.remove("pulsing");
  badgeText.textContent = "ANALYSIS DONE";
  scanOverlay.classList.add("hidden");
}

// ─── Render ───────────────────────────────────────────
function renderAllResults() {
  // Model card badges
  Object.entries(allResults).forEach(([model, r]) => {
    const badge = document.getElementById(`badge-${model}`);
    if (!badge) return;
    badge.classList.remove("hidden");
    badge.textContent = r.label;
    const color = r.label === "Malignant" ? "#e84393" : "#00c9a7";
    badge.style.color      = color;
    badge.style.background = `${color}22`;
    badge.style.border     = `1px solid ${color}66`;
  });

  // Show/hide sections
  resultCard.classList.remove("hidden");
  howItWorks.classList.add("hidden");
  compareTable.classList.remove("hidden");

  renderResult(activeModel);
  renderCompareTable();
}

function renderResult(modelName) {
  const r = allResults?.[modelName];
  if (!r) return;

  const color = MODEL_COLORS[modelName];
  const isMal = r.label === "Malignant";

  // Ring
  const circ = 2 * Math.PI * 30; // r=30
  const offset = circ - (r.confidence / 100) * circ;
  ringFill.style.stroke = color;
  ringFill.style.strokeDashoffset = offset;
  ringText.textContent  = r.confidence.toFixed(1) + "%";

  // Label
  resultLabel.textContent = r.label;
  resultLabel.style.color = isMal ? "#e84393" : "#00c9a7";

  // Metrics
  metricPrecision.textContent = r.precision ? r.precision.toFixed(1) + "%" : "—";
  metricRecall.textContent    = r.recall    ? r.recall.toFixed(1)    + "%" : "—";
  metricF1.textContent        = r.f1        ? r.f1.toFixed(1)        + "%" : "—";

  // Metrics color
  [metricPrecision, metricRecall, metricF1].forEach((el) => {
    el.style.color = color;
  });

  // Verdict
  verdictEl.textContent = isMal
    ? "⚠ Signs consistent with malignant tissue detected. Please consult a radiologist immediately for further evaluation."
    : "✓ No malignant patterns detected. Tissue appears benign. Regular screening is still recommended.";
  verdictEl.style.background = isMal ? "#e8439310" : "#00c9a710";
  verdictEl.style.border     = `1px solid ${isMal ? "#e8439344" : "#00c9a744"}`;

  // Demo mode badge
  if (r.mode === "demo") {
    modeBadge.classList.remove("hidden");
  } else {
    modeBadge.classList.add("hidden");
  }

  // Result card border
  resultCard.style.borderColor = `${color}44`;
}

function renderCompareTable() {
  compareTbody.innerHTML = "";
  Object.entries(allResults).forEach(([model, r]) => {
    const color = MODEL_COLORS[model];
    const labelColor = r.label === "Malignant" ? "#e84393" : "#00c9a7";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="color:${color};font-family:'DM Mono',monospace">${model}</td>
      <td style="color:${labelColor}">${r.label}</td>
      <td>${r.confidence.toFixed(1)}%</td>
      <td>${r.f1 ? r.f1.toFixed(1) + "%" : "—"}</td>
    `;
    tr.addEventListener("click", () => {
      activeModel = model;
      modelCards.forEach((c) => {
        c.classList.remove("active");
        c.style.borderColor = "";
        c.style.background  = "";
        if (c.dataset.model === model) {
          c.classList.add("active");
          c.style.borderColor = color;
          c.style.background  = `${color}18`;
        }
      });
      renderResult(model);
    });
    compareTbody.appendChild(tr);
  });
}

function resetResults() {
  allResults = null;
  resultCard.classList.add("hidden");
  compareTable.classList.add("hidden");
  howItWorks.classList.remove("hidden");
  document.querySelectorAll(".mc-badge").forEach((b) => b.classList.add("hidden"));
}

// ─── Helpers ─────────────────────────────────────────
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

"use strict";

const byId = (id) => document.getElementById(id);
const classLabels = { authentic: "Authentic", ai_generated: "AI-generated", manipulated: "Manipulated" };
const state = { status: null, file: null, previewUrl: null, result: null, frame: 0, history: [], total: 0, operation: 0, busy: false, view: "analyze", toastTimer: null };
const defaultExtensions = [".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".avi", ".webm"];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function icon(name, extraClass = "") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", `icon ${extraClass}`.trim());
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#icon-${name}`);
  svg.append(use);
  return svg;
}

function formatBytes(bytes) {
  if (!Number.isFinite(Number(bytes))) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function timecode(seconds) {
  if (!Number.isFinite(Number(seconds))) return "00:00";
  const value = Math.max(0, Number(seconds));
  const minutes = Math.floor(value / 60).toString().padStart(2, "0");
  const remainder = (value % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${remainder}`;
}

function formatDate(date) {
  const value = new Date(date);
  if (Number.isNaN(value.getTime())) return "Date unavailable";
  return new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", hour: "numeric", minute: "2-digit", timeZone: "Asia/Kolkata" }).format(value);
}

function percentage(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(Math.min(1, Math.max(0, number)) * 100).toFixed(1)}%` : "—";
}

function fileIcon(mediaType) {
  const node = element("span", "file-icon");
  node.append(icon(mediaType === "video" ? "video" : "image"));
  return node;
}

function setError(message) {
  byId("upload-error").textContent = message || "";
  byId("upload-error").hidden = !message;
}

function toast(message) {
  clearTimeout(state.toastTimer);
  byId("toast").textContent = message;
  byId("toast").hidden = false;
  state.toastTimer = setTimeout(() => { byId("toast").hidden = true; }, 5000);
}

function showView(view) {
  if (!["analyze", "history", "guide"].includes(view)) return;
  state.view = view;
  document.querySelectorAll(".view").forEach((section) => { section.hidden = section.id !== `view-${view}`; });
  document.querySelectorAll(".nav-item").forEach((button) => {
    const current = button.dataset.view === view;
    button.classList.toggle("active", current);
    if (current) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  byId("breadcrumb-view").textContent = { analyze: "MEDIA ANALYSIS", history: "ANALYSIS HISTORY", guide: "PROJECT GUIDE" }[view];
  document.title = `TruthLens AI · ${view === "history" ? "Analysis history" : view === "guide" ? "Project guide" : "Media analysis"}`;
  window.scrollTo({ top: 0, behavior: "instant" });
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { throw new Error("The workspace returned an unexpected response. Please try again."); }
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The request could not be completed.");
  return data;
}

async function checkStatus() {
  try {
    state.status = await fetchJson("/api/status");
    const ready = Boolean(state.status.model?.ready);
    const failed = state.status.model?.status === "load_failed";
    byId("model-status-text").textContent = ready ? "Trained model loaded" : failed ? "Model could not load" : "Training required";
    byId("model-status-dot").className = `status-dot${ready ? "" : " amber"}`;
    byId("sidebar-status").textContent = "Workspace connected";
    byId("sidebar-status-dot").className = "status-dot";
    byId("sidebar-model-copy").textContent = ready ? "Your detector is loaded. Inspect media and review model evidence with care." : failed ? "The checkpoint could not load. Media inspection is available; review model configuration to restore predictions." : "Prepare a dataset and train your detector to unlock predictions.";
    const limit = state.status.limits?.max_upload_bytes || 52428800;
    byId("upload-limit").textContent = `Up to ${formatBytes(limit).replace(".0", "")} per file`;
    const extensions = [...(state.status.supported?.images || []), ...(state.status.supported?.videos || [])];
    if (extensions.length) byId("file-input").accept = extensions.join(",");
  } catch {
    byId("model-status-text").textContent = "Status unavailable";
    byId("model-status-dot").className = "status-dot offline";
    byId("sidebar-status").textContent = "Workspace unavailable";
    byId("sidebar-status-dot").className = "status-dot offline";
  }
}

function releasePreview() {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
}

function clearResult() {
  state.result = null;
  byId("result-content").hidden = true;
  byId("busy-result").hidden = true;
  byId("empty-result").hidden = false;
  byId("evidence-section").hidden = true;
  byId("result-status").className = "status-pill neutral";
  byId("result-status").textContent = "Awaiting media";
}

function clearFile() {
  if (state.busy) return;
  state.operation += 1;
  releasePreview();
  state.file = null;
  byId("file-input").value = "";
  byId("media-preview").replaceChildren();
  byId("selected-media").hidden = true;
  byId("drop-zone").hidden = false;
  byId("analyze-button").disabled = true;
  byId("analyze-button-label").textContent = "Inspect media";
  clearResult();
  setError("");
}

function selectFile(file) {
  if (!file || state.busy) return;
  state.operation += 1;
  const extensions = state.status ? [...(state.status.supported?.images || []), ...(state.status.supported?.videos || [])] : defaultExtensions;
  const suffix = `.${file.name.split(".").pop().toLowerCase()}`;
  if (!(extensions.length ? extensions : defaultExtensions).includes(suffix)) {
    clearFile();
    setError("Choose a JPG, PNG, WEBP image or an MP4, MOV, AVI, WEBM video.");
    return;
  }
  const limit = state.status?.limits?.max_upload_bytes || 52428800;
  if (file.size > limit) {
    clearFile();
    setError(`This file is too large. Choose a file under ${formatBytes(limit)}.`);
    return;
  }
  if (!file.size) { clearFile(); setError("This file is empty. Choose a different image or video."); return; }
  releasePreview();
  clearResult();
  setError("");
  state.file = file;
  state.previewUrl = URL.createObjectURL(file);
  const isVideo = [".mp4", ".mov", ".avi", ".webm"].includes(suffix);
  const preview = element(isVideo ? "video" : "img");
  preview.src = state.previewUrl;
  if (isVideo) {
    preview.controls = true;
    preview.playsInline = true;
    preview.preload = "metadata";
    preview.setAttribute("aria-label", `Preview of ${file.name}`);
  } else preview.alt = `Preview of ${file.name}`;
  preview.addEventListener("error", () => {
    if (state.file === file) toast("Your browser cannot preview this format. The server can still attempt to inspect the file.");
  }, { once: true });
  byId("media-preview").replaceChildren(preview);
  byId("selected-filename").textContent = file.name;
  byId("selected-details").textContent = `${formatBytes(file.size)} · ${isVideo ? "Video" : "Image"} · ${suffix.slice(1).toUpperCase()}`;
  byId("selected-media").hidden = false;
  byId("drop-zone").hidden = true;
  byId("analyze-button").disabled = false;
  byId("analyze-button-label").textContent = "Inspect media";
  byId("result-status").textContent = "Ready to inspect";
}

function setBusy(busy) {
  state.busy = busy;
  byId("analyze-button").disabled = busy || !state.file;
  byId("analyze-button-label").textContent = busy ? "Inspecting media…" : "Inspect media";
  byId("clear-file").disabled = busy;
  byId("drop-zone").disabled = busy;
  byId("file-input").disabled = busy;
  document.querySelector(".result-panel").setAttribute("aria-busy", String(busy));
  byId("analysis-progress").hidden = !busy;
  if (busy) {
    byId("empty-result").hidden = true;
    byId("result-content").hidden = true;
    byId("busy-result").hidden = false;
    byId("evidence-section").hidden = true;
    byId("result-status").textContent = "Processing";
    byId("result-status").className = "status-pill neutral";
    byId("progress-label").textContent = "Uploading media";
    byId("progress-percent").textContent = "";
    byId("progress-bar").style.width = "0%";
    byId("progress-bar").parentElement.classList.remove("indeterminate");
  } else byId("busy-result").hidden = true;
}

function upload(file) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/analyze");
    request.responseType = "json";
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        const progress = Math.round(event.loaded / event.total * 100);
        byId("progress-bar").style.width = `${progress}%`;
        byId("progress-percent").textContent = `${progress}%`;
      }
    });
    request.upload.addEventListener("load", () => {
      byId("progress-label").textContent = "Preparing frames and measurements";
      byId("progress-percent").textContent = "";
      byId("progress-bar").parentElement.classList.add("indeterminate");
    });
    request.addEventListener("load", () => {
      if (request.status >= 200 && request.status < 300 && request.response) resolve(request.response);
      else reject(new Error(typeof request.response?.detail === "string" ? request.response.detail : "The media could not be inspected. Check the file and try again."));
    });
    request.addEventListener("error", () => reject(new Error("Cannot reach the workspace. Check that the server is running and try again.")));
    request.addEventListener("abort", () => reject(new Error("The upload was interrupted. Please try again.")));
    const form = new FormData();
    form.append("file", file);
    request.send(form);
  });
}

async function analyze() {
  if (!state.file || state.busy) return;
  state.operation += 1;
  setError("");
  setBusy(true);
  try {
    const result = await upload(state.file);
    setBusy(false);
    renderResult(result);
    await Promise.allSettled([loadHistory(), checkStatus()]);
  } catch (error) {
    setBusy(false);
    clearResult();
    setError(error.message);
    byId("result-status").className = "status-pill danger";
    byId("result-status").textContent = "Inspection failed";
  }
}

function metadataItem(label, value) {
  const row = element("div");
  row.append(element("dt", "", label), element("dd", "", value));
  return row;
}

function renderResult(result) {
  state.result = result;
  state.frame = 0;
  byId("empty-result").hidden = true;
  byId("busy-result").hidden = true;
  byId("result-content").hidden = false;
  const content = byId("result-content");
  content.replaceChildren();
  const prediction = result.prediction;
  const verdict = element("div", `verdict${prediction ? ` complete${prediction.label === "authentic" ? "" : " flagged"}` : ""}`);
  const verdictLabel = element("div", "verdict-label");
  verdictLabel.append(element("span", "", prediction ? "MODEL ASSESSMENT" : "MEDIA INSPECTION COMPLETE"), icon(prediction ? "scan" : "info"));
  verdict.append(verdictLabel);
  if (prediction) {
    verdict.append(element("h3", "", classLabels[prediction.label] || "Unrecognized model label"));
    verdict.append(element("p", "", `${percentage(prediction.confidence)} model confidence. Review the evidence and limitations before drawing a conclusion.`));
    byId("result-status").textContent = "Model assessed";
    byId("result-status").className = "status-pill success";
  } else {
    verdict.append(element("h3", "", "Detection model needed"));
    verdict.append(element("p", "", "Your media is ready to review. Train and load a model to get an authenticity assessment. No detection score has been assigned."));
    const link = element("button", "result-inline-link", "How to unlock predictions");
    link.append(icon("arrow"));
    link.addEventListener("click", () => showView("guide"));
    verdict.append(link);
    byId("result-status").textContent = "Model required";
    byId("result-status").className = "status-pill amber";
  }
  content.append(verdict);
  if (prediction?.probabilities) {
    const probabilities = element("div", "probabilities");
    Object.keys(classLabels).forEach((key) => {
      const value = prediction.probabilities[key];
      const row = element("div", `probability-row${key === prediction.label ? " winner" : ""}`);
      const label = element("div", "probability-label");
      label.append(element("span", "", classLabels[key]), element("span", "", percentage(value)));
      const track = element("div", "probability-track");
      const fill = element("div", "probability-fill");
      fill.style.width = `${Math.max(0, Math.min(1, Number(value) || 0)) * 100}%`;
      track.append(fill);
      row.append(label, track);
      probabilities.append(row);
    });
    content.append(probabilities);
  }
  content.append(element("h3", "metadata-heading", "MEDIA AT A GLANCE"));
  const media = result.media || {};
  const metadata = element("dl", "metadata-grid");
  metadata.append(
    metadataItem("Dimensions", media.width && media.height ? `${media.width.toLocaleString()} × ${media.height.toLocaleString()} px` : "—"),
    metadataItem("File size", formatBytes(media.size_bytes)),
    metadataItem("Media type", `${result.media_type === "video" ? "Video" : "Image"} · ${media.format || "Unknown"}`),
    metadataItem("Frames inspected", `${result.summary?.sampled_frames ?? result.frames?.length ?? 0}${result.media_type === "video" && media.frame_count ? ` of ${media.frame_count.toLocaleString()}` : ""}`)
  );
  if (result.media_type === "video") metadata.append(metadataItem("Duration", `${Number(media.duration_seconds || 0).toFixed(1)} seconds`), metadataItem("Frame rate", `${Number(media.fps || 0).toFixed(1)} fps`));
  content.append(metadata);
  const frames = result.frames || [];
  byId("evidence-section").hidden = !frames.length;
  byId("download-report").href = `/api/analyses/${encodeURIComponent(result.id)}/report`;
  byId("download-report").download = `truthlens-${result.id}.json`;
  byId("frame-thumbnails").replaceChildren();
  frames.forEach((frame, position) => {
    const button = element("button", "frame-thumbnail");
    button.type = "button";
    const caption = result.media_type === "video" ? timecode(frame.timestamp_seconds) : "Original";
    button.setAttribute("aria-label", `Inspect ${result.media_type === "video" ? `frame ${position + 1} at ${caption}` : "original image"}`);
    const thumbnail = element("img");
    thumbnail.src = frame.thumbnail_url;
    thumbnail.alt = "";
    thumbnail.loading = "lazy";
    button.append(thumbnail, element("span", "", caption));
    button.addEventListener("click", () => selectFrame(position));
    byId("frame-thumbnails").append(button);
  });
  if (frames.length) selectFrame(0);
  byId("analyze-button-label").textContent = "Inspect again";
}

function selectFrame(position) {
  const result = state.result;
  const frame = result?.frames?.[position];
  if (!frame) return;
  state.frame = position;
  const isVideo = result.media_type === "video";
  byId("frame-caption").textContent = isVideo ? `FRAME ${position + 1} · ${timecode(frame.timestamp_seconds)}` : "ORIGINAL IMAGE";
  byId("evidence-description").textContent = isVideo ? `${result.frames.length} frames sampled across this video` : "Image preview from the saved analysis";
  byId("heatmap-control").hidden = !frame.heatmap_url;
  if (!frame.heatmap_url) byId("heatmap-toggle").checked = false;
  updateEvidenceImage();
  document.querySelectorAll(".frame-thumbnail").forEach((button, index) => {
    button.classList.toggle("active", index === position);
    button.setAttribute("aria-pressed", String(index === position));
  });
  const quality = frame.quality || {};
  const qualityList = byId("quality-list");
  qualityList.replaceChildren();
  [
    ["Brightness", "Mean intensity · 0–255", quality.brightness],
    ["Contrast", "Intensity deviation", quality.contrast],
    ["Sharpness", "Laplacian variance", quality.sharpness]
  ].forEach(([label, note, value]) => {
    const row = element("div");
    const term = element("dt", "", label);
    term.append(element("span", "", note));
    row.append(term, element("dd", "", Number.isFinite(value) ? value.toFixed(1) : "—"));
    qualityList.append(row);
  });
  byId("face-diagnostics").hidden = !Number.isInteger(frame.face_count);
  byId("face-count").textContent = Number.isInteger(frame.face_count) ? frame.face_count : "—";
  byId("frame-prediction").hidden = !frame.prediction;
  byId("frame-prediction").replaceChildren();
  if (frame.prediction) byId("frame-prediction").append(element("span", "", "FRAME ASSESSMENT"), element("strong", "", `${classLabels[frame.prediction.label] || frame.prediction.label} · ${percentage(frame.prediction.confidence)}`));
}

function updateEvidenceImage() {
  const frame = state.result?.frames?.[state.frame];
  if (!frame) return;
  const heatmap = Boolean(frame.heatmap_url && byId("heatmap-toggle").checked);
  byId("evidence-image").src = heatmap ? frame.heatmap_url : frame.thumbnail_url;
  byId("evidence-image").alt = `${heatmap ? "Model attention visualization" : "Evidence image"} for frame ${state.frame + 1} of ${state.result.filename}`;
  byId("attention-note").hidden = !heatmap;
}

function historyStatus(analysis) {
  return analysis.prediction ? `${classLabels[analysis.prediction.label] || analysis.prediction.label} · ${percentage(analysis.prediction.confidence)}` : "Inspected · model required";
}

function emptyHistory(message) {
  const empty = element("div", "history-empty");
  const emptyIcon = element("span", "history-empty-icon");
  emptyIcon.append(icon("history"));
  empty.append(emptyIcon, element("span", "", message));
  return empty;
}

function renderHistory() {
  const analyses = state.history;
  const total = state.total;
  byId("nav-history-count").hidden = !total;
  byId("nav-history-count").textContent = total;
  byId("history-summary").textContent = `${total} saved ${total === 1 ? "analysis" : "analyses"}${total > analyses.length ? ` · showing latest ${analyses.length}` : ""}`;
  byId("history-search").placeholder = total > analyses.length ? "Search recent analyses" : "Find a filename";
  byId("history-search").setAttribute("aria-label", total > analyses.length ? `Search the ${analyses.length} most recent analyses` : "Search analysis history");
  const recent = byId("recent-history");
  recent.replaceChildren();
  analyses.slice(0, 3).forEach((analysis) => {
    const card = element("button", "recent-card");
    card.type = "button";
    card.setAttribute("aria-label", `Open analysis for ${analysis.filename}`);
    const info = element("div", "recent-card-info");
    info.append(element("div", "recent-card-title", analysis.filename), element("div", "recent-card-date", formatDate(analysis.created_at)), element("div", "recent-card-status", historyStatus(analysis)));
    card.append(fileIcon(analysis.media_type), info, icon("chevron", "icon-chevron"));
    card.addEventListener("click", () => openAnalysis(analysis.id));
    recent.append(card);
  });
  if (!analyses.length) recent.append(emptyHistory("No analyses yet. Your first inspection will appear here."));
  const search = byId("history-search").value.trim().toLowerCase();
  const filtered = analyses.filter((analysis) => analysis.filename.toLowerCase().includes(search));
  const full = byId("full-history");
  full.replaceChildren();
  filtered.forEach((analysis) => {
    const row = element("article", "history-row");
    const info = element("div", "history-row-info");
    info.append(element("h2", "history-row-title", analysis.filename), element("p", "history-row-date", `${formatDate(analysis.created_at)} · ${analysis.media_type === "video" ? "Video" : "Image"} · ${formatBytes(analysis.media?.size_bytes)}`));
    const status = element("span", `status-pill ${analysis.prediction ? "success" : "amber"}`, analysis.prediction ? classLabels[analysis.prediction.label] || "Assessed" : "Model required");
    const actions = element("div", "history-row-actions");
    const view = element("button", "secondary-button", "View analysis");
    view.type = "button";
    view.append(icon("arrow"));
    view.addEventListener("click", () => openAnalysis(analysis.id));
    const download = element("a", "secondary-button", "Report");
    download.prepend(icon("download"));
    download.href = `/api/analyses/${encodeURIComponent(analysis.id)}/report`;
    download.download = `truthlens-${analysis.id}.json`;
    download.setAttribute("aria-label", `Download JSON report for ${analysis.filename}`);
    actions.append(view, download);
    row.append(fileIcon(analysis.media_type), info, status, actions);
    full.append(row);
  });
  if (!filtered.length) full.append(emptyHistory(search ? "No displayed analyses match this filename." : "Your saved inspections will appear here. Start by uploading an image or video."));
}

async function loadHistory() {
  try {
    const data = await fetchJson("/api/analyses?limit=100");
    state.history = data.analyses || [];
    state.total = Number.isInteger(data.total) ? data.total : state.history.length;
    renderHistory();
  } catch (error) {
    byId("history-summary").textContent = "History unavailable";
    byId("full-history").replaceChildren(emptyHistory(error.message));
    if (!state.history.length) byId("recent-history").replaceChildren(emptyHistory("Saved analyses will be available when the workspace is connected."));
  }
}

async function openAnalysis(id) {
  if (state.busy) { toast("Wait for your current inspection to finish before opening another analysis."); return; }
  const operation = ++state.operation;
  try {
    const result = await fetchJson(`/api/analyses/${encodeURIComponent(id)}`);
    if (operation !== state.operation || state.busy) return;
    releasePreview();
    state.file = null;
    byId("file-input").value = "";
    byId("media-preview").replaceChildren();
    if (result.frames?.[0]?.thumbnail_url) {
      const image = element("img");
      image.src = result.frames[0].thumbnail_url;
      image.alt = `Saved ${result.media_type === "video" ? "video frame" : "image"} preview of ${result.filename}`;
      byId("media-preview").append(image);
    }
    byId("selected-filename").textContent = result.filename;
    byId("selected-details").textContent = `${formatBytes(result.media?.size_bytes)} · Saved analysis${result.media_type === "video" ? " · Sampled video frame" : ""}`;
    byId("selected-media").hidden = false;
    byId("drop-zone").hidden = true;
    byId("analyze-button").disabled = true;
    setError("");
    renderResult(result);
    byId("analyze-button-label").textContent = "Saved inspection";
    showView("analyze");
  } catch (error) { if (operation === state.operation && !state.busy) toast(error.message); }
}

document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
byId("drop-zone").addEventListener("click", () => byId("file-input").click());
byId("file-input").addEventListener("change", (event) => selectFile(event.target.files[0]));
byId("clear-file").addEventListener("click", clearFile);
byId("analyze-button").addEventListener("click", analyze);
byId("heatmap-toggle").addEventListener("change", updateEvidenceImage);
byId("history-search").addEventListener("input", renderHistory);
byId("refresh-history").addEventListener("click", async () => {
  byId("refresh-history").disabled = true;
  await loadHistory();
  byId("refresh-history").disabled = false;
});

const uploadPanel = document.querySelector(".upload-panel");
let dragDepth = 0;
uploadPanel.addEventListener("dragenter", (event) => {
  event.preventDefault();
  dragDepth += 1;
  if (!state.busy) byId("drop-zone").classList.add("dragging");
});
uploadPanel.addEventListener("dragover", (event) => {
  event.preventDefault();
  event.dataTransfer.dropEffect = state.busy ? "none" : "copy";
});
uploadPanel.addEventListener("dragleave", (event) => {
  event.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) byId("drop-zone").classList.remove("dragging");
});
uploadPanel.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  byId("drop-zone").classList.remove("dragging");
  if (state.busy) return;
  if (event.dataTransfer.files.length > 1) toast("One file at a time. The first file has been selected.");
  selectFile(event.dataTransfer.files[0]);
});
document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("drop", (event) => event.preventDefault());
window.addEventListener("beforeunload", releasePreview);

Promise.allSettled([checkStatus(), loadHistory()]);

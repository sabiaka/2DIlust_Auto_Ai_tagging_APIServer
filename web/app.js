const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const preview = document.querySelector("#preview");
const dropText = document.querySelector("#dropText");
const imageUrlInput = document.querySelector("#imageUrlInput");
const loadImageUrlButton = document.querySelector("#loadImageUrlButton");
const analyzeOverlay = document.querySelector("#analyzeOverlay");
const analyzeButton = document.querySelector("#analyzeButton");
const clearButton = document.querySelector("#clearButton");
const copyButton = document.querySelector("#copyButton");
const copyDescriptionButton = document.querySelector("#copyDescriptionButton");
const descriptionOutput = document.querySelector("#descriptionOutput");
const expressionOutput = ensureTextarea(document.querySelector("#expressionOutput"));
const situationOutput = ensureTextarea(document.querySelector("#situationOutput"));
const copyExpressionButton = addDetailCopyButton(expressionOutput, "Copy");
const copySituationButton = addDetailCopyButton(situationOutput, "Copy");
const promptOutput = document.querySelector("#promptOutput");
const tagList = document.querySelector("#tagList");
const tagCount = document.querySelector("#tagCount");
const statusText = document.querySelector("#status");
const backendBadge = document.querySelector("#backendBadge");
const tabs = document.querySelectorAll(".tab");
const views = document.querySelectorAll(".view");
const translationFile = document.querySelector("#translationFile");
const translationSearch = document.querySelector("#translationSearch");
const translationRows = document.querySelector("#translationRows");
const translationSummary = document.querySelector("#translationSummary");
const translationSaveStatus = document.querySelector("#translationSaveStatus");
const reloadTranslationsButton = document.querySelector("#reloadTranslationsButton");
const historyList = document.querySelector("#historyList");
const reloadHistoryButton = document.querySelector("#reloadHistoryButton");
const imageModal = document.querySelector("#imageModal");
const closeImageModal = document.querySelector("#closeImageModal");
const imageStage = document.querySelector("#imageStage");
const modalImage = document.querySelector("#modalImage");
const zoomOutButton = document.querySelector("#zoomOutButton");
const zoomInButton = document.querySelector("#zoomInButton");
const resetZoomButton = document.querySelector("#resetZoomButton");
const zoomSlider = document.querySelector("#zoomSlider");

let selectedFile = null;
let previewUrl = null;
let currentTags = [];
let translationLoadTimer = null;
let activeDetailHost = null;
let imageViewer = {
  scale: 1,
  x: 0,
  y: 0,
  dragging: false,
  startX: 0,
  startY: 0,
  originX: 0,
  originY: 0,
  moved: false,
};

setResultLabels();

function ensureTextarea(output) {
  if (!output || output.tagName === "TEXTAREA") {
    return output;
  }
  const textarea = document.createElement("textarea");
  textarea.id = output.id;
  textarea.className = output.className;
  textarea.spellcheck = false;
  textarea.placeholder = output.id === "expressionOutput"
    ? "Expression details appear here"
    : "Situation details appear here";
  textarea.value = "";
  output.replaceWith(textarea);
  return textarea;
}

function setResultLabels() {
  copyButton.textContent = "Copy";
  copyDescriptionButton.textContent = "Copy";

  const descriptionHead = descriptionOutput.closest(".result-panel")?.querySelector(".panel-head h2");
  if (descriptionHead) {
    descriptionHead.textContent = "Natural description";
  }

  const expressionTitle = expressionOutput.closest(".detail-card")?.querySelector("h3");
  if (expressionTitle) {
    expressionTitle.textContent = "Expression";
  }

  const situationTitle = situationOutput.closest(".detail-card")?.querySelector("h3");
  if (situationTitle) {
    situationTitle.textContent = "Situation";
  }

  const promptHead = promptOutput.closest(".result-panel")?.querySelectorAll(".panel-head h2")[1];
  if (promptHead) {
    promptHead.textContent = "Tag prompt";
  }
}

function addDetailCopyButton(output, label) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ghost mini";
  button.textContent = label;
  button.disabled = true;

  const host = output?.closest(".detail-card");
  const title = host?.querySelector("h3");
  if (host && title) {
    const head = document.createElement("div");
    head.className = "detail-card-head";
    title.replaceWith(head);
    head.append(title, button);
  }
  return button;
}

function outputText(output) {
  return "value" in output ? output.value : output.textContent;
}

function setOutputText(output, value) {
  if ("value" in output) {
    output.value = value || "";
  } else {
    output.textContent = value || "";
  }
}

function parseJsonishText(value) {
  const text = String(value || "").trim();
  if (!text) {
    return null;
  }
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const source = fenced ? fenced[1].trim() : text;
  const start = source.indexOf("{");
  const end = source.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    return null;
  }
  try {
    return JSON.parse(source.slice(start, end + 1));
  } catch {
    return null;
  }
}

function normalizeDescriptionResult(result) {
  const embedded = parseJsonishText(result.description);
  const source = embedded && typeof embedded === "object" ? { ...result, ...embedded } : result;
  return {
    description: source.description || "",
    expression: source.expression || "",
    situation: source.situation || "",
  };
}

function setNaturalOutputs(values) {
  setOutputText(descriptionOutput, values.description);
  setOutputText(expressionOutput, values.expression);
  setOutputText(situationOutput, values.situation);
  copyDescriptionButton.disabled = outputText(descriptionOutput).trim().length === 0;
  copyExpressionButton.disabled = outputText(expressionOutput).trim().length === 0;
  copySituationButton.disabled = outputText(situationOutput).trim().length === 0;
}

async function loadBackendInfo() {
  try {
    const response = await fetch("/tagger-info");
    const info = await response.json();
    backendBadge.textContent = info.backend === "pixai" ? `PixAI ${info.pixai_model_name}` : "WD ONNX";
    backendBadge.dataset.state = "online";
  } catch {
    backendBadge.textContent = "オフライン";
    backendBadge.dataset.state = "offline";
  }
}

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

function setAnalyzing(isAnalyzing) {
  analyzeOverlay.hidden = !isAnalyzing;
  dropZone.classList.toggle("is-analyzing", isAnalyzing);
  analyzeButton.disabled = isAnalyzing || !selectedFile;
}

function getPromptItems() {
  return promptOutput.value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function setPromptItems(items) {
  promptOutput.value = items.join(", ");
  copyButton.disabled = promptOutput.value.trim().length === 0;
  syncTagStates();
}

function syncTagStates() {
  const activeTags = new Set(getPromptItems());
  for (const chip of tagList.querySelectorAll(".tag")) {
    const isActive = activeTags.has(chip.dataset.promptTag);
    chip.classList.toggle("is-active", isActive);
    chip.classList.toggle("is-inactive", !isActive);
    const state = chip.querySelector(".tag-state");
    state.textContent = isActive ? "有効" : "無効";
  }
}

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("画像ファイルを選択してください", true);
    return;
  }

  selectedFile = file;
  analyzeButton.disabled = false;
  clearButton.disabled = false;
  setNaturalOutputs({ description: "", expression: "", situation: "" });
  promptOutput.value = "";
  currentTags = [];
  tagList.replaceChildren();
  tagCount.textContent = "0件";
  copyButton.disabled = true;
  activeDetailHost = null;

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(file);
  preview.onload = () => updatePreviewAspect(preview.naturalWidth, preview.naturalHeight);
  preview.src = previewUrl;
  preview.hidden = false;
  dropText.hidden = true;
  setStatus(file.name);
}

async function loadImageFromUrl(url = imageUrlInput.value) {
  const imageUrl = url.trim();
  if (!imageUrl) {
    imageUrlInput.focus();
    return;
  }

  try {
    setStatus("画像リンクを読み込み中...");
    const response = await fetch(`/image-from-url?url=${encodeURIComponent(imageUrl)}`);
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new Error(detail?.detail || "画像リンクを読み込めませんでした");
    }
    const blob = await response.blob();
    if (!blob.type.startsWith("image/")) {
      throw new Error("画像として読み込めないリンクです");
    }
    const pathname = new URL(imageUrl).pathname;
    const filename = decodeURIComponent(pathname.split("/").filter(Boolean).pop() || "linked-image");
    const file = new File([blob], filename, { type: blob.type });
    setFile(file);
    imageUrlInput.value = imageUrl;
    setStatus("画像リンクを読み込みました");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function clearAll() {
  selectedFile = null;
  fileInput.value = "";
  analyzeButton.disabled = true;
  clearButton.disabled = true;
  copyButton.disabled = true;
  setNaturalOutputs({ description: "", expression: "", situation: "" });
  promptOutput.value = "";
  currentTags = [];
  activeDetailHost = null;
  tagList.replaceChildren();
  tagCount.textContent = "0件";
  preview.hidden = true;
  preview.removeAttribute("src");
  dropZone.style.removeProperty("--preview-aspect");
  dropText.hidden = false;
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  setStatus("待機中");
}

function updatePreviewAspect(width, height) {
  if (!width || !height) {
    dropZone.style.removeProperty("--preview-aspect");
    return;
  }
  const aspect = Math.max(0.35, Math.min(2.4, width / height));
  dropZone.style.setProperty("--preview-aspect", String(aspect));
}

function renderTags(tags) {
  currentTags = tags;
  tagList.replaceChildren();
  activeDetailHost = null;
  tagCount.textContent = `${tags.length}件`;

  if (!tags.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "タグはまだありません。画像を解析するとここに表示されます。";
    tagList.append(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const tag of tags) {
    const card = document.createElement("article");
    card.className = "tag-card";

    const chip = document.createElement("div");
    chip.className = "tag";
    chip.role = "button";
    chip.tabIndex = 0;
    chip.dataset.category = tag.category;
    chip.dataset.promptTag = tag.prompt_tag;

    const body = document.createElement("span");
    body.className = "tag-body";

    const promptTag = document.createElement("span");
    promptTag.className = "tag-prompt";
    promptTag.textContent = tag.prompt_tag;
    body.append(promptTag);

    if (tag.translated && tag.translated !== tag.tag) {
      const translated = document.createElement("span");
      translated.className = "tag-translated";
      translated.textContent = tag.translated;
      body.append(translated);
    }

    const meta = document.createElement("span");
    meta.className = "tag-meta";

    const state = document.createElement("span");
    state.className = "tag-state";
    meta.append(state);

    if (typeof tag.score === "number") {
      const score = document.createElement("small");
      score.textContent = tag.score.toFixed(2);
      meta.append(score);
    }

    chip.append(body, meta);
    const toggleTag = () => {
      const current = getPromptItems();
      const next = current.includes(tag.prompt_tag)
        ? current.filter((item) => item !== tag.prompt_tag)
        : [...current, tag.prompt_tag];
      setPromptItems(next);
    };
    chip.addEventListener("click", toggleTag);
    chip.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggleTag();
      }
    });

    const detailButton = document.createElement("button");
    detailButton.type = "button";
    detailButton.className = "tag-detail-button";
    detailButton.textContent = "詳細";
    detailButton.title = "Danbooru候補を表示";
    detailButton.addEventListener("click", (event) => {
      event.stopPropagation();
      showInlineDanbooru(card, tag.prompt_tag, tag.translated || tag.tag);
    });

    meta.append(detailButton);
    card.append(chip);
    fragment.append(card);
  }
  tagList.append(fragment);
  syncTagStates();
}

async function analyze() {
  if (!selectedFile) {
    return;
  }

  copyButton.disabled = true;
  setNaturalOutputs({ description: "", expression: "", situation: "" });
  setAnalyzing(true);
  setStatus("解析中...");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/describe", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new Error(detail?.detail || `HTTP ${response.status}`);
    }

    const result = await response.json();
    setNaturalOutputs(normalizeDescriptionResult(result));
    promptOutput.value = result.prompt;
    renderTags(result.tags);
    copyButton.disabled = result.prompt.length === 0;
    setStatus(`${result.tags.length}件のタグを検出`);
    loadHistory();
  } catch (error) {
    setStatus(`解析に失敗しました: ${error.message}`, true);
  } finally {
    setAnalyzing(false);
  }
}

function activateView(viewId) {
  for (const tab of tabs) {
    tab.classList.toggle("is-active", tab.dataset.view === viewId);
  }
  for (const view of views) {
    const isActive = view.id === viewId;
    view.classList.toggle("is-active", isActive);
    view.hidden = !isActive;
  }
  activeDetailHost = null;
  for (const detail of document.querySelectorAll(".inline-detail")) {
    detail.remove();
  }
  if (viewId === "translationView" && !translationRows.children.length) {
    loadTranslations();
  }
}

function scheduleTranslationLoad() {
  clearTimeout(translationLoadTimer);
  translationLoadTimer = setTimeout(loadTranslations, 220);
}

async function loadTranslations() {
  const fileKey = translationFile.value;
  const query = translationSearch.value.trim();
  activeDetailHost = null;
  translationRows.innerHTML = '<p class="empty-state">読み込み中...</p>';
  translationSummary.textContent = "読み込み中";

  try {
    const params = new URLSearchParams({ limit: "300" });
    if (query) {
      params.set("q", query);
    }
    const response = await fetch(`/translations/${fileKey}?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    renderTranslationRows(data);
  } catch (error) {
    translationRows.innerHTML = "";
    const message = document.createElement("p");
    message.className = "empty-state error";
    message.textContent = `CSVを読み込めませんでした: ${error.message}`;
    translationRows.append(message);
    translationSummary.textContent = "読み込み失敗";
  }
}

function renderTranslationRows(data) {
  translationRows.replaceChildren();
  activeDetailHost = null;
  translationSummary.textContent = `${data.label}: ${data.rows.length} / ${data.total}件を表示`;

  if (!data.rows.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "一致する行がありません。検索語を変えてください。";
    translationRows.append(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const row of data.rows) {
    const item = document.createElement("article");
    item.className = "translation-row";
    item.tabIndex = 0;
    item.dataset.tag = row.name;

    const name = document.createElement("div");
    name.className = "translation-name";
    name.innerHTML = `<strong>${escapeHtml(row.name)}</strong><span>${formatTranslationMeta(row)}</span>`;

    const input = document.createElement("input");
    input.type = "text";
    input.value = row.japanese_name || "";
    input.placeholder = "日本語訳";
    input.autocomplete = "off";
    input.addEventListener("click", (event) => event.stopPropagation());

    const save = document.createElement("button");
    save.type = "button";
    save.className = "save-row";
    save.textContent = "保存";
    save.addEventListener("click", (event) => {
      event.stopPropagation();
      saveTranslation(data.key, row.index, input.value, save);
    });

    const open = document.createElement("button");
    open.type = "button";
    open.className = "open-row ghost";
    open.textContent = "詳細";
    open.addEventListener("click", (event) => {
      event.stopPropagation();
      showInlineDanbooru(item, row.name, input.value || row.japanese_name || row.name);
    });

    item.addEventListener("click", () => showInlineDanbooru(item, row.name, input.value || row.japanese_name || row.name));
    item.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        showInlineDanbooru(item, row.name, input.value || row.japanese_name || row.name);
      }
    });

    item.append(name, input, save, open);
    fragment.append(item);
  }
  translationRows.append(fragment);
}

function formatTranslationMeta(row) {
  const parts = [];
  if (row.category !== null && row.category !== undefined && row.category !== "") {
    parts.push(`category ${row.category}`);
  }
  if (row.count !== null && row.count !== undefined && row.count !== "") {
    parts.push(`count ${row.count}`);
  }
  return escapeHtml(parts.join(" / "));
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

async function saveTranslation(fileKey, index, japaneseName, button) {
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = "保存中";
  translationSaveStatus.textContent = "保存中...";

  try {
    const response = await fetch("/translations/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_key: fileKey, index, japanese_name: japaneseName }),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const result = await response.json();
    translationSaveStatus.textContent = `${result.row.name} を保存しました`;
    button.textContent = "保存済み";
    setTimeout(() => {
      button.textContent = originalText;
      button.disabled = false;
    }, 800);
  } catch (error) {
    translationSaveStatus.textContent = `保存に失敗しました: ${error.message}`;
    button.textContent = originalText;
    button.disabled = false;
  }
}

async function reloadTranslations() {
  translationSaveStatus.textContent = "翻訳マップを再読込中...";
  try {
    const response = await fetch("/translations/reload", { method: "POST" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    translationSaveStatus.textContent = "翻訳マップを再読込しました";
  } catch (error) {
    translationSaveStatus.textContent = `再読込に失敗しました: ${error.message}`;
  }
}

async function showInlineDanbooru(host, tag, label = tag) {
  const normalized = tag.trim().replaceAll(" ", "_");
  if (!normalized) {
    return;
  }

  if (activeDetailHost && activeDetailHost !== host) {
    activeDetailHost.classList.remove("is-expanded");
    activeDetailHost.querySelector(".inline-detail")?.remove();
  }

  const existing = host.querySelector(".inline-detail");
  if (activeDetailHost === host && existing?.dataset.tag === normalized) {
    host.classList.remove("is-expanded");
    existing.remove();
    activeDetailHost = null;
    return;
  }

  existing?.remove();
  activeDetailHost = host;
  host.classList.add("is-expanded");

  const detail = document.createElement("section");
  detail.className = "inline-detail";
  detail.dataset.tag = normalized;
  detail.innerHTML = `
    <div class="inline-detail-head">
      <div>
        <h3>${escapeHtml(normalized)}</h3>
        <p>${escapeHtml(label && label !== normalized ? `${label} / 候補を取得中` : "候補を取得中")}</p>
      </div>
      <a class="link-button" href="https://danbooru.donmai.us/posts?tags=${encodeURIComponent(normalized)}" target="_blank" rel="noreferrer">検索結果を開く</a>
    </div>
    <div class="danbooru-images"><p class="empty-state">Danbooruから候補画像を取得中...</p></div>
  `;
  host.append(detail);

  try {
    const response = await fetch(`/danbooru/preview/${encodeURIComponent(normalized)}?limit=6`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    detail.querySelector(".link-button").href = data.search_url;
    detail.querySelector("p").textContent = data.posts.length
      ? `${data.posts.length}件の候補を表示`
      : "候補画像が見つかりませんでした";
    renderDanbooruImages(detail.querySelector(".danbooru-images"), data.posts);
  } catch (error) {
    detail.querySelector("p").textContent = "Danbooru候補を取得できませんでした";
    detail.querySelector(".danbooru-images").innerHTML = `<p class="empty-state error">${escapeHtml(error.message)}</p>`;
  }
}

function renderDanbooruImages(container, posts) {
  container.replaceChildren();
  if (!posts.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "該当する画像がありません。検索結果ページを確認してください。";
    container.append(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const post of posts) {
    const link = document.createElement("a");
    link.className = "danbooru-thumb";
    link.href = post.post_url;
    link.target = "_blank";
    link.rel = "noreferrer";

    const img = document.createElement("img");
    img.src = post.preview_url;
    img.alt = `Danbooru post ${post.id}`;
    img.loading = "lazy";

    const meta = document.createElement("span");
    meta.textContent = `#${post.id}${post.rating ? ` / ${post.rating}` : ""}${post.score !== null && post.score !== undefined ? ` / score ${post.score}` : ""}`;

    link.append(img, meta);
    fragment.append(link);
  }
  container.append(fragment);
}

async function loadHistory() {
  historyList.innerHTML = '<p class="empty-state">読み込み中...</p>';
  try {
    const response = await fetch("/describe-history");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const items = await response.json();
    renderHistory(items);
  } catch (error) {
    historyList.innerHTML = `<p class="empty-state error">履歴を読み込めませんでした: ${escapeHtml(error.message)}</p>`;
  }
}

function renderHistory(items) {
  historyList.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "まだ判定履歴がありません。";
    historyList.append(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const item of items.slice(0, 12)) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "history-card";

    const img = document.createElement("img");
    img.src = item.image_url;
    img.alt = item.filename;
    img.loading = "lazy";

    const body = document.createElement("span");
    body.className = "history-body";

    const name = document.createElement("strong");
    name.textContent = item.filename || "判定画像";

    const meta = document.createElement("small");
    const created = item.created_at ? new Date(item.created_at).toLocaleString() : "";
    meta.textContent = `${item.tags.length}件 / ${created}`;

    body.append(name, meta);
    card.append(img, body);
    card.addEventListener("click", () => restoreHistoryItem(item));
    fragment.append(card);
  }
  historyList.append(fragment);
}

function restoreHistoryItem(item) {
  selectedFile = null;
  fileInput.value = "";
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  preview.onload = () => updatePreviewAspect(preview.naturalWidth, preview.naturalHeight);
  preview.src = item.image_url;
  preview.hidden = false;
  dropText.hidden = true;
  analyzeButton.disabled = true;
  clearButton.disabled = false;
  setNaturalOutputs(normalizeDescriptionResult(item));
  promptOutput.value = item.prompt || "";
  renderTags(item.tags || []);
  copyButton.disabled = promptOutput.value.trim().length === 0;
  setStatus(`${item.filename} を復元`);
}

function hasPreviewImage() {
  return !preview.hidden && Boolean(preview.getAttribute("src"));
}

function openFilePicker() {
  fileInput.click();
}

function openImageModal() {
  if (!hasPreviewImage()) {
    openFilePicker();
    return;
  }
  modalImage.src = preview.src;
  imageModal.hidden = false;
  document.body.classList.add("modal-open");
  resetImageViewer();
}

function closeViewer() {
  imageModal.hidden = true;
  document.body.classList.remove("modal-open");
  imageViewer.dragging = false;
  imageStage.classList.remove("is-dragging");
}

function resetImageViewer() {
  imageViewer.scale = 1;
  imageViewer.x = 0;
  imageViewer.y = 0;
  zoomSlider.value = "1";
  applyImageTransform();
}

function setZoom(nextScale, anchorX = 0, anchorY = 0) {
  const previous = imageViewer.scale;
  const scale = Math.max(1, Math.min(5, nextScale));
  if (scale === previous) {
    return;
  }
  const ratio = scale / previous;
  imageViewer.x = anchorX - (anchorX - imageViewer.x) * ratio;
  imageViewer.y = anchorY - (anchorY - imageViewer.y) * ratio;
  if (scale === 1) {
    imageViewer.x = 0;
    imageViewer.y = 0;
  }
  imageViewer.scale = scale;
  zoomSlider.value = String(scale);
  applyImageTransform();
}

function applyImageTransform() {
  modalImage.style.transform = `translate3d(${imageViewer.x}px, ${imageViewer.y}px, 0) scale(${imageViewer.scale})`;
  imageStage.classList.toggle("is-zoomed", imageViewer.scale > 1);
}

function stageAnchorFromEvent(event) {
  const rect = imageStage.getBoundingClientRect();
  return {
    x: event.clientX - rect.left - rect.width / 2,
    y: event.clientY - rect.top - rect.height / 2,
  };
}

function shouldIgnoreDropZoneClick(event) {
  return event.target === fileInput || Boolean(event.target.closest("[data-drop-control]"));
}

function droppedText(event) {
  return event.dataTransfer.getData("text/uri-list") || event.dataTransfer.getData("text/plain");
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
loadImageUrlButton.addEventListener("click", () => loadImageFromUrl());
imageUrlInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    loadImageFromUrl();
  }
});
analyzeButton.addEventListener("click", analyze);
clearButton.addEventListener("click", clearAll);
dropZone.addEventListener("click", (event) => {
  if (shouldIgnoreDropZoneClick(event)) {
    return;
  }
  if (hasPreviewImage()) {
    openImageModal();
  } else {
    openFilePicker();
  }
});
dropZone.addEventListener("keydown", (event) => {
  if (shouldIgnoreDropZoneClick(event)) {
    return;
  }
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    if (hasPreviewImage()) {
      openImageModal();
    } else {
      openFilePicker();
    }
  }
});
promptOutput.addEventListener("input", () => {
  copyButton.disabled = promptOutput.value.trim().length === 0;
  if (currentTags.length) {
    syncTagStates();
  }
});

copyButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(promptOutput.value);
  setStatus("コピーしました");
});

copyDescriptionButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(outputText(descriptionOutput));
  setStatus("自然文をコピーしました");
});

copyExpressionButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(outputText(expressionOutput));
  setStatus("Expression copied");
});

copySituationButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(outputText(situationOutput));
  setStatus("Situation copied");
});

descriptionOutput.addEventListener("input", () => {
  copyDescriptionButton.disabled = outputText(descriptionOutput).trim().length === 0;
});

expressionOutput.addEventListener("input", () => {
  copyExpressionButton.disabled = outputText(expressionOutput).trim().length === 0;
});

situationOutput.addEventListener("input", () => {
  copySituationButton.disabled = outputText(situationOutput).trim().length === 0;
});

for (const tab of tabs) {
  tab.addEventListener("click", () => activateView(tab.dataset.view));
}

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
}

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) {
    setFile(file);
    return;
  }
  const text = droppedText(event).trim();
  if (text) {
    imageUrlInput.value = text;
    loadImageFromUrl(text);
  }
});

document.addEventListener("paste", (event) => {
  const items = Array.from(event.clipboardData?.items || []);
  const imageItem = items.find((item) => item.type.startsWith("image/"));
  if (imageItem) {
    const file = imageItem.getAsFile();
    if (file) {
      event.preventDefault();
      setFile(file);
      setStatus("クリップボード画像を読み込みました");
    }
    return;
  }

  const text = event.clipboardData?.getData("text/plain")?.trim();
  if (text && /^https?:\/\//i.test(text)) {
    event.preventDefault();
    imageUrlInput.value = text;
    loadImageFromUrl(text);
  }
});

closeImageModal.addEventListener("click", closeViewer);
imageModal.addEventListener("click", (event) => {
  if (event.target.classList.contains("image-modal-backdrop")) {
    closeViewer();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !imageModal.hidden) {
    closeViewer();
  }
});
imageStage.addEventListener("wheel", (event) => {
  event.preventDefault();
  const anchor = stageAnchorFromEvent(event);
  const delta = event.deltaY < 0 ? 0.18 : -0.18;
  setZoom(imageViewer.scale + delta, anchor.x, anchor.y);
}, { passive: false });
imageStage.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) {
    return;
  }
  imageViewer.dragging = true;
  imageViewer.startX = event.clientX;
  imageViewer.startY = event.clientY;
  imageViewer.originX = imageViewer.x;
  imageViewer.originY = imageViewer.y;
  imageViewer.moved = false;
  imageStage.setPointerCapture(event.pointerId);
  imageStage.classList.toggle("is-dragging", imageViewer.scale > 1);
});
imageStage.addEventListener("pointermove", (event) => {
  if (!imageViewer.dragging) {
    return;
  }
  const deltaX = event.clientX - imageViewer.startX;
  const deltaY = event.clientY - imageViewer.startY;
  if (Math.hypot(deltaX, deltaY) > 4) {
    imageViewer.moved = true;
  }
  if (imageViewer.scale > 1) {
    imageViewer.x = imageViewer.originX + deltaX;
    imageViewer.y = imageViewer.originY + deltaY;
    applyImageTransform();
  }
});
imageStage.addEventListener("pointerup", (event) => {
  const wasClick = imageViewer.dragging && !imageViewer.moved;
  imageViewer.dragging = false;
  imageStage.classList.remove("is-dragging");
  if (wasClick) {
    const anchor = stageAnchorFromEvent(event);
    setZoom(imageViewer.scale >= 4.95 ? 1 : imageViewer.scale + 0.55, anchor.x, anchor.y);
  }
});
imageStage.addEventListener("pointercancel", () => {
  imageViewer.dragging = false;
  imageStage.classList.remove("is-dragging");
});
imageStage.addEventListener("dblclick", (event) => {
  event.preventDefault();
  const anchor = stageAnchorFromEvent(event);
  setZoom(imageViewer.scale + 0.8, anchor.x, anchor.y);
});
zoomOutButton.addEventListener("click", () => setZoom(imageViewer.scale - 0.3));
zoomInButton.addEventListener("click", () => setZoom(imageViewer.scale + 0.3));
resetZoomButton.addEventListener("click", resetImageViewer);
zoomSlider.addEventListener("input", () => setZoom(Number(zoomSlider.value)));

translationFile.addEventListener("change", loadTranslations);
translationSearch.addEventListener("input", scheduleTranslationLoad);
reloadTranslationsButton.addEventListener("click", reloadTranslations);
reloadHistoryButton.addEventListener("click", loadHistory);

loadBackendInfo();
loadHistory();

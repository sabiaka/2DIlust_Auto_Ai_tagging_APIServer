const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const preview = document.querySelector("#preview");
const dropText = document.querySelector("#dropText");
const analyzeButton = document.querySelector("#analyzeButton");
const clearButton = document.querySelector("#clearButton");
const copyButton = document.querySelector("#copyButton");
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

let selectedFile = null;
let previewUrl = null;
let currentTags = [];
let translationLoadTimer = null;

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
  promptOutput.value = "";
  currentTags = [];
  tagList.replaceChildren();
  tagCount.textContent = "0件";
  copyButton.disabled = true;

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  dropText.hidden = true;
  setStatus(file.name);
}

function clearAll() {
  selectedFile = null;
  fileInput.value = "";
  analyzeButton.disabled = true;
  clearButton.disabled = true;
  copyButton.disabled = true;
  promptOutput.value = "";
  currentTags = [];
  tagList.replaceChildren();
  tagCount.textContent = "0件";
  preview.hidden = true;
  preview.removeAttribute("src");
  dropText.hidden = false;
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  setStatus("待機中");
}

function renderTags(tags) {
  currentTags = tags;
  tagList.replaceChildren();
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
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "tag";
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
    chip.addEventListener("click", () => {
      const current = getPromptItems();
      const next = current.includes(tag.prompt_tag)
        ? current.filter((item) => item !== tag.prompt_tag)
        : [...current, tag.prompt_tag];
      setPromptItems(next);
    });

    fragment.append(chip);
  }
  tagList.append(fragment);
  syncTagStates();
}

async function analyze() {
  if (!selectedFile) {
    return;
  }

  analyzeButton.disabled = true;
  copyButton.disabled = true;
  setStatus("解析中...");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();
    promptOutput.value = result.prompt;
    renderTags(result.tags);
    copyButton.disabled = result.prompt.length === 0;
    setStatus(`${result.tags.length}件のタグを検出`);
  } catch (error) {
    setStatus(`解析に失敗しました: ${error.message}`, true);
  } finally {
    analyzeButton.disabled = !selectedFile;
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

    const name = document.createElement("div");
    name.className = "translation-name";
    name.innerHTML = `<strong>${escapeHtml(row.name)}</strong><span>${formatTranslationMeta(row)}</span>`;

    const input = document.createElement("input");
    input.type = "text";
    input.value = row.japanese_name || "";
    input.placeholder = "日本語訳";
    input.autocomplete = "off";

    const save = document.createElement("button");
    save.type = "button";
    save.className = "save-row";
    save.textContent = "保存";
    save.addEventListener("click", () => saveTranslation(data.key, row.index, input.value, save));

    item.append(name, input, save);
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

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
analyzeButton.addEventListener("click", analyze);
clearButton.addEventListener("click", clearAll);
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
  setFile(event.dataTransfer.files[0]);
});

translationFile.addEventListener("change", loadTranslations);
translationSearch.addEventListener("input", scheduleTranslationLoad);
reloadTranslationsButton.addEventListener("click", reloadTranslations);

loadBackendInfo();

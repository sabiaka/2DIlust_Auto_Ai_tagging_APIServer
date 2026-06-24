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

let selectedFile = null;
let previewUrl = null;

async function loadBackendInfo() {
  try {
    const response = await fetch("/tagger-info");
    const info = await response.json();
    backendBadge.textContent = info.backend === "pixai" ? `PixAI ${info.pixai_model_name}` : "WD ONNX";
  } catch {
    backendBadge.textContent = "offline";
  }
}

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("画像ファイルを選んでください", true);
    return;
  }

  selectedFile = file;
  analyzeButton.disabled = false;
  clearButton.disabled = false;
  promptOutput.value = "";
  tagList.replaceChildren();
  tagCount.textContent = "0";
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
  tagList.replaceChildren();
  tagCount.textContent = "0";
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
  tagList.replaceChildren();
  tagCount.textContent = String(tags.length);

  const fragment = document.createDocumentFragment();
  for (const tag of tags) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "tag";
    chip.dataset.category = tag.category;

    const promptTag = document.createElement("span");
    promptTag.className = "tag-prompt";
    promptTag.textContent = tag.prompt_tag;
    chip.append(promptTag);

    if (tag.translated && tag.translated !== tag.tag) {
      const translated = document.createElement("span");
      translated.className = "tag-translated";
      translated.textContent = tag.translated;
      chip.append(translated);
    }

    if (typeof tag.score === "number") {
      const score = document.createElement("small");
      score.textContent = tag.score.toFixed(2);
      chip.append(score);
    }

    chip.addEventListener("click", () => {
      const current = promptOutput.value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
      promptOutput.value = current.includes(tag.prompt_tag)
        ? current.filter((item) => item !== tag.prompt_tag).join(", ")
        : [...current, tag.prompt_tag].join(", ");
      copyButton.disabled = promptOutput.value.trim().length === 0;
    });

    fragment.append(chip);
  }
  tagList.append(fragment);
}

async function analyze() {
  if (!selectedFile) {
    return;
  }

  analyzeButton.disabled = true;
  copyButton.disabled = true;
  setStatus("判定中...");

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
    setStatus(`${result.tags.length} tags`);
  } catch (error) {
    setStatus(`判定に失敗しました: ${error.message}`, true);
  } finally {
    analyzeButton.disabled = !selectedFile;
  }
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
analyzeButton.addEventListener("click", analyze);
clearButton.addEventListener("click", clearAll);

copyButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(promptOutput.value);
  setStatus("コピーしました");
});

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

loadBackendInfo();

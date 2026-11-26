"use strict";

const byId = (id) => document.getElementById(id);
const ui = Object.fromEntries([
  "check-tab", "history-tab", "check-panel", "history-panel", "new-check", "history-message",
  "history-table-wrap", "history-body", "sample-button", "choose-button", "file-input",
  "photo-name", "original-button", "annotated-button", "stage", "stage-image", "preview", "overlay",
  "no-photo", "processed-link", "analysis-badge", "analysis-title", "inspection-meta", "feedback",
  "empty-result", "empty-title", "empty-text", "summary", "count-all", "count-defects", "count-review",
  "object-list", "analyze-button", "analyze-label", "analysis-note",
].map((id) => [id, byId(id)]));
const spinner = ui["analyze-button"].querySelector(".spinner");
const analysisPanel = ui["analysis-title"].closest("section");
const sampleUrl = "./samples/tksh-022.jpg";
const dateFormat = new Intl.DateTimeFormat("ru-RU", { dateStyle: "short", timeStyle: "short" });
const state = {
  file: null, objectUrl: null, result: null, selected: 0, view: "original",
  analyzing: false, sampleToken: 0, tab: "check",
};

async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    if (typeof payload.detail === "string") throw new Error(payload.detail);
    throw new Error(`Сервер вернул ошибку ${response.status}.`);
  }
  return response.json();
}

function message(text = "", error = false) {
  ui.feedback.textContent = text;
  ui.feedback.classList.toggle("error", error);
}

function badge(text, style) {
  ui["analysis-badge"].textContent = text;
  ui["analysis-badge"].className = `badge ${style}`;
}

function selectTab(name) {
  state.tab = name;
  for (const tab of ["check", "history"]) {
    ui[`${tab}-panel`].hidden = tab !== name;
    ui[`${tab}-tab`].classList.toggle("active", tab === name);
    ui[`${tab}-tab`].setAttribute("aria-pressed", String(tab === name));
  }
}

function objectKind(object) {
  if (object.needs_repair) return "defect";
  return object.requires_inspection ? "review" : "normal";
}

const statusStyles = {
  defect: { label: "Дефект", color: "#ff93ad" },
  review: { label: "На проверку", color: "#ffcf72" },
  normal: { label: "Норма", color: "#a0e8bb" },
};
const objectNumber = (index) => String(index + 1).padStart(2, "0");

function renderBoxes() {
  ui.overlay.replaceChildren();
  const { naturalWidth: width, naturalHeight: height } = ui.preview;
  if (!width || !height || !state.objectUrl || state.view !== "annotated") return;
  const detections = state.result?.detections ?? [];
  for (const [index, object] of detections.entries()) {
    const [x1, y1, x2, y2] = object.bbox;
    const box = document.createElement("div");
    box.className = `box${index === state.selected ? " selected" : ""}`;
    box.style.setProperty("--object-color", statusStyles[objectKind(object)].color);
    Object.assign(box.style, {
      left: `${100 * x1 / width}%`, top: `${100 * y1 / height}%`,
      width: `${100 * (x2 - x1) / width}%`, height: `${100 * (y2 - y1) / height}%`,
    });
    const label = document.createElement("span");
    label.textContent = `${objectNumber(index)} · ${object.object_type}`;
    box.append(label);
    ui.overlay.append(box);
  }
}

function renderImage() {
  const processed = state.result?.processed_image_url;
  const source = state.objectUrl || processed;
  if (source && ui.preview.getAttribute("src") !== source) ui.preview.src = source;
  if (!source) ui.preview.removeAttribute("src");
  ui["stage-image"].hidden = !source;
  ui["no-photo"].hidden = !!source;
  ui["original-button"].disabled = !state.objectUrl;
  ui["annotated-button"].disabled = !processed;
  ui["original-button"].setAttribute("aria-pressed", String(state.view === "original" && !!state.objectUrl));
  ui["annotated-button"].setAttribute("aria-pressed", String(state.view === "annotated" && !!processed));
  ui["processed-link"].hidden = !processed;
  if (processed) ui["processed-link"].href = processed;
  renderBoxes();
}

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

function renderResults() {
  const objects = state.result?.detections ?? [];
  ui.summary.hidden = !state.result;
  ui["empty-result"].hidden = objects.length > 0;
  ui["empty-title"].textContent = state.result ? "Объекты не найдены" : "Готово к запуску";
  ui["empty-text"].textContent = state.result
    ? "Попробуйте более крупный и чёткий кадр оборудования."
    : "Здесь появятся найденные объекты и их состояния.";
  ui["count-all"].textContent = objects.length;
  ui["count-defects"].textContent = objects.filter((object) => objectKind(object) === "defect").length;
  ui["count-review"].textContent = objects.filter((object) => objectKind(object) === "review").length;
  ui["object-list"].replaceChildren();
  for (const [index, object] of objects.entries()) {
    const kind = objectKind(object);
    const card = element("button", `object-card${index === state.selected ? " selected" : ""}`);
    card.type = "button";
    card.setAttribute("aria-pressed", String(index === state.selected));
    const header = element("span", "object-head");
    const name = element("span", null);
    name.append(element("span", "object-id", objectNumber(index)), object.object_type);
    header.append(name, element("span", `tag ${kind}`, statusStyles[kind].label));
    card.append(header, element("span", "object-status", object.status));
    const coordinates = object.latitude != null && object.longitude != null
      ? ` · GPS ${object.latitude.toFixed(5)}, ${object.longitude.toFixed(5)}` : "";
    card.append(element("span", "object-meta",
      `Уверенность детектора ${Math.round(object.confidence * 100)}%${coordinates}`));
    if (object.description) card.append(element("span", "object-meta", object.description));
    card.addEventListener("click", () => {
      state.selected = index;
      state.view = "annotated";
      for (const [position, button] of Array.from(ui["object-list"].children).entries()) {
        button.classList.toggle("selected", position === index);
        button.setAttribute("aria-pressed", String(position === index));
      }
      renderImage();
    });
    ui["object-list"].append(card);
  }
}

function busy(value) {
  state.analyzing = value;
  analysisPanel.setAttribute("aria-busy", String(value));
  ui["analyze-button"].disabled = value || !state.file;
  ui["sample-button"].disabled = value;
  ui["choose-button"].disabled = value;
  ui["history-tab"].disabled = value;
  spinner.hidden = !value;
  ui["analyze-label"].textContent = value ? "Анализируем кадр…" : "Проанализировать кадр";
}

function setPhoto(file, title = file.name) {
  if (state.analyzing) return;
  state.sampleToken++;
  if (file.size === 0 || file.size > 10 * 1024 * 1024 || (file.type && !file.type.startsWith("image/"))) {
    message("Выберите непустой файл изображения размером до 10 МБ.", true);
    return;
  }
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = file;
  state.objectUrl = URL.createObjectURL(file);
  state.result = null;
  state.selected = 0;
  state.view = "original";
  ui["photo-name"].textContent = title;
  ui["no-photo"].textContent = "Выберите фотографию";
  ui["inspection-meta"].textContent = "Фото готово к анализу";
  ui["analysis-note"].textContent = "Тип оборудования и состояние определяются по фото.";
  message();
  badge("Ожидание", "neutral");
  busy(false);
  renderResults();
  renderImage();
}

async function openSample() {
  const token = ++state.sampleToken;
  try {
    const response = await fetch(sampleUrl);
    if (!response.ok) throw new Error();
    const blob = await response.blob();
    if (token === state.sampleToken) setPhoto(new File([blob], "tksh-022.jpg", { type: "image/jpeg" }), "Демо-кадр · ТКШ");
  } catch {
    if (token !== state.sampleToken) return;
    if (!state.file) ui["photo-name"].textContent = "Фото не выбрано";
    message("Не удалось открыть демо-кадр. Можно выбрать своё фото.", true);
  }
}

function finish(result, fromHistory = false) {
  state.result = result;
  state.selected = 0;
  state.view = result.processed_image_url ? "annotated" : "original";
  ui["inspection-meta"].textContent = `Проверка №${result.id}`;
  ui["analysis-note"].textContent = fromHistory ? "Результат загружен из журнала." : "Проверка сохранена в журнале.";
  badge("Готово", "complete");
  message();
  renderResults();
  renderImage();
}

async function analyze() {
  if (!state.file || state.analyzing) return;
  state.sampleToken++;
  busy(true);
  message("Ищем оборудование и оцениваем его состояние.");
  badge("Анализ", "running");
  try {
    const body = new FormData();
    body.append("file", state.file);
    finish(await request("/api/v1/analyze", { method: "POST", body }));
  } catch (error) {
    badge("Ошибка", "error");
    message(error.message || "Не удалось выполнить проверку.", true);
  } finally {
    busy(false);
  }
}

function inspectionDate(value) {
  const normalized = /[zZ]|[+-]\d\d:\d\d$/.test(value) ? value : `${value}Z`;
  return dateFormat.format(new Date(normalized));
}

async function openHistory() {
  selectTab("history");
  ui["history-message"].textContent = "Загружаем журнал…";
  ui["history-table-wrap"].hidden = true;
  try {
    const inspections = await request("/api/v1/inspections");
    ui["history-body"].replaceChildren();
    for (const inspection of inspections) {
      const row = element("tr", null);
      const values = [
        `№${inspection.id} · ${inspection.image_filename}`,
        inspectionDate(inspection.created_at),
        inspection.detections.length,
        inspection.detections.filter((object) => object.requires_inspection).length,
      ];
      for (const value of values) row.append(element("td", null, value));
      const action = element("td", null);
      const open = element("button", "button button-outline", "Открыть");
      open.type = "button";
      open.addEventListener("click", () => loadInspection(inspection.id, open));
      action.append(open);
      row.append(action);
      ui["history-body"].append(row);
    }
    ui["history-table-wrap"].hidden = !inspections.length;
    ui["history-message"].textContent = inspections.length ? "" : "Пока нет сохранённых проверок.";
  } catch (error) {
    ui["history-message"].textContent = error.message;
  }
}

async function loadInspection(id, button) {
  button.disabled = true;
  try {
    const result = await request(`/api/v1/inspections/${id}`);
    if (state.tab !== "history") return;
    state.sampleToken++;
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
    state.file = null;
    ui["photo-name"].textContent = result.image_filename;
    ui["no-photo"].textContent = "Сохранённая разметка отсутствует";
    finish(result, true);
    busy(false);
    selectTab("check");
  } catch (error) {
    ui["history-message"].textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

ui.preview.addEventListener("load", renderBoxes);
ui.preview.addEventListener("error", () => {
  if (!ui.preview.getAttribute("src")) return;
  ui["stage-image"].hidden = true;
  ui["no-photo"].hidden = false;
  ui["no-photo"].textContent = "Не удалось открыть фото";
});
ui["original-button"].addEventListener("click", () => {
  state.view = "original";
  renderImage();
});
ui["annotated-button"].addEventListener("click", () => {
  state.view = "annotated";
  renderImage();
});
ui["check-tab"].addEventListener("click", () => selectTab("check"));
ui["new-check"].addEventListener("click", () => selectTab("check"));
ui["history-tab"].addEventListener("click", openHistory);
ui["sample-button"].addEventListener("click", openSample);
ui["choose-button"].addEventListener("click", () => ui["file-input"].click());
ui["file-input"].addEventListener("change", () => {
  const [file] = ui["file-input"].files;
  if (file) setPhoto(file);
  ui["file-input"].value = "";
});
ui.stage.addEventListener("dragover", (event) => {
  event.preventDefault();
  if (!state.analyzing) ui.stage.classList.add("dragging");
});
ui.stage.addEventListener("dragleave", () => ui.stage.classList.remove("dragging"));
ui.stage.addEventListener("drop", (event) => {
  event.preventDefault();
  ui.stage.classList.remove("dragging");
  const [file] = event.dataTransfer.files;
  if (file) setPhoto(file);
});
ui["analyze-button"].addEventListener("click", analyze);
openSample();

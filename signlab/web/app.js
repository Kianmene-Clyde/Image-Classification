let catalog = [];
let previewUrl;

const byId = id => document.getElementById(id);

function textNode(tag, value) {
  const element = document.createElement(tag);
  element.textContent = value;
  return element;
}

async function getJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Erreur HTTP ${response.status}`);
  return data;
}

function drawCatalog() {
  const area = byId("catalog");
  const family = byId("filter").value;
  area.replaceChildren();
  for (const model of catalog) {
    if (family && model.family !== family) continue;
    const card = document.createElement("article");
    card.className = "model";
    card.append(
      textNode("small", model.family),
      textNode("h3", model.name),
      textNode("p", model.description),
      textNode("p", `Échelle : ${model.scale}`),
      textNode("code", JSON.stringify(model.candidates)),
    );
    area.append(card);
  }
}

function updateForm() {
  const name = byId("model").value;
  byId("predict").disabled = !name || !byId("file").files.length;
  byId("modelDescription").textContent = catalog.find(model => model.name === name)?.description || "";
}

function updatePreview() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const file = byId("file").files[0];
  const preview = byId("preview");
  if (file) {
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.style.display = "block";
  } else {
    preview.removeAttribute("src");
    preview.style.display = "none";
  }
  updateForm();
}

async function predict() {
  byId("error").textContent = "";
  byId("prediction").textContent = "";
  byId("predictionNote").textContent = "";
  byId("predict").disabled = true;
  try {
    const file = byId("file").files[0];
    if (file.size > 8 * 1024 * 1024) throw new Error("Image supérieure à 8 Mo.");
    const name = encodeURIComponent(byId("model").value);
    const result = await getJson(`/api/predict?model=${name}`, {
      method: "POST",
      body: file,
      headers: { "Content-Type": file.type || "application/octet-stream" },
    });
    byId("prediction").textContent = result.class.replaceAll("_", " ");
    const label = result.data_kind === "synthetic" ? "MODÈLE SYNTHÉTIQUE · " : "";
    byId("predictionNote").textContent = label + result.score_notice;
  } catch (error) {
    byId("error").textContent = error.message;
  } finally {
    updateForm();
  }
}

function drawResults(result) {
  const area = byId("results");
  if (!result.comparison.length) {
    area.append(textNode("p", "Remplis les dossiers d’images puis lance une comparaison."));
    return;
  }
  byId("runNotice").textContent = result.selection.data_kind === "synthetic"
    ? "Données synthétiques utilisées pour vérifier le fonctionnement."
    : "La validation sert à la sélection. L’écart-type indique la variabilité entre graines.";
  const table = document.createElement("table");
  const header = document.createElement("tr");
  for (const name of ["Modèle", "Macro-F1 ± écart-type", "Temps (s)"]) {
    const cell = textNode("th", name);
    cell.scope = "col";
    header.append(cell);
  }
  table.createTHead().append(header);
  const body = table.createTBody();
  for (const resultRow of result.comparison) {
    const row = body.insertRow();
    const mean = resultRow.validation_macro_f1_mean.toFixed(3);
    const std = resultRow.validation_macro_f1_std.toFixed(3);
    for (const value of [resultRow.model, `${mean} ± ${std}`, resultRow.fit_seconds_mean.toFixed(2)]) {
      row.append(textNode("td", value));
    }
  }
  area.append(table);
}

async function initialize() {
  try {
    catalog = await getJson("/api/catalog");
    const native = catalog.filter(model => model.name.startsWith("native_")).length;
    byId("status").textContent = `${catalog.length - native} modèles Python · ${native} variantes C++`;
    for (const family of new Set(catalog.map(model => model.family))) {
      const option = textNode("option", family);
      option.value = family;
      byId("filter").append(option);
    }
    const trained = catalog.filter(model => model.trained);
    if (trained.length) {
      byId("model").replaceChildren();
      for (const model of trained) {
        const option = textNode("option", model.name);
        option.value = model.name;
        byId("model").append(option);
      }
    }
    drawCatalog();
    updateForm();
    drawResults(await getJson("/api/results"));
  } catch (error) {
    byId("error").textContent = `Chargement de SignLab impossible : ${error.message}`;
  }
}

byId("filter").addEventListener("change", drawCatalog);
byId("model").addEventListener("change", updateForm);
byId("file").addEventListener("change", updatePreview);
byId("predict").addEventListener("click", predict);
initialize();

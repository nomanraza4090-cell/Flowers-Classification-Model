const file = document.getElementById("file");
const drop = document.getElementById("drop");
const previewWrap = document.getElementById("previewWrap");
const preview = document.getElementById("preview");
const removeImg = document.getElementById("removeImg");
const predictBtn = document.getElementById("predict");
const resetBtn = document.getElementById("reset");
const resultEmpty = document.getElementById("resultEmpty");
const result = document.getElementById("result");
const label = document.getElementById("label");
const conf = document.getElementById("confidence");
const confText = document.getElementById("confidenceText");
const top3Box = document.getElementById("top3");
const error = document.getElementById("error");
const btnText = predictBtn.querySelector(".btn-text");
const btnLoader = predictBtn.querySelector(".btn-loader");

let selected = null;

drop.addEventListener("click", () => file.click());
file.addEventListener("change", () => setFile(file.files[0]));

["dragover", "dragenter"].forEach(evt =>
  drop.addEventListener(evt, e => { e.preventDefault(); drop.classList.add("dragover"); })
);
["dragleave", "drop"].forEach(evt =>
  drop.addEventListener(evt, e => { e.preventDefault(); drop.classList.remove("dragover"); })
);
drop.addEventListener("drop", e => setFile(e.dataTransfer.files[0]));

removeImg.addEventListener("click", (e) => { e.stopPropagation(); clearFile(); });

function setFile(f) {
  if (!f || !f.type.startsWith("image/")) {
    error.textContent = "Please select a valid image file.";
    return;
  }
  selected = f;
  drop.hidden = true;
  previewWrap.hidden = false;
  preview.src = URL.createObjectURL(f);
  predictBtn.disabled = false;
  result.hidden = true;
  resultEmpty.hidden = false;
  error.textContent = "";
}

function clearFile() {
  selected = null;
  file.value = "";
  drop.hidden = false;
  previewWrap.hidden = true;
  predictBtn.disabled = true;
  result.hidden = true;
  resultEmpty.hidden = false;
  error.textContent = "";
}

predictBtn.addEventListener("click", async () => {
  if (!selected) return;
  predictBtn.disabled = true;
  btnText.textContent = "Analysing...";
  btnLoader.hidden = false;
  error.textContent = "";

  const fd = new FormData();
  fd.append("image", selected);

  try {
    const r = await fetch("/api/predict", { method: "POST", body: fd });
    const d = await r.json();
    if (!d.success) throw new Error(d.error || "Prediction failed");

    label.textContent = d.prediction;
    const p = d.confidence;
    conf.style.width = p + "%";
    confText.textContent = `Confidence: ${p.toFixed(2)}%`;

    top3Box.innerHTML = "";
    if (d.top_predictions) {
      d.top_predictions.forEach(item => {
        const row = document.createElement("div");
        row.className = "top3-row";
        row.innerHTML = `
          <span class="top3-name">${item.class_name}</span>
          <span class="top3-track"><span class="top3-fill" style="width:${item.confidence_percent}%"></span></span>
          <span class="top3-pct">${item.confidence_percent.toFixed(1)}%</span>`;
        top3Box.appendChild(row);
      });
    }

    resultEmpty.hidden = true;
    result.hidden = false;
  } catch (e) {
    error.textContent = e.message;
  } finally {
    predictBtn.disabled = false;
    btnText.textContent = "🔍 Predict Flower";
    btnLoader.hidden = true;
  }
});

resetBtn.addEventListener("click", clearFile);
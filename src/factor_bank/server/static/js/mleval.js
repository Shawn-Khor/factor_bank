// ─── State ─────────────────────────────────────────────────────────────────
const ML_MAX_FACTORS = 20;
const ML_HORIZON_VALUES = [1, 5, 21, 63];

const mlState = {
  catalog: null,
  selected: new Set(),
  jobId: null,
  mdiChart: null,
};

document.getElementById("ml-from").value = yearsAgoISO(3);
document.getElementById("ml-to").value = todayISO();

// ─── Load catalog ──────────────────────────────────────────────────────────
async function loadMlCatalog() {
  if (window.fbCatalog) {
    mlState.catalog = window.fbCatalog;
  } else {
    const res = await fetch("/api/factors");
    if (!res.ok) {
      setMlStatus("Failed to load factor catalog.", true);
      return;
    }
    mlState.catalog = await res.json();
    window.fbCatalog = mlState.catalog;
  }
  if (mlState.catalog.date_floor) {
    document.getElementById("ml-from").min = mlState.catalog.date_floor;
    document.getElementById("ml-to").min = mlState.catalog.date_floor;
  }
  renderMlFactorPills();
}

function renderMlFactorPills() {
  const el = document.getElementById("ml-factor-pills");
  el.innerHTML = "";
  for (const [group, factors] of Object.entries(mlState.catalog.groups)) {
    const wrap = document.createElement("div");
    wrap.className = "factor-group";
    wrap.innerHTML = `
      <div class="factor-group-label">
        <span>${group}</span>
        <span class="divider"></span>
      </div>
      <div class="factor-pills"></div>
    `;
    const pills = wrap.querySelector(".factor-pills");
    for (const [name, desc] of Object.entries(factors)) {
      const pill = document.createElement("button");
      pill.className = "factor-pill";
      pill.textContent = name;
      pill.title = desc;
      pill.addEventListener("click", () => toggleMlFactor(name, pill));
      pills.appendChild(pill);
    }
    el.appendChild(wrap);
  }
}

function toggleMlFactor(name, pillEl) {
  if (mlState.selected.has(name)) {
    mlState.selected.delete(name);
    pillEl.classList.remove("active");
  } else {
    if (mlState.selected.size >= ML_MAX_FACTORS) {
      setMlStatus(`Max ${ML_MAX_FACTORS} factors selected — deselect one first.`, true);
      return;
    }
    mlState.selected.add(name);
    pillEl.classList.add("active");
  }
  updateMlCount();
}

function updateMlCount() {
  const n = mlState.selected.size;
  document.getElementById("ml-count").textContent = `${n} selected`;
  document.getElementById("ml-run").disabled = n < 2;
  if (n >= 2) {
    setMlStatus(`${n} factors selected. Click Run.`);
  } else {
    setMlStatus("Pick 2–20 factors and click Run.");
  }
}

function setMlStatus(msg, isError = false) {
  const el = document.getElementById("ml-status");
  el.textContent = msg;
  el.className = "status" + (isError ? " error" : "");
}

function getSelectedHorizons() {
  return ML_HORIZON_VALUES.filter(h => document.getElementById(`mlh-${h}`).checked);
}

// ─── Run ───────────────────────────────────────────────────────────────────
document.getElementById("ml-run").addEventListener("click", runMlEval);

async function runMlEval() {
  const factors = Array.from(mlState.selected);
  if (factors.length < 2) return;
  const horizons = getSelectedHorizons();
  if (horizons.length === 0) {
    setMlStatus("Select at least one horizon.", true);
    return;
  }

  const btn = document.getElementById("ml-run");
  const progressEl = document.getElementById("ml-progress");
  const progressText = document.getElementById("ml-progress-text");
  const elapsedEl = document.getElementById("ml-elapsed");

  btn.disabled = true;
  document.getElementById("ml-results").classList.add("hidden");
  setMlStatus("Submitting job…");

  let elapsedTimer = null;
  const startedAt = Date.now();

  try {
    const res = await fetch("/api/ml-eval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        factors,
        horizons,
        from_date: document.getElementById("ml-from").value,
        to_date: document.getElementById("ml-to").value,
        quantiles: parseInt(document.getElementById("ml-quantiles").value, 10),
        mode: document.getElementById("ml-mode").value,
        tier2: document.getElementById("ml-tier2").checked,
      }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      setMlStatus(data.error || `HTTP ${res.status}`, true);
      return;
    }

    mlState.jobId = data.job_id;
    progressEl.classList.remove("hidden");
    progressText.textContent = "queued";
    elapsedEl.textContent = "0s";
    elapsedTimer = setInterval(() => {
      elapsedEl.textContent = `${Math.floor((Date.now() - startedAt) / 1000)}s`;
    }, 1000);
    setMlStatus(`Running job ${data.job_id}…`);

    const result = await pollJob(data.job_id, rec => {
      progressText.textContent = rec.progress || rec.status;
    });
    renderMlResults(result);
    setMlStatus("Done.");
  } catch (err) {
    setMlStatus(err.message || String(err), true);
  } finally {
    if (elapsedTimer) clearInterval(elapsedTimer);
    progressEl.classList.add("hidden");
    btn.disabled = mlState.selected.size < 2;
  }
}

// ─── Render results ────────────────────────────────────────────────────────
function renderMlResults(result) {
  document.getElementById("ml-results").classList.remove("hidden");
  renderScreening(result.screening, result.meta);
  renderMiHeatmap(result.mutual_info);
  renderRedundancy(result.redundancy);
  renderMdi(result.mdi);
}

const SCREENING_COLS = [
  { key: "feature", label: "Feature" },
  { key: "chosen_horizon", label: "Horizon", format: v => (v == null ? "—" : `${v}D`) },
  { key: "mean_ic", label: "Mean IC", format: v => fmt(v, 4) },
  { key: "mi_score", label: "MI", format: v => fmt(v, 4) },
  { key: "mi_pvalue_adj", label: "MI p (adj)", format: v => fmt(v, 4) + pStar(v) },
  { key: "dcor", label: "dCor", format: v => fmt(v, 4) },
  { key: "composite_rank", label: "Rank", format: v => (v == null ? "—" : String(v)) },
  { key: "consistent", label: "Consistent", format: v => (v ? "✓" : "—") },
  { key: "nonlinear_gain", label: "Nonlinear gain", format: v => (v ? "✓" : "—") },
];

function renderScreening(records, meta) {
  const el = document.getElementById("ml-screening");
  if (!records || records.length === 0) {
    el.innerHTML = `<div class="status">No screening results.</div>`;
    return;
  }
  const cols = SCREENING_COLS.filter(c => records.some(r => r[c.key] !== undefined));
  const sorted = [...records].sort(
    (a, b) => (a.composite_rank ?? Infinity) - (b.composite_rank ?? Infinity)
  );
  const tableHtml = `<div style="overflow-x:auto"><table class="matrix-table"><thead><tr>${
    cols.map(c => `<th>${c.label}</th>`).join("")
  }</tr></thead><tbody>${
    sorted.map(row => `<tr>${
      cols.map(c => `<td>${c.format ? c.format(row[c.key]) : (row[c.key] ?? "—")}</td>`).join("")
    }</tr>`).join("")
  }</tbody></table></div>`;

  const warnings = meta && meta.warnings;
  const warningsHtml = (warnings && warnings.length > 0)
    ? `<div class="ml-warnings">${warnings.map(w => `<div>⚠ ${w}</div>`).join("")}</div>`
    : "";

  el.innerHTML = tableHtml + warningsHtml;
}

function renderMiHeatmap(records) {
  const el = document.getElementById("ml-heatmap");
  if (!records || records.length === 0) {
    el.innerHTML = `<div class="status">No mutual-information results.</div>`;
    return;
  }
  const features = [...new Set(records.map(r => r.feature))];
  const horizons = [...new Set(records.map(r => r.horizon))].sort((a, b) => a - b);
  const byKey = new Map(records.map(r => [`${r.feature}|${r.horizon}`, r.mi_score]));
  const values = features.map(f => horizons.map(h => {
    const v = byKey.get(`${f}|${h}`);
    return v == null ? null : v;
  }));
  renderMatrixTable(el, features, horizons.map(h => `${h}D`), values);
}

function renderRedundancy(red) {
  const el = document.getElementById("ml-redundancy");
  if (!red || !red.features || red.features.length === 0) {
    el.innerHTML = `<div class="status">Redundancy matrix unavailable (needs ≥2 features).</div>`;
    return;
  }
  renderMatrixTable(el, red.features, red.features, red.values);
}

function renderMdi(records) {
  const wrap = document.getElementById("ml-mdi-wrap");
  if (!records || records.length === 0) {
    wrap.classList.add("hidden");
    if (mlState.mdiChart) {
      mlState.mdiChart.destroy();
      mlState.mdiChart = null;
    }
    return;
  }
  wrap.classList.remove("hidden");
  const top = [...records]
    .sort((a, b) => (b.mdi_importance ?? 0) - (a.mdi_importance ?? 0))
    .slice(0, 15);

  const ctx = document.getElementById("ml-mdi-chart").getContext("2d");
  if (mlState.mdiChart) mlState.mdiChart.destroy();
  mlState.mdiChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: top.map(r => `${r.feature} (${r.horizon}D)`),
      datasets: [{ label: "MDI importance", data: top.map(r => r.mdi_importance), backgroundColor: "#2563eb" }],
    },
    options: {
      indexAxis: "y",
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: { display: true, text: "Top-15 MDI importance (Tier 2)" },
      },
    },
  });
}

// ─── Saved scans ───────────────────────────────────────────────────────────
function getMlConfig() {
  return {
    factors: Array.from(mlState.selected),
    horizons: getSelectedHorizons(),
    from_date: document.getElementById("ml-from").value,
    to_date: document.getElementById("ml-to").value,
    quantiles: parseInt(document.getElementById("ml-quantiles").value, 10),
    mode: document.getElementById("ml-mode").value,
    tier2: document.getElementById("ml-tier2").checked,
  };
}

const saveScanMlBtn = document.getElementById("save-scan-mleval");
if (saveScanMlBtn) {
  saveScanMlBtn.addEventListener("click", () => saveScan("mleval", getMlConfig));
}

window.fbRestore.mleval = function (config) {
  if (!config) return;
  if (!mlState.catalog && window.fbCatalog) mlState.catalog = window.fbCatalog;
  renderMlFactorPills(); // idempotent — rebuilds pills from mlState.catalog

  if (config.from_date) document.getElementById("ml-from").value = config.from_date;
  if (config.to_date) document.getElementById("ml-to").value = config.to_date;
  if (config.quantiles) document.getElementById("ml-quantiles").value = String(config.quantiles);
  if (config.mode) document.getElementById("ml-mode").value = config.mode;
  document.getElementById("ml-tier2").checked = !!config.tier2;
  for (const h of ML_HORIZON_VALUES) {
    document.getElementById(`mlh-${h}`).checked = (config.horizons || []).includes(h);
  }

  mlState.selected = new Set();
  document.querySelectorAll("#ml-factor-pills .factor-pill").forEach(pill => {
    if ((config.factors || []).includes(pill.textContent)) {
      mlState.selected.add(pill.textContent);
      pill.classList.add("active");
    }
  });
  updateMlCount();
  runMlEval();
};

// ─── Boot ──────────────────────────────────────────────────────────────────
loadMlCatalog();

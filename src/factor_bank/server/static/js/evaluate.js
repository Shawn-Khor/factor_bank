// ─── State ─────────────────────────────────────────────────────────────────
const state = {
  catalog: null,
  selectedFactor: null,
  horizon: 21,
  quantiles: 5,
  fromDate: null,
  toDate: null,
  quantileChart: null,
  cumulativeChart: null,
  quantile_means: null,
  longshort_cumulative: null,
};

// ─── Init defaults ─────────────────────────────────────────────────────────
function todayISO() { return new Date().toISOString().slice(0, 10); }
function yearsAgoISO(n) {
  const d = new Date(); d.setFullYear(d.getFullYear() - n);
  return d.toISOString().slice(0, 10);
}

document.getElementById("from-date").value = yearsAgoISO(3);
document.getElementById("to-date").value = todayISO();
state.fromDate = document.getElementById("from-date").value;
state.toDate = document.getElementById("to-date").value;

document.getElementById("from-date").addEventListener("input", e => state.fromDate = e.target.value);
document.getElementById("to-date").addEventListener("input", e => state.toDate = e.target.value);
document.getElementById("quantiles").addEventListener("change", e => state.quantiles = parseInt(e.target.value));

// ─── Load catalog ──────────────────────────────────────────────────────────
async function loadCatalog() {
  const res = await fetch("/api/factors");
  if (!res.ok) {
    setStatus("Failed to load factor catalog.", true);
    return;
  }
  state.catalog = await res.json();
  window.fbCatalog = state.catalog; // shared cache — ML Eval tab reuses this instead of re-fetching
  if (state.catalog.date_floor) {
    document.getElementById("from-date").min = state.catalog.date_floor;
    document.getElementById("to-date").min = state.catalog.date_floor;
  }
  renderHorizons();
  renderFactors();
}

function renderHorizons() {
  const row = document.getElementById("horizon-row");
  row.innerHTML = "";
  for (const h of state.catalog.horizons) {
    const btn = document.createElement("button");
    btn.className = "horizon-btn" + (h === state.horizon ? " active" : "");
    btn.textContent = `${h}D`;
    btn.addEventListener("click", () => {
      state.horizon = h;
      row.querySelectorAll(".horizon-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    });
    row.appendChild(btn);
  }
}

function renderFactors() {
  const el = document.getElementById("factor-catalog");
  el.innerHTML = "";
  for (const [group, factors] of Object.entries(state.catalog.groups)) {
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
      pill.addEventListener("click", () => selectFactor(name));
      pills.appendChild(pill);
    }
    el.appendChild(wrap);
  }
}

function selectFactor(name) {
  state.selectedFactor = name;
  document.querySelectorAll(".factor-pill").forEach(p => {
    p.classList.toggle("active", p.textContent === name);
  });
  document.getElementById("selected-factor").textContent = name ? `· ${name}` : "";
  document.getElementById("compute-btn").disabled = !name;
  setStatus(`Selected ${name}. Click Compute.`);
}

function setStatus(msg, isError = false) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = "status" + (isError ? " error" : "");
}

// ─── Compute ───────────────────────────────────────────────────────────────
document.getElementById("compute-btn").addEventListener("click", compute);

async function compute() {
  if (!state.selectedFactor) return;
  const btn = document.getElementById("compute-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>Computing…`;
  setStatus("Running cross-sectional evaluation…");

  try {
    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        factor: state.selectedFactor,
        from_date: state.fromDate,
        to_date: state.toDate,
        horizon: state.horizon,
        n_quantiles: state.quantiles,
      }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      setStatus(data.error || `HTTP ${res.status}`, true);
      return;
    }
    renderResults(data);
    setStatus(`Computed ${state.selectedFactor} at ${state.horizon}D horizon.`);
  } catch (err) {
    setStatus(err.message || String(err), true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Compute";
  }
}

// ─── Render results ────────────────────────────────────────────────────────
function renderResults(data) {
  document.getElementById("results").classList.remove("hidden");

  // Verdict badge
  const badge = document.getElementById("verdict-badge");
  badge.textContent = data.verdict || "—";
  badge.className = `badge ${data.verdict || ""}`;

  // Meta bar
  const meta = data.meta || {};
  document.getElementById("meta-bar").innerHTML = `
    <span><strong>${data.factor}</strong></span>
    <span>·</span>
    <span><strong>${data.horizon}D</strong> horizon (${data.horizon} trading sessions)</span>
    <span>·</span>
    <span><strong>${meta.n_dates ?? "—"}</strong> dates</span>
    <span>·</span>
    <span><strong>${meta.n_tickers_universe ?? "—"}</strong> tickers (S&P 500 historical)</span>
    <span>·</span>
    <span>${data.from_date} → ${data.to_date}</span>
  `;

  // Metrics
  const m = data.metrics || {};
  const grid = document.getElementById("metrics-grid");
  const metricDefs = [
    { key: "ic",            label: "IC (Pearson)",   format: v => fmt(v) },
    { key: "rank_ic",       label: "Rank IC",         format: v => fmt(v) },
    { key: "std_ic",        label: "Std IC",          format: v => fmt(v) },
    { key: "ic_ir",         label: "IC IR",           format: v => fmt(v, 3) },
    { key: "pct_positive",  label: "% Positive",      format: v => fmtPct(v) },
    { key: "t_stat",        label: "t-stat",          format: v => fmt(v, 3) + pStar(m.p_value) },
    { key: "p_value",       label: "p-value",         format: v => fmt(v, 5) },
    { key: "n_obs",         label: "N obs",           format: v => String(v ?? "—") },
  ];
  grid.innerHTML = metricDefs.map(d => `
    <div class="metric">
      <div class="metric-label">${d.label}</div>
      <div class="metric-value ${metricClass(m[d.key])}">${d.format(m[d.key])}</div>
    </div>
  `).join("");

  // Data quality panel
  const q = data.quality || {};
  document.getElementById("quality-body").innerHTML = `
    <div>Coverage: <strong>${fmtPct(q.coverage)}</strong>
      (${Object.entries(q.coverage_by_year || {}).map(([y, v]) => `${y}: ${fmtPct(v)}`).join(" · ")})</div>
    <div>Staleness (mean fraction unchanged): <strong>${fmt(q.staleness_mean, 3)}</strong></div>
    <div>Rows: ${q.n_rows ?? "—"} · Duplicates: ${q.duplicates ?? "—"}</div>`;

  // Long/short stats bar
  const lss = data.longshort_stats || {};
  document.getElementById("ls-stats-bar").innerHTML = `
    <span>Long Q${data.n_quantiles} / Short Q1</span>
    <span>·</span>
    <span>Total return <strong>${fmtPct(lss.total_return)}</strong></span>
    <span>·</span>
    <span>Annualized Sharpe <strong>${fmt(lss.annualized_sharpe, 2)}</strong></span>
    <span>·</span>
    <span><strong>${lss.n_days ?? "—"}</strong> trading days</span>
  `;

  state.quantile_means = data.quantile_means;
  state.longshort_cumulative = data.longshort_cumulative;
  renderQuantileChart(data.quantile_means, data.n_quantiles);
  renderCumulativeChart(data.longshort_cumulative);
  applyTooltips(document.getElementById("results"));
}

function renderQuantileChart(quantileMeans, nQuantiles) {
  const ctx = document.getElementById("quantile-chart").getContext("2d");
  if (state.quantileChart) state.quantileChart.destroy();
  const labels = quantileMeans.map(q => `Q${q.quantile}`);
  const values = quantileMeans.map(q => q.mean_return);
  // Color the extreme bars
  const colors = values.map((v, i) => {
    if (i === 0)                 return "#dc2626"; // Q1 — short side
    if (i === values.length - 1) return "#16a34a"; // Q_top — long side
    return "#94a3b8"; // muted
  });
  state.quantileChart = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{ label: "Mean forward return", data: values, backgroundColor: colors }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        title: { display: true, text: `Mean ${state.horizon}D return per quantile` },
        legend: { display: false },
        tooltip: { callbacks: { label: c => (c.parsed.y * 100).toFixed(3) + "%" } },
      },
      scales: {
        y: {
          ticks: { callback: v => (v * 100).toFixed(2) + "%" },
          title: { display: true, text: "Mean forward return" },
        },
      },
    },
  });
}

function renderCumulativeChart(series) {
  const ctx = document.getElementById("cumulative-chart").getContext("2d");
  if (state.cumulativeChart) state.cumulativeChart.destroy();
  if (!series || series.length === 0) {
    state.cumulativeChart = null;
    return;
  }
  const labels = series.map(p => p.date);
  const values = series.map(p => p.value);
  state.cumulativeChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Long/short cumulative return",
        data: values,
        borderColor: "#2563eb", backgroundColor: "rgba(37,99,235,0.08)",
        fill: true, pointRadius: 0, borderWidth: 1.5, tension: 0.05,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        title: { display: true, text: `Q${state.quantiles} − Q1 long/short, compounded` },
        legend: { display: false },
        tooltip: { callbacks: { label: c => (c.parsed.y * 100).toFixed(2) + "%" } },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 8 } },
        y: {
          ticks: { callback: v => (v * 100).toFixed(0) + "%" },
          title: { display: true, text: "Cumulative return" },
        },
      },
    },
  });
}

// ─── Custom factor upload ──────────────────────────────────────────────────
const customUploadBtn = document.getElementById("custom-upload-btn");
if (customUploadBtn) {
  customUploadBtn.addEventListener("click", uploadCustomFactor);
}

async function uploadCustomFactor() {
  const nameEl = document.getElementById("custom-name");
  const fileEl = document.getElementById("custom-file");
  const statusEl = document.getElementById("custom-upload-status");
  const name = nameEl.value.trim();
  const file = fileEl.files[0];
  if (!name || !file) {
    statusEl.textContent = "Provide a name and a CSV file.";
    statusEl.className = "status error";
    return;
  }
  const formData = new FormData();
  formData.append("name", name);
  formData.append("file", file);
  statusEl.textContent = "Uploading…";
  statusEl.className = "status";
  try {
    const res = await fetch("/api/custom-factors", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok || data.error) {
      statusEl.textContent = data.error || `HTTP ${res.status}`;
      statusEl.className = "status error";
      return;
    }
    statusEl.textContent =
      `Uploaded '${data.name}': ${data.n_rows} rows, ${data.n_tickers} tickers ` +
      `(${data.date_min} → ${data.date_max}).`;
    statusEl.className = "status";
    nameEl.value = "";
    fileEl.value = "";
    await loadCatalog(); // re-fetch catalog so the Custom pill appears
  } catch (err) {
    statusEl.textContent = err.message || String(err);
    statusEl.className = "status error";
  }
}

// ─── Saved scans ───────────────────────────────────────────────────────────
function getEvaluateConfig() {
  return {
    factor: state.selectedFactor,
    from_date: state.fromDate,
    to_date: state.toDate,
    horizon: state.horizon,
    n_quantiles: state.quantiles,
  };
}

const saveScanBtn = document.getElementById("save-scan-evaluate");
if (saveScanBtn) {
  saveScanBtn.addEventListener("click", () => saveScan("evaluate", getEvaluateConfig));
}

window.fbRestore.evaluate = async function (config) {
  if (!config) return;
  // Same catalog-ready guard the other tabs' restores carry (I-4): boot fires
  // three independent /api/factors fetches (evaluate/mleval/lab), and this
  // restore can otherwise run before evaluate's own `state.catalog` is set,
  // making renderHorizons() throw on a null catalog and die silently.
  if (!state.catalog) {
    if (window.fbCatalog) {
      state.catalog = window.fbCatalog;
    } else {
      await window.fbWaitForCatalog();
      if (!state.catalog && window.fbCatalog) state.catalog = window.fbCatalog;
    }
  }
  if (!state.catalog) {
    setStatus("Couldn't load the factor catalog — restore aborted.", true);
    return;
  }
  if (config.from_date) {
    document.getElementById("from-date").value = config.from_date;
    state.fromDate = config.from_date;
  }
  if (config.to_date) {
    document.getElementById("to-date").value = config.to_date;
    state.toDate = config.to_date;
  }
  if (config.n_quantiles) {
    state.quantiles = config.n_quantiles;
    document.getElementById("quantiles").value = String(config.n_quantiles);
  }
  if (config.horizon) state.horizon = config.horizon;
  // Re-render so the horizon/factor widgets pick up state.catalog + state.horizon
  // (idempotent — safe to call even if already rendered).
  renderHorizons();
  renderFactors();
  if (config.factor) selectFactor(config.factor);
  compute();
};

// ─── Cross-tab entry point (Factor Lab "Open in Evaluate") ────────────────
// `factor` may be a generated "base__transform" name from the lab grid that
// isn't one of the catalog pills — there's nothing to highlight, so we set
// the state directly and surface the name via the status line instead.
window.fbOpenInEvaluate = function (factor, from_date, to_date, horizon) {
  if (from_date) {
    document.getElementById("from-date").value = from_date;
    state.fromDate = from_date;
  }
  if (to_date) {
    document.getElementById("to-date").value = to_date;
    state.toDate = to_date;
  }
  if (horizon) {
    state.horizon = horizon;
    if (state.catalog) renderHorizons();
  }
  state.selectedFactor = factor || null;
  document.querySelectorAll(".factor-pill").forEach(p => {
    p.classList.toggle("active", p.textContent === factor);
  });
  document.getElementById("selected-factor").textContent = factor ? `· ${factor}` : "";
  document.getElementById("compute-btn").disabled = !factor;
  switchTab("evaluate");
  if (!factor) return;
  setStatus(`Opened '${factor}' from Factor Lab. Running…`);
  compute();
};

// ─── Quantile analysis CSV export ──────────────────────────────────────────
const csvQuantileBtn = document.getElementById("csv-quantile");
if (csvQuantileBtn) {
  csvQuantileBtn.addEventListener("click", () => {
    const quantileCsv = state.quantile_means ? recordsToCsv(state.quantile_means) : "";
    const cumulativeCsv = state.longshort_cumulative ? recordsToCsv(state.longshort_cumulative) : "";
    const combined = [quantileCsv, "", cumulativeCsv].filter(s => s || s === "").join("\r\n");
    if (combined.trim()) {
      downloadCsv("quantile_analysis.csv", combined);
    }
  });
}

// ─── Boot ──────────────────────────────────────────────────────────────────
loadCatalog();

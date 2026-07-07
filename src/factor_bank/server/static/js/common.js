// ─── Shared formatting helpers ──────────────────────────────────────────────
function fmt(v, dp = 4) {
  if (v == null || !isFinite(v)) return "—";
  return Number(v).toFixed(dp);
}
function fmtPct(v) {
  if (v == null || !isFinite(v)) return "—";
  return (v * 100).toFixed(2) + "%";
}
function pStar(p) {
  if (p == null) return "";
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "";
}
// `mode` picks the color semantics for a metric tile:
//  - "pos" (default): large positive is good (green), large negative is bad
//    (red) — e.g. IC, Rank IC, t-stat.
//  - "low-good": small values are good (green), large values are bad (red)
//    — e.g. p-value, where LOW means statistically significant.
//  - "neutral": never colored — e.g. std_ic, n_obs, where there's no
//    universal "good" direction to signal.
function metricClass(v, mode = "pos") {
  if (v == null || !isFinite(v)) return "";
  if (mode === "neutral") return "";
  if (mode === "low-good") {
    if (v < 0.05) return "green";
    if (v > 0.05) return "red";
    return "";
  }
  if (v >= 0.05) return "green";
  if (v <= -0.05) return "red";
  return "";
}

// ─── Badges ──────────────────────────────────────────────────────────────────
function badgeHtml(cls, text) {
  return `<span class="badge ${cls}">${text}</span>`;
}

// ─── CSV export ──────────────────────────────────────────────────────────────
function tableToCsv(tableEl) {
  if (!tableEl) return "";
  const escapeCell = text => {
    const s = text == null ? "" : String(text);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const rows = Array.from(tableEl.querySelectorAll("tr"));
  return rows
    .map(row => Array.from(row.children).map(cell => escapeCell(cell.textContent.trim())).join(","))
    .join("\r\n");
}

function downloadCsv(filename, csv) {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function recordsToCsv(records) {
  if (!records || !records.length) return "";
  const cols = Object.keys(records[0]);
  const esc = v => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [cols.join(","), ...records.map(r => cols.map(c => esc(r[c])).join(","))].join("\r\n");
}

// Binds a "⬇ CSV" button (by id) to export whatever <table> currently lives
// inside a container (by id). The table is resolved at CLICK time (not bind
// time) so the button always exports the most recently rendered table, even
// after a re-render has replaced the container's contents.
function bindCsvExport(buttonId, containerId, filename) {
  const btn = document.getElementById(buttonId);
  if (!btn) return;
  btn.addEventListener("click", () => {
    const container = document.getElementById(containerId);
    const table = container && container.querySelector("table");
    if (!table) return;
    const name = typeof filename === "function" ? filename() : filename;
    downloadCsv(name, tableToCsv(table));
  });
}

// ─── Metric glossary + tooltips ─────────────────────────────────────────────
// Keyed by the lowercased label text as it appears in the UI (th / .metric-label
// elements). A few entries carry aliases where different tabs word the same
// concept slightly differently (e.g. "Horizon" vs "Chosen Horizon").
const GLOSSARY = {
  "ic (pearson)": "Mean cross-sectional Pearson correlation between the factor value and the forward return — captures linear predictive power; sensitive to outliers.",
  "ic": "Mean cross-sectional Pearson correlation between the factor value and the forward return — captures linear predictive power; sensitive to outliers.",
  "rank ic": "Mean cross-sectional Spearman correlation between factor rank and forward-return rank — captures monotonic predictive power and is more robust to outliers than IC.",
  "std ic": "Standard deviation of the daily IC series — higher means the factor's edge is noisier and less consistent over time.",
  "ic ir": "Information ratio: mean(Rank IC) divided by std(Rank IC) — a risk-adjusted measure of predictive power; above 0.3 is good, above 0.5 is excellent.",
  "% positive": "Fraction of dates where Rank IC is positive — above roughly 55% suggests the factor's direction is consistent over time.",
  "t-stat": "t-statistic testing whether the mean Rank IC is significantly different from zero.",
  "p-value": "Two-tailed probability of seeing this t-stat (or a more extreme one) if the true mean IC were zero — smaller is stronger evidence of a real effect.",
  "q-value": "The p-value after Benjamini-Hochberg false-discovery-rate correction for testing many candidates at once — guards against multiple-testing false positives.",
  "n obs": "Number of dates with a valid, non-missing cross-sectional signal that went into this metric.",
  "mi": "Mutual information — quantifies any statistical dependence, linear or non-linear, between the factor and the forward return (in nats); zero only if truly independent.",
  "dcor": "Distance correlation — like Pearson correlation but detects any dependence, not just linear or monotone; bounded in [0, 1] and zero only if independent.",
  "composite rank": "Overall rank of the feature across the screening's combined criteria (IC, MI, dCor, consistency) — 1 is the best-ranked feature.",
  "consistent": "Whether the factor's direction agrees across the horizons/folds tested rather than flipping sign — a stability check, not a strength measure.",
  "nl gain": "Whether a non-linear measure (MI or dCor) meaningfully exceeds what the linear IC alone would predict — flags factors that may have exploitable non-linear structure.",
  "nonlinear gain": "Whether a non-linear measure (MI or dCor) meaningfully exceeds what the linear IC alone would predict — flags factors that may have exploitable non-linear structure.",
  "holdout rank ic": "Rank IC recomputed on the holdout (out-of-sample) date range after the candidate was selected on the training range — the key check against overfitting.",
  "verdict": "Overall strength label (Strong / Moderate / Weak / Insignificant) assigned from the factor's Rank IC, IC IR, and statistical significance.",
  "mean ic": "Mean cross-sectional IC for the feature at its chosen horizon.",
  "chosen horizon": "The forward-return horizon (in trading sessions) selected for this feature during screening.",
  "horizon": "The forward-return horizon (in trading sessions) selected for this feature during screening.",
};

function applyTooltips(scopeEl) {
  const root = scopeEl || document;
  root.querySelectorAll("th, .metric-label").forEach(el => {
    const key = el.textContent.trim().toLowerCase();
    const def = GLOSSARY[key];
    if (def) el.title = def;
  });
}

// ─── Job polling + heatmap helpers (ML Eval) ────────────────────────────────
async function pollJob(jobId, onProgress, intervalMs = 2000) {
  for (;;) {
    const res = await fetch(`/api/jobs/${jobId}`);
    const rec = await res.json();
    // A 404 (unknown job id — server restarted, or the job was evicted from
    // the 20-job history) or any body missing `status` used to fall through
    // to onProgress(rec), rendering "undefined" and polling forever. Treat
    // both as a hard, user-visible failure instead.
    if (!res.ok || !rec.status) {
      throw new Error("job no longer exists — the server may have restarted");
    }
    if (rec.status === "done") return rec.result;
    if (rec.status === "error") throw new Error(rec.error + "\n" + (rec.detail || ""));
    onProgress(rec);
    await new Promise(res => setTimeout(res, intervalMs));
  }
}

function heatColor(v, vmin, vmax) {
  if (v == null || !isFinite(v)) return "";
  const t = vmax > vmin ? (v - vmin) / (vmax - vmin) : 0;
  const alpha = (0.08 + 0.72 * t).toFixed(3);
  return `background: rgba(37, 99, 235, ${alpha}); color: ${t > 0.6 ? "#fff" : "inherit"}`;
}

function renderMatrixTable(el, rowLabels, colLabels, values) {
  const flat = values.flat().filter(v => v != null && isFinite(v));
  const vmin = Math.min(...flat), vmax = Math.max(...flat);
  el.innerHTML = `<div style="overflow-x:auto"><table class="matrix-table"><thead><tr><th></th>${
    colLabels.map(c => `<th>${c}</th>`).join("")
  }</tr></thead><tbody>${
    rowLabels.map((r, i) => `<tr><th>${r}</th>${
      values[i].map(v => `<td style="${heatColor(v, vmin, vmax)}">${fmt(v, 3)}</td>`).join("")
    }</tr>`).join("")
  }</tbody></table></div>`;
}

// ─── Tab switching ───────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t =>
    t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-content").forEach(c =>
    c.classList.toggle("active", c.id === `tab-${name}`));
}
document.querySelectorAll(".tab:not([disabled])").forEach(t =>
  t.addEventListener("click", () => switchTab(t.dataset.tab)));

// ─── Saved scans ─────────────────────────────────────────────────────────────
// Populated by evaluate.js / mleval.js (and, later, lab.js) — each tab
// registers a restore(config) function here so ?scan= links can rehydrate it.
window.fbRestore = {};

function fbSetStatusFor(tab, msg, isError = false) {
  if (tab === "evaluate" && typeof setStatus === "function") { setStatus(msg, isError); return; }
  if (tab === "mleval" && typeof setMlStatus === "function") { setMlStatus(msg, isError); return; }
  // eslint-disable-next-line no-console
  console.log(`[${tab}] ${msg}`);
}

async function saveScan(tab, getConfig) {
  const name = prompt("Name this scan:");
  if (name == null || !name.trim()) return;
  try {
    const res = await fetch("/api/scans", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name.trim(), tab, config: getConfig() }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      fbSetStatusFor(tab, data.error || `HTTP ${res.status}`, true);
      return;
    }
    const url = `${location.origin}/?scan=${data.id}`;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url).catch(() => {});
    }
    fbSetStatusFor(tab, `Saved as "${name.trim()}". Share URL copied to clipboard: ${url}`);
  } catch (err) {
    fbSetStatusFor(tab, err.message || String(err), true);
  }
}

function fbWaitForCatalog(timeoutMs = 5000) {
  return new Promise(resolve => {
    const start = Date.now();
    (function poll() {
      if (window.fbCatalog || Date.now() - start > timeoutMs) { resolve(); return; }
      setTimeout(poll, 50);
    })();
  });
}

async function fbRestoreFromQuery() {
  const params = new URLSearchParams(location.search);
  const scanId = params.get("scan");
  if (!scanId) return;
  try {
    const res = await fetch(`/api/scans/${scanId}`);
    if (!res.ok) return;
    const rec = await res.json();
    switchTab(rec.tab);
    await fbWaitForCatalog();
    const restore = window.fbRestore[rec.tab];
    if (typeof restore === "function") {
      restore(rec.config);
    } else {
      fbSetStatusFor(rec.tab, `Restore isn't available yet for the "${rec.tab}" tab.`, true);
    }
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error(err);
  }
}
fbRestoreFromQuery();

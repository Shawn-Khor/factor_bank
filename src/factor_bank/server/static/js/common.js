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
function metricClass(v, posIsGood = true) {
  if (v == null || !isFinite(v)) return "";
  if (posIsGood) {
    if (v >= 0.05) return "green";
    if (v <= -0.05) return "red";
  }
  return "";
}

// ─── Job polling + heatmap helpers (ML Eval) ────────────────────────────────
async function pollJob(jobId, onProgress, intervalMs = 2000) {
  for (;;) {
    const rec = await fetch(`/api/jobs/${jobId}`).then(r => r.json());
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

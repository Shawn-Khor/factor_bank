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

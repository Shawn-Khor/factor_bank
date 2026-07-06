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

// ─── Tab switching ───────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t =>
    t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-content").forEach(c =>
    c.classList.toggle("active", c.id === `tab-${name}`));
}
document.querySelectorAll(".tab:not([disabled])").forEach(t =>
  t.addEventListener("click", () => switchTab(t.dataset.tab)));

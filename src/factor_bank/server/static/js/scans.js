// ─── Scans tab ───────────────────────────────────────────────────────────────
function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : String(s);
  return div.innerHTML;
}

function setScansStatus(msg, isError = false) {
  const el = document.getElementById("scans-status");
  if (!el) return;
  el.textContent = msg;
  el.className = "status" + (isError ? " error" : "");
}

async function loadScans() {
  setScansStatus("Loading…");
  try {
    const res = await fetch("/api/scans");
    const data = await res.json();
    if (!res.ok || data.error) {
      setScansStatus(data.error || `HTTP ${res.status}`, true);
      return;
    }
    const scans = data.scans || [];
    renderScansTable(scans);
    setScansStatus(scans.length === 1 ? "1 saved scan." : `${scans.length} saved scans.`);
  } catch (err) {
    setScansStatus(err.message || String(err), true);
  }
}

function renderScansTable(scans) {
  const el = document.getElementById("scans-table");
  if (!el) return;
  if (scans.length === 0) {
    el.innerHTML = `<div class="status">No saved scans yet — use "Save scan" on the Evaluate or ML Eval tabs.</div>`;
    return;
  }
  el.innerHTML = `<div style="overflow-x:auto"><table class="matrix-table"><thead><tr>
    <th style="text-align:left">Name</th><th>Tab</th><th>Created</th><th></th><th></th>
  </tr></thead><tbody>${
    scans.map(s => `<tr>
      <td style="text-align:left">${escapeHtml(s.name)}</td>
      <td>${escapeHtml(s.tab)}</td>
      <td>${escapeHtml(new Date(s.created_at * 1000).toLocaleString())}</td>
      <td><button class="save-scan-btn" data-open="${escapeHtml(s.id)}">Open</button></td>
      <td><button class="save-scan-btn" data-delete="${escapeHtml(s.id)}">Delete</button></td>
    </tr>`).join("")
  }</tbody></table></div>`;

  el.querySelectorAll("[data-open]").forEach(btn =>
    btn.addEventListener("click", () => { location.href = `/?scan=${btn.dataset.open}`; }));
  el.querySelectorAll("[data-delete]").forEach(btn =>
    btn.addEventListener("click", () => deleteScan(btn.dataset.delete)));
}

async function deleteScan(id) {
  try {
    await fetch(`/api/scans/${id}`, { method: "DELETE" });
  } finally {
    loadScans();
  }
}

// Refresh whenever the Scans tab is opened (picks up scans saved elsewhere).
const scansTabBtn = document.querySelector('.tab[data-tab="scans"]');
if (scansTabBtn) scansTabBtn.addEventListener("click", loadScans);

// ─── Boot ──────────────────────────────────────────────────────────────────
loadScans();

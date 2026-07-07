// ─── State ─────────────────────────────────────────────────────────────────
const labState = {
  catalog: null,
  horizon: 21,
  jobId: null,
  lastResult: null,
};

document.getElementById("lab-from").value = yearsAgoISO(3);
document.getElementById("lab-to").value = todayISO();

function labEscapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : String(s);
  return div.innerHTML;
}

// ─── Load catalog (horizons) + candidate grid header ───────────────────────
async function loadLabCatalog() {
  if (window.fbCatalog) {
    labState.catalog = window.fbCatalog;
  } else {
    const res = await fetch("/api/factors");
    if (!res.ok) {
      setLabStatus("Failed to load factor catalog.", true);
      return;
    }
    labState.catalog = await res.json();
    window.fbCatalog = labState.catalog;
  }
  if (labState.catalog.date_floor) {
    document.getElementById("lab-from").min = labState.catalog.date_floor;
    document.getElementById("lab-to").min = labState.catalog.date_floor;
  }
  renderLabHorizons();
  loadLabCandidateInfo();
}

function renderLabHorizons() {
  const row = document.getElementById("lab-horizon-row");
  row.innerHTML = "";
  for (const h of labState.catalog.horizons) {
    const btn = document.createElement("button");
    btn.className = "horizon-btn" + (h === labState.horizon ? " active" : "");
    btn.textContent = `${h}D`;
    btn.addEventListener("click", () => {
      labState.horizon = h;
      row.querySelectorAll(".horizon-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    });
    row.appendChild(btn);
  }
}

async function loadLabCandidateInfo() {
  try {
    const res = await fetch("/api/lab/candidates");
    if (!res.ok) return;
    const data = await res.json();
    document.getElementById("lab-candidates-count").textContent =
      `· ${data.n_candidates} candidates · ${data.transforms.length} transforms`;
  } catch (err) {
    // non-fatal — header annotation only
  }
}

function setLabStatus(msg, isError = false) {
  const el = document.getElementById("lab-status");
  el.textContent = msg;
  el.className = "status" + (isError ? " error" : "");
}

// ─── Run ───────────────────────────────────────────────────────────────────
document.getElementById("lab-run").addEventListener("click", runLabScreen);

async function runLabScreen() {
  const fromDate = document.getElementById("lab-from").value;
  const toDate = document.getElementById("lab-to").value;
  const topK = parseInt(document.getElementById("lab-top-k").value, 10) || 30;
  const horizon = labState.horizon;

  const btn = document.getElementById("lab-run");
  const progressEl = document.getElementById("lab-progress");
  const progressText = document.getElementById("lab-progress-text");
  const elapsedEl = document.getElementById("lab-elapsed");

  btn.disabled = true;
  document.getElementById("lab-results").classList.add("hidden");
  setLabStatus("Submitting job…");

  let elapsedTimer = null;
  const startedAt = Date.now();

  try {
    const res = await fetch("/api/lab/screen", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        horizon,
        from_date: fromDate,
        to_date: toDate,
        top_k: topK,
      }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      setLabStatus(data.error || `HTTP ${res.status}`, true);
      return;
    }

    labState.jobId = data.job_id;
    progressEl.classList.remove("hidden");
    progressText.textContent = "queued";
    elapsedEl.textContent = "0s";
    elapsedTimer = setInterval(() => {
      elapsedEl.textContent = `${Math.floor((Date.now() - startedAt) / 1000)}s`;
    }, 1000);
    setLabStatus(`Running job ${data.job_id}…`);

    const result = await pollJob(data.job_id, rec => {
      if (rec.status === "queued" && rec.n_ahead > 0) {
        progressText.textContent = `queued behind ${rec.n_ahead} job(s)`;
      } else {
        progressText.textContent = rec.progress || rec.status;
      }
    });
    labState.lastResult = result;
    renderLabResults(result, { fromDate, toDate, horizon });
    setLabStatus("Done.");
  } catch (err) {
    setLabStatus(err.message || String(err), true);
  } finally {
    if (elapsedTimer) clearInterval(elapsedTimer);
    progressEl.classList.add("hidden");
    btn.disabled = false;
  }
}

// ─── Render results ────────────────────────────────────────────────────────
function renderLabResults(result, ctx) {
  document.getElementById("lab-results").classList.remove("hidden");
  const meta = result.meta || {};
  const split = result.split || {};
  document.getElementById("lab-meta").innerHTML = `
    <span><strong>${meta.horizon ?? ctx.horizon}D</strong> horizon</span>
    <span>·</span>
    <span><strong>${result.n_candidates ?? "—"}</strong> candidates screened</span>
    <span>·</span>
    <span>Train <strong>${split.train_dates ?? "—"}</strong> / holdout <strong>${split.holdout_dates ?? "—"}</strong> dates</span>
    <span>·</span>
    <span>${meta.from_date ?? ctx.fromDate} → ${meta.to_date ?? ctx.toDate}</span>
  `;
  renderLabLeaderboard(result.leaderboard || [], ctx);
  renderLabSkipped(result.n_skipped || 0, result.skipped || []);
  applyTooltips(document.getElementById("lab-results"));
}

function renderLabLeaderboard(rows, ctx) {
  const el = document.getElementById("lab-leaderboard");
  if (!rows.length) {
    el.innerHTML = `<div class="status">No leaderboard rows.</div>`;
    return;
  }
  const bodyRows = rows.map((r, i) => {
    const flipBadge = r.sign_flip
      ? `<span class="badge WEAK" style="margin-left:6px" title="Sign flipped between train and holdout">flip</span>`
      : "";
    return `<tr>
      <td>${i + 1}</td>
      <td style="text-align:left; font-family: ui-monospace, monospace">${labEscapeHtml(r.candidate)}</td>
      <td>${fmt(r.train_rank_ic, 4)}</td>
      <td>${fmt(r.ic_ir, 3)}</td>
      <td>${fmt(r.q_value, 4)}${pStar(r.q_value)}</td>
      <td>${fmt(r.mi, 4)}</td>
      <td>${fmt(r.dcor, 4)}</td>
      <td>${fmt(r.holdout_rank_ic, 4)}${flipBadge}</td>
      <td><button class="save-scan-btn lab-open-btn" data-candidate="${labEscapeHtml(r.candidate)}">Open in Evaluate</button></td>
    </tr>`;
  }).join("");
  el.innerHTML = `<div style="overflow-x:auto"><table class="matrix-table"><thead><tr>
    <th>Rank</th><th style="text-align:left">Candidate</th><th>Train Rank IC</th><th>IC IR</th>
    <th>q-value</th><th>MI</th><th>dCor</th><th>Holdout Rank IC</th><th></th>
  </tr></thead><tbody>${bodyRows}</tbody></table></div>`;

  el.querySelectorAll(".lab-open-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const candidate = btn.dataset.candidate;
      if (typeof window.fbOpenInEvaluate === "function") {
        window.fbOpenInEvaluate(candidate, ctx.fromDate, ctx.toDate, ctx.horizon);
      }
    });
  });
}

function renderLabSkipped(nSkipped, names) {
  const details = document.getElementById("lab-skipped-details");
  const summary = document.getElementById("lab-skipped-summary");
  const body = document.getElementById("lab-skipped-body");
  if (!nSkipped) {
    details.classList.add("hidden");
    return;
  }
  details.classList.remove("hidden");
  summary.textContent = `${nSkipped} candidates skipped`;
  body.textContent = names.join(", ");
}

// ─── Saved scans ───────────────────────────────────────────────────────────
function getLabConfig() {
  return {
    horizon: labState.horizon,
    from_date: document.getElementById("lab-from").value,
    to_date: document.getElementById("lab-to").value,
    top_k: parseInt(document.getElementById("lab-top-k").value, 10) || 30,
  };
}

const saveScanLabBtn = document.getElementById("save-scan-lab");
if (saveScanLabBtn) {
  saveScanLabBtn.addEventListener("click", () => saveScan("lab", getLabConfig));
}

window.fbRestore.lab = function (config) {
  if (!config) return;
  if (!labState.catalog && window.fbCatalog) labState.catalog = window.fbCatalog;
  if (config.from_date) document.getElementById("lab-from").value = config.from_date;
  if (config.to_date) document.getElementById("lab-to").value = config.to_date;
  if (config.top_k) document.getElementById("lab-top-k").value = String(config.top_k);
  if (config.horizon) labState.horizon = config.horizon;
  if (labState.catalog) renderLabHorizons(); // idempotent — reflects labState.horizon
  runLabScreen();
};

// ─── CSV export ────────────────────────────────────────────────────────────
bindCsvExport("lab-leaderboard-csv", "lab-leaderboard", "lab_leaderboard.csv");

// ─── Boot ──────────────────────────────────────────────────────────────────
loadLabCatalog();

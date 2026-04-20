/* ================================================================
   CarbonCast — Script
   State: landing → loading → results
   All logic clean, product-grade
================================================================ */

const API = "http://127.0.0.1:5000";

let currentMode = "explain";
let lastYear    = null;

// ── Year input sync ───────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("yearInput");
  if (input) {
    input.addEventListener("keydown", e => {
      if (e.key === "Enter") runPrediction();
    });
  }
});

// ── Mode ──────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  document.getElementById("segExplain").classList.toggle("active", mode === "explain");
  document.getElementById("segPolicy").classList.toggle("active",  mode === "policy");
}

// ── Year chips ────────────────────────────────────────────────
function setYear(y) {
  const input = document.getElementById("yearInput");
  if (input) input.value = y;
}

// ── State transitions ─────────────────────────────────────────
function showView(name) {
  ["viewLanding","viewLoading","viewResults"].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    if (id === name) {
      el.classList.remove("hidden");
    } else {
      el.classList.add("hidden");
    }
  });
}

function resetView() {
  showView("viewLanding");
  document.getElementById("predictBtn").disabled = false;
  // Reset loading steps
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById(`ls${i}`);
    if (el) el.className = "lstep";
  }
}

// ── Pipeline step animation ───────────────────────────────────
function pipeStep(n) {
  return new Promise(resolve => {
    setTimeout(() => {
      for (let i = 1; i <= 4; i++) {
        const el = document.getElementById(`ls${i}`);
        if (!el) continue;
        if (i < n)  el.className = "lstep done";
        if (i === n) el.className = "lstep active";
        if (i > n)  el.className = "lstep";
      }
      resolve();
    }, 400);
  });
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ═══════════════════════════════════════════════════════════
// MAIN PREDICTION
// ═══════════════════════════════════════════════════════════
async function runPrediction() {
  const input = document.getElementById("yearInput");
  const year  = parseInt(input.value);

  if (!input.value.trim()) return showToast("Please enter a target year.");
  if (isNaN(year) || year < 1965 || year > 2100)
    return showToast("Enter a year between 1965 and 2100.");

  lastYear = year;
  document.getElementById("loadingYear").textContent = year;
  showView("viewLoading");
  document.getElementById("predictBtn").disabled = true;

  const endpoint = currentMode === "policy"
    ? "/predict/explain-policy"
    : "/predict/explain";

  try {
    await pipeStep(1);

    const res = await fetch(`${API}${endpoint}`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ year }),
    });

    await pipeStep(2);
    await delay(200);
    await pipeStep(3);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Prediction failed");
    }

    const data = await res.json();

    if (currentMode === "policy") {
      await pipeStep(4);
      await delay(200);
    }

    await delay(300);
    renderResults(data, year);

  } catch (err) {
    resetView();
    showToast(err.message || "Cannot connect to server. Is the Flask API running?");
  }
}

// ═══════════════════════════════════════════════════════════
// RENDER RESULTS
// ═══════════════════════════════════════════════════════════
function renderResults(data, year) {
  const co2      = data.predicted_co2_per_capita;
  const baseline = data.baseline;
  const delta    = co2 - baseline;
  const pct      = ((delta / baseline) * 100).toFixed(1);

  // Header
  document.getElementById("resYear").textContent = year;

  // Gauge
  animateGauge(co2);

  // Readout
  document.getElementById("grVal").textContent = co2.toFixed(3);

  const grDelta = document.getElementById("grDelta");
  if (Math.abs(delta) < 0.005) {
    grDelta.className = "gr-delta flat";
    grDelta.textContent = "≈ At baseline";
  } else if (delta > 0) {
    grDelta.className = "gr-delta up";
    grDelta.textContent = `↑ +${delta.toFixed(3)} t  (+${pct}% vs baseline)`;
  } else {
    grDelta.className = "gr-delta down";
    grDelta.textContent = `↓ ${delta.toFixed(3)} t  (${pct}% vs baseline)`;
  }

  // Side metrics
  document.getElementById("smBaselineVal").textContent = baseline.toFixed(3) + " t";

  const smDeltaVal  = document.getElementById("smDeltaVal");
  const smDeltaNote = document.getElementById("smDeltaNote");
  smDeltaVal.textContent  = (delta >= 0 ? "+" : "") + delta.toFixed(3) + " t";
  smDeltaNote.textContent = pct + "% vs historical avg";
  smDeltaVal.style.color  = delta > 0 ? "var(--danger)" : delta < 0 ? "var(--success)" : "var(--text)";

  // Top driver
  const profile  = data.responsibility_profile;
  const explan   = data.explanation;
  let contribs   = {};

  if (profile) {
    profile.forEach(p => { contribs[p.factor] = p.impact_value; });
  } else if (explan) {
    contribs = explan.contributions || {};
  }

  const topDriver = Object.entries(contribs)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))[0];

  const DRIVER_LABELS = {
    energy_per_capita:       "Energy per Capita",
    fossil_share_energy:     "Fossil Share of Energy",
    energy_per_gdp:          "Energy Intensity (GDP)",
    renewables_share_energy: "Renewables Share",
  };

  document.getElementById("smTopDriverVal").textContent =
    topDriver ? (DRIVER_LABELS[topDriver[0]] || topDriver[0]) : "—";

  // Render panels
  renderSHAP(data);
  renderDrivers(data);

  // Policy
  const tabPolicy = document.getElementById("tabPolicy");
  if (data.policy_insights) {
    renderPolicy(data);
    tabPolicy.style.opacity = "1";
  } else {
    tabPolicy.style.opacity = "0.5";
    document.getElementById("policyContent").innerHTML = `
      <div class="policy-empty">
        <div class="pe-icon">🏛</div>
        <p class="pe-title">Policy AI not loaded</p>
        <p class="pe-body">Policy insights were not included in this request. Click below to load them for year ${year}.</p>
        <button class="btn-outline" onclick="loadPolicy()">Generate Policy Insights</button>
      </div>`;
  }

  switchTab("shap");
  showView("viewResults");
  document.getElementById("predictBtn").disabled = false;
}

// ═══════════════════════════════════════════════════════════
// GAUGE ANIMATION
// ═══════════════════════════════════════════════════════════
function animateGauge(co2Val) {
  // Scale: 0 → 5+ tonnes maps to 0° → 180° on the semicircle
  const maxVal  = 5;
  const ratio   = Math.min(co2Val / maxVal, 1);

  // Arc length of the semicircle path (approx 377px for r=120)
  const arcLen  = 377;
  const offset  = arcLen * (1 - ratio);

  // Color: green → amber → red based on value
  let color;
  if (co2Val < 1.5)      color = "#5A7A4A";
  else if (co2Val < 2.5) color = "#B07830";
  else if (co2Val < 3.5) color = "#C4522A";
  else                   color = "#8C2020";

  const fill = document.getElementById("gaugeFill");
  if (fill) {
    fill.style.strokeDashoffset = offset;
    fill.style.stroke = color;
  }

  // Needle: 0° = pointing left (-90°), 180° = pointing right (+90°)
  const angle  = -90 + ratio * 180;
  const needle = document.getElementById("gaugeNeedle");
  if (needle) {
    needle.setAttribute("transform", `rotate(${angle}, 150, 175)`);
    needle.style.stroke = color;
  }

  // Pivot dot color
  const pivot = document.querySelector(".gauge-svg circle:last-of-type");
  if (pivot) pivot.style.fill = color;
}

// ═══════════════════════════════════════════════════════════
// SHAP RENDERING
// ═══════════════════════════════════════════════════════════
function renderSHAP(data) {
  const profile  = data.responsibility_profile;
  const explan   = data.explanation;
  const baseline = data.baseline;
  const co2      = data.predicted_co2_per_capita;

  let contribs = {};
  let pcts     = {};

  if (profile) {
    profile.forEach(p => {
      contribs[p.factor] = p.impact_value;
      pcts[p.factor]     = p.impact_percent;
    });
  } else if (explan) {
    contribs = explan.contributions || {};
    pcts     = explan.percentages   || {};
  }

  const LABELS = {
    energy_per_capita:       "Energy per Capita",
    fossil_share_energy:     "Fossil Share of Energy",
    energy_per_gdp:          "Energy Intensity (GDP)",
    renewables_share_energy: "Renewables Share",
  };

  const sorted = Object.entries(contribs)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

  const maxAbs = Math.max(...sorted.map(([,v]) => Math.abs(v))) || 1;

  // Bar list
  const list = document.getElementById("shapList");
  list.innerHTML = sorted.map(([feat, val]) => {
    const pct   = pcts[feat] || 0;
    const barW  = (Math.abs(val) / maxAbs * 100).toFixed(1);
    const isPos = val >= 0;
    const label = LABELS[feat] || feat.replace(/_/g, " ");
    const sign  = isPos ? "+" : "";

    return `
      <div class="shap-row">
        <span class="shap-name">${label}</span>
        <div class="shap-track">
          <div class="shap-bar ${isPos ? "pos" : "neg"}"
               style="width:0" data-target="${barW}%"></div>
        </div>
        <div class="shap-meta">
          <span class="shap-val ${isPos ? "pos" : "neg"}">${sign}${val.toFixed(4)} t</span>
          <span class="shap-pct">${pct.toFixed(1)}% impact</span>
        </div>
      </div>`;
  }).join("");

  // Animate bars
  requestAnimationFrame(() => requestAnimationFrame(() => {
    list.querySelectorAll(".shap-bar[data-target]").forEach(b => {
      b.style.width = b.dataset.target;
    });
  }));

  // Waterfall summary
  const totalD  = sorted.reduce((s, [,v]) => s + v, 0);
  const sum     = document.getElementById("shapSum");
  sum.innerHTML = `
    <p class="shap-sum-title">Prediction Breakdown</p>
    <div class="sum-row">
      <span class="sum-label">Historical baseline</span>
      <span class="sum-sep">=</span>
      <span class="sum-num">${baseline.toFixed(3)} t</span>
    </div>
    ${sorted.map(([feat, val]) => {
      const label = LABELS[feat] || feat.replace(/_/g," ");
      const sign  = val >= 0 ? "+" : "";
      const color = val >= 0 ? "var(--danger)" : "var(--success)";
      return `
        <div class="sum-row">
          <span class="sum-label">+ ${label}</span>
          <span class="sum-sep">=</span>
          <span class="sum-num" style="color:${color}">${sign}${val.toFixed(4)} t</span>
        </div>`;
    }).join("")}
    <div class="sum-row" style="padding-top:0.875rem;border-top:1px solid var(--border);margin-top:0.5rem;">
      <span class="sum-label" style="font-weight:600;color:var(--text)">Predicted CO₂</span>
      <span class="sum-sep">=</span>
      <span class="sum-num terra">${co2.toFixed(3)} t / person</span>
    </div>`;
}

// ═══════════════════════════════════════════════════════════
// DRIVERS RENDERING
// ═══════════════════════════════════════════════════════════
function renderDrivers(data) {
  const drivers = data.projected_drivers || {};
  const META = {
    energy_per_capita:       { label: "Energy per Capita",       unit: "kWh / person" },
    fossil_share_energy:     { label: "Fossil Share of Energy",  unit: "% of total" },
    energy_per_gdp:          { label: "Energy Intensity (GDP)",  unit: "kWh per USD" },
    renewables_share_energy: { label: "Renewables Share",        unit: "% of total" },
  };

  document.getElementById("driversGrid").innerHTML =
    Object.entries(drivers).map(([k, v]) => {
      const m = META[k] || { label: k.replace(/_/g," "), unit: "" };
      return `
        <div class="driver-card">
          <p class="dc-label">${m.label}</p>
          <p class="dc-value">${v.toFixed(2)}</p>
          <p class="dc-unit">${m.unit}</p>
        </div>`;
    }).join("");
}

// ═══════════════════════════════════════════════════════════
// POLICY RENDERING
// ═══════════════════════════════════════════════════════════
function renderPolicy(data) {
  const insights = data.policy_insights || [];
  const isGenAI  = data.genai_enabled;

  let html = `
    <div class="policy-header">
      <span class="${isGenAI ? "genai-badge" : "fallback-badge"}">
        ${isGenAI ? "✦ AI-Generated · Llama 3.2" : "Rule-based fallback"}
      </span>
      <span class="policy-year-tag">India · ${data.year}</span>
    </div>
    <div class="policy-cards">`;

  insights.forEach((ins, i) => {
    const area    = ins.policy_area || ins.theme || "Policy Area";
    const why     = ins.rationale   || ins.why_it_matters || "";
    const actions = ins.actions     || ins.policy_focus   || [];

    html += `
      <div class="pol-card">
        <div class="pol-card-head">
          <div class="pol-num">${i + 1}</div>
          <h3 class="pol-area">${area}</h3>
        </div>
        <p class="pol-why">${why}</p>
        ${actions.length ? `
          <div class="pol-actions">
            ${actions.slice(0, 3).map(a =>
              `<div class="pol-action">
                 <span class="pol-arrow">→</span>
                 <span>${a}</span>
               </div>`
            ).join("")}
          </div>` : ""}
      </div>`;
  });

  html += `</div>`;
  document.getElementById("policyContent").innerHTML = html;
  document.getElementById("tabPolicy").style.opacity = "1";
}

// ── Load policy on demand ─────────────────────────────────────
async function loadPolicy() {
  if (!lastYear) return;

  document.getElementById("policyContent").innerHTML = `
    <div class="policy-empty">
      <div class="pe-icon" style="animation:spin 1s linear infinite;display:inline-block">⟳</div>
      <p class="pe-title">Generating insights…</p>
      <p class="pe-body">Llama 3.2 is analysing the SHAP drivers for ${lastYear}.</p>
    </div>`;

  try {
    const res = await fetch(`${API}/predict/explain-policy`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ year: lastYear }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    renderPolicy(data);
  } catch (err) {
    document.getElementById("policyContent").innerHTML = `
      <div class="policy-empty">
        <div class="pe-icon">⚠</div>
        <p class="pe-title">Failed to load</p>
        <p class="pe-body">${err.message}</p>
        <button class="btn-outline" onclick="loadPolicy()">Retry</button>
      </div>`;
  }
}

// ═══════════════════════════════════════════════════════════
// TAB SWITCHING
// ═══════════════════════════════════════════════════════════
function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t =>
    t.classList.toggle("active", t.dataset.tab === name)
  );
  document.querySelectorAll(".tab-panel").forEach(p =>
    p.classList.toggle("active", p.id === `tp-${name}`)
  );
}

// ═══════════════════════════════════════════════════════════
// TOAST
// ═══════════════════════════════════════════════════════════
let _toastTimer = null;
function showToast(msg) {
  const el = document.getElementById("toast");
  document.getElementById("toastMsg").textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(closeToast, 5000);
}
function closeToast() {
  document.getElementById("toast").classList.add("hidden");
}

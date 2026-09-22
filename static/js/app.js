/**
 * Axiom Credit Risk Analyzer
 * Streamlined Client-Side Controller
 */

const PRESETS = {
  prime: {
    person_age: 32,
    person_emp_length: 6.0,
    person_home_ownership: 'MORTGAGE',
    person_income: 85000,
    loan_amnt: 10000,
    loan_intent: 'EDUCATION',
    loan_grade: 'A',
    loan_int_rate: 7.20,
    cb_person_cred_hist_length: 8,
    cb_person_default_on_file: 'N'
  },
  subprime: {
    person_age: 22,
    person_emp_length: 1.0,
    person_home_ownership: 'RENT',
    person_income: 28000,
    loan_amnt: 22000,
    loan_intent: 'DEBTCONSOLIDATION',
    loan_grade: 'F',
    loan_int_rate: 18.50,
    cb_person_cred_hist_length: 2,
    cb_person_default_on_file: 'Y'
  }
};

let historyRecords = [];
const appStartTime = performance.now();

document.addEventListener('DOMContentLoaded', async () => {
  initIcons();
  setupNavigation();
  setupPresets();
  setupForm();
  updateLiveDti();
  await fetchHistory();

  // Smooth dismissal of Axiom splash loading animation
  const elapsed = performance.now() - appStartTime;
  const remaining = Math.max(0, 750 - elapsed);
  setTimeout(dismissPreloader, remaining);
});

function dismissPreloader() {
  const preloader = document.getElementById('preloader');
  if (preloader) {
    preloader.classList.add('loaded');
    setTimeout(() => {
      if (preloader.parentNode) preloader.parentNode.removeChild(preloader);
    }, 600);
  }
}

function initIcons() {
  if (window.lucide && typeof window.lucide.createIcons === 'function') {
    window.lucide.createIcons();
  }
}

// Navigation between Assessment and History
function setupNavigation() {
  const links = document.querySelectorAll('.nav-link');
  links.forEach(link => {
    link.addEventListener('click', () => {
      const view = link.dataset.view;
      switchView(view);
    });
  });
}

function switchView(viewName) {
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.toggle('active', link.dataset.view === viewName);
  });

  document.querySelectorAll('.view-panel').forEach(panel => {
    panel.classList.toggle('active', panel.id === `view-${viewName}`);
  });

  const titleEl = document.getElementById('pageTitle');
  const descEl = document.getElementById('pageDesc');
  const demoBar = document.getElementById('demoPresetsBar');

  if (viewName === 'assessment') {
    titleEl.textContent = 'Credit Risk Assessment';
    descEl.textContent = 'Evaluate loan default probability and underwriting decision.';
    demoBar.style.display = 'flex';
  } else {
    titleEl.textContent = 'Assessment History';
    descEl.textContent = 'Ledger of all evaluated credit applications.';
    demoBar.style.display = 'none';
  }

  initIcons();
}

// Preset Handlers
function setupPresets() {
  document.querySelectorAll('.btn-chip[data-preset]').forEach(btn => {
    btn.addEventListener('click', () => {
      const p = PRESETS[btn.dataset.preset];
      if (p) applyData(p);
    });
  });

  const clearBtn = document.getElementById('btnClearForm');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      document.getElementById('loanForm').reset();
      applyData({
        person_age: 28,
        person_emp_length: 4.0,
        person_home_ownership: 'RENT',
        person_income: 65000,
        loan_amnt: 10000,
        loan_intent: 'PERSONAL',
        loan_grade: 'B',
        loan_int_rate: 10.5,
        cb_person_cred_hist_length: 5,
        cb_person_default_on_file: 'N'
      });
      document.querySelectorAll('.field-error').forEach(e => e.classList.remove('active'));
    });
  }
}

function applyData(data) {
  for (const [key, val] of Object.entries(data)) {
    const el = document.getElementById(key);
    if (el) el.value = val;
  }

  const defaultVal = data.cb_person_default_on_file || 'N';
  document.querySelectorAll('.binary-switch .switch-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.val === defaultVal);
  });
  document.getElementById('cb_person_default_on_file').value = defaultVal;

  updateLiveDti();
}

// Form Listeners & DTI live calculation
function setupForm() {
  const form = document.getElementById('loanForm');
  const incomeInput = document.getElementById('person_income');
  const loanInput = document.getElementById('loan_amnt');

  incomeInput.addEventListener('input', updateLiveDti);
  loanInput.addEventListener('input', updateLiveDti);

  // Binary switch for prior default
  document.querySelectorAll('.binary-switch .switch-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.binary-switch .switch-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('cb_person_default_on_file').value = btn.dataset.val;
    });
  });

  // Submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!validateForm()) return;
    await submitEvaluation();
  });

  // Export CSV
  const btnExport = document.getElementById('btnExportCsv');
  if (btnExport) {
    btnExport.addEventListener('click', () => {
      window.location.href = '/api/export-history';
    });
  }
}

function updateLiveDti() {
  const income = parseFloat(document.getElementById('person_income').value) || 1;
  const loan = parseFloat(document.getElementById('loan_amnt').value) || 0;
  
  const dti = Math.min(Math.max(loan / Math.max(income, 1), 0), 1.0);
  const dtiPct = (dti * 100).toFixed(1);

  const textEl = document.getElementById('dtiLiveText');
  const hiddenInput = document.getElementById('loan_percent_income');

  if (textEl) {
    textEl.textContent = `${dtiPct}%`;
    if (dti > 0.35) {
      textEl.style.color = 'var(--risk-high)';
    } else {
      textEl.style.color = 'var(--text-main)';
    }
  }
  if (hiddenInput) hiddenInput.value = (dti).toFixed(4);
}

function validateForm() {
  let ok = true;
  document.querySelectorAll('.field-error').forEach(e => {
    e.textContent = '';
    e.classList.remove('active');
  });

  const age = parseInt(document.getElementById('person_age').value, 10);
  if (isNaN(age) || age < 18 || age > 100) {
    showError('person_age', 'Age must be 18–100.');
    ok = false;
  }

  const income = parseFloat(document.getElementById('person_income').value);
  if (isNaN(income) || income <= 0) {
    showError('person_income', 'Income must be > $0.');
    ok = false;
  }

  const loan = parseFloat(document.getElementById('loan_amnt').value);
  if (isNaN(loan) || loan <= 0) {
    showError('loan_amnt', 'Loan amount must be > $0.');
    ok = false;
  }

  const rate = parseFloat(document.getElementById('loan_int_rate').value);
  if (isNaN(rate) || rate < 0.5 || rate > 40) {
    showError('loan_int_rate', 'Rate must be between 0.5% and 40%.');
    ok = false;
  }

  return ok;
}

function showError(field, msg) {
  const el = document.getElementById(`err_${field}`);
  if (el) {
    el.textContent = msg;
    el.classList.add('active');
  }
}

// Submit Inference
async function submitEvaluation() {
  const btn = document.getElementById('btnSubmit');
  const originalHtml = btn.innerHTML;

  try {
    btn.disabled = true;
    btn.innerHTML = '<span>Evaluating Risk...</span>';

    const payload = {
      person_age: parseInt(document.getElementById('person_age').value, 10),
      person_income: parseFloat(document.getElementById('person_income').value),
      person_home_ownership: document.getElementById('person_home_ownership').value,
      person_emp_length: parseFloat(document.getElementById('person_emp_length').value),
      loan_intent: document.getElementById('loan_intent').value,
      loan_grade: document.getElementById('loan_grade').value,
      loan_amnt: parseFloat(document.getElementById('loan_amnt').value),
      loan_int_rate: parseFloat(document.getElementById('loan_int_rate').value),
      loan_percent_income: parseFloat(document.getElementById('loan_percent_income').value),
      cb_person_default_on_file: document.getElementById('cb_person_default_on_file').value,
      cb_person_cred_hist_length: parseInt(document.getElementById('cb_person_cred_hist_length').value, 10)
    };

    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Evaluation failed.');
    }

    const data = await resp.json();
    renderResults(data);
    await fetchHistory();

  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalHtml;
    initIcons();
  }
}

// Display Result Card
function renderResults(data) {
  document.getElementById('placeholderState').classList.add('hidden');
  document.getElementById('resultsState').classList.remove('hidden');

  // Decision Banner
  const banner = document.getElementById('decisionBanner');
  const actionEl = document.getElementById('decisionAction');
  const tierPill = document.getElementById('decisionTierPill');
  const explEl = document.getElementById('decisionExplanation');

  banner.className = 'decision-banner';
  if (data.risk_tier === 'Low Risk') {
    banner.classList.add('tier-low');
    actionEl.textContent = 'APPROVED';
    tierPill.textContent = 'Low Risk';
  } else if (data.risk_tier === 'Moderate Risk') {
    banner.classList.add('tier-med');
    actionEl.textContent = 'MANUAL REVIEW';
    tierPill.textContent = 'Moderate Risk';
  } else {
    banner.classList.add('tier-high');
    actionEl.textContent = 'DECLINED';
    tierPill.textContent = 'High Risk';
  }
  explEl.textContent = data.decision_notes;

  // Stats
  document.getElementById('statPd').textContent = `${data.probability_percent.toFixed(1)}%`;
  document.getElementById('statScore').textContent = data.credit_score;
  document.getElementById('statDti').textContent = `${((data.inputs.loan_percent_income || 0) * 100).toFixed(1)}%`;

  // Update Dial Gauge Needle
  updateDialNeedle(data.default_probability);

  // Top Factors (Horizontal Bar Chart)
  renderFactors(data.key_drivers);

  initIcons();
}

function updateDialNeedle(prob) {
  const p = Math.min(Math.max(prob, 0.0), 1.0);
  // Center is (110, 115), Radius is 80
  // 0% -> 180 deg (left), 100% -> 0 deg (right)
  const angle = Math.PI - (p * Math.PI);
  const nx = 110 + 80 * Math.cos(angle);
  const ny = 115 - 80 * Math.sin(angle);

  const needle = document.getElementById('dialNeedle');
  if (needle) {
    needle.setAttribute('x2', nx.toFixed(1));
    needle.setAttribute('y2', ny.toFixed(1));
  }
}

function renderFactors(factors) {
  const list = document.getElementById('factorsList');
  if (!list) return;
  list.innerHTML = '';

  if (!factors || factors.length === 0) {
    list.innerHTML = '<span class="text-muted">No factors available.</span>';
    return;
  }

  const top = factors.slice(0, 5);
  const maxAbs = Math.max(...top.map(f => Math.abs(f.impact)), 0.1);

  top.forEach(f => {
    const isIncrease = f.impact > 0;
    const barWidth = Math.min((Math.abs(f.impact) / maxAbs) * 50, 48);

    const row = document.createElement('div');
    row.className = 'factor-item';

    const colorClass = isIncrease ? 'text-red' : 'text-green';
    const fillClass = isIncrease ? 'increases' : 'reduces';
    const sign = isIncrease ? '+' : '';

    row.innerHTML = `
      <span class="factor-title" title="${f.feature}">${f.feature}</span>
      <div class="factor-bar-track">
        <div class="factor-axis"></div>
        <div class="factor-fill ${fillClass}" style="width: ${barWidth.toFixed(1)}%;"></div>
      </div>
      <span class="factor-num mono ${colorClass}">${sign}${f.impact.toFixed(2)}</span>
    `;

    list.appendChild(row);
  });
}

// History Table
async function fetchHistory() {
  try {
    const resp = await fetch('/api/history');
    if (!resp.ok) return;
    const data = await resp.json();
    historyRecords = data.records || [];

    const badge = document.getElementById('navHistoryCount');
    if (badge) badge.textContent = historyRecords.length;

    renderHistoryTable();
  } catch (err) {
    console.error('History fetch failed:', err);
  }
}

function renderHistoryTable() {
  const tbody = document.getElementById('historyTableBody');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (historyRecords.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding: 20px; color: var(--text-muted);">No assessments recorded yet.</td></tr>`;
    return;
  }

  historyRecords.forEach(r => {
    const tr = document.createElement('tr');
    
    let badgeClass = 'badge-tier-low';
    if (r.risk_tier === 'High Risk') badgeClass = 'badge-tier-high';
    else if (r.risk_tier === 'Moderate Risk') badgeClass = 'badge-tier-med';

    const incomeFmt = Number(r.person_income || 0).toLocaleString();
    const loanFmt = Number(r.loan_amnt || 0).toLocaleString();
    const pdPct = ((r.default_probability || 0) * 100).toFixed(1);

    tr.innerHTML = `
      <td class="mono"><strong>${r.assessment_id || 'APP'}</strong></td>
      <td class="mono" style="font-size: 11px; color: var(--text-muted);">${r.timestamp || ''}</td>
      <td class="mono">$${incomeFmt}</td>
      <td class="mono">$${loanFmt}</td>
      <td>${r.loan_intent || ''}</td>
      <td class="mono font-semibold">${pdPct}%</td>
      <td><span class="${badgeClass}">${r.risk_tier}</span></td>
      <td><strong>${r.recommendation}</strong></td>
      <td>
        <button type="button" class="btn-load" onclick="loadFromHistory('${r.assessment_id}')">Load</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

window.loadFromHistory = function(id) {
  const rec = historyRecords.find(r => r.assessment_id === id);
  if (!rec) return;
  applyData(rec);
  switchView('assessment');
};

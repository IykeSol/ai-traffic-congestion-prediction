/**
 * AI Traffic Congestion Predictor
 * Frontend Logic — Chart.js dashboards, form handling, NLP requests
 */

'use strict';

/* ────────────────────────────────────────────────────────────
   Constants & State
   ──────────────────────────────────────────────────────────── */
const API = {
  predict:    '/api/predict',
  nlpPredict: '/api/nlp-predict',
  stats:      '/api/model-stats',
  history:    '/api/history',
  health:     '/api/health',
};

const RECOMMENDATIONS = {
  low:      { status: 'No significant delays expected. Traffic is free-flowing.',
              tip:    'Optimal travel time. Consider this window for regular commutes.' },
  moderate: { status: 'Minor slowdowns expected. Allow 10–15 additional minutes.',
              tip:    'Monitor live traffic updates. Alternative routes may be beneficial.' },
  high:     { status: 'Significant congestion detected. Major delays are likely.',
              tip:    'Strongly consider public transport or departing outside peak hours.' },
  severe:   { status: 'Critical gridlock conditions. Severe, unpredictable delays.',
              tip:    'Postpone travel if at all possible. Emergency routes may be active.' },
};

let featureChart = null;
let perClassChart = null;

/* ────────────────────────────────────────────────────────────
   Hero Canvas — Particle Network
   ──────────────────────────────────────────────────────────── */
function initHeroCanvas() {
  const canvas = document.getElementById('heroCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, particles;

  function resize() {
    W = canvas.width  = canvas.offsetWidth;
    H = canvas.height = canvas.offsetHeight;
  }

  function makeParticles(n = 60) {
    return Array.from({ length: n }, () => ({
      x:  Math.random() * W,
      y:  Math.random() * H,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r:  Math.random() * 1.5 + 0.5,
    }));
  }

  resize();
  particles = makeParticles();
  window.addEventListener('resize', () => { resize(); particles = makeParticles(); });

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(99,179,237,${0.18 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.6;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw dots
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(99,179,237,0.6)';
      ctx.fill();

      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
    });

    requestAnimationFrame(draw);
  }
  draw();
}

/* ────────────────────────────────────────────────────────────
   Navbar scroll effect
   ──────────────────────────────────────────────────────────── */
function initNavbar() {
  const nav = document.getElementById('navbar');
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 20);
  });
}

/* ────────────────────────────────────────────────────────────
   Health check + status dot
   ──────────────────────────────────────────────────────────── */
async function checkHealth() {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  try {
    const res  = await fetch(API.health);
    const data = await res.json();
    if (data.status === 'ok') {
      dot.className  = 'status-dot online';
      text.textContent = data.model_loaded ? 'Model Online' : 'API Online (No Model)';
    } else {
      throw new Error();
    }
  } catch {
    dot.className  = 'status-dot offline';
    text.textContent = 'Offline';
  }
}

/* ────────────────────────────────────────────────────────────
   Load model stats → Dashboard
   ──────────────────────────────────────────────────────────── */
async function loadModelStats() {
  try {
    const res  = await fetch(API.stats);
    const data = await res.json();
    if (!data.success) return;

    const m = data.metrics;

    // Hero stats
    const acc = m.accuracy * 100;
    const f1  = m.f1_macro * 100;
    animateCounter('stat-accuracy', acc, '%', 1);
    animateCounter('stat-f1',       f1,  '%', 1);

    // Dashboard cards
    setMetric('dash-accuracy',   acc, '%', 1, 'bar-accuracy');
    setMetric('dash-f1macro',    f1,  '%', 1, 'bar-f1macro');
    setMetric('dash-f1weighted', m.f1_weighted * 100, '%', 1, 'bar-f1weighted');
    document.getElementById('dash-status').textContent = data.model_loaded ? 'Online' : 'No Model';
    document.getElementById('bar-status').style.width = data.model_loaded ? '100%' : '20%';

    // Charts
    if (m.feature_importances && Object.keys(m.feature_importances).length) {
      buildFeatureChart(m.feature_importances);
    }
    if (m.per_class) {
      buildPerClassChart(m.per_class);
    }

  } catch (e) {
    console.warn('Model stats unavailable:', e.message);
  }
}

function setMetric(valueId, value, suffix, decimals, barId) {
  const el  = document.getElementById(valueId);
  const bar = document.getElementById(barId);
  if (el)  animateCounter(valueId, value, suffix, decimals);
  if (bar) setTimeout(() => { bar.style.width = Math.min(value, 100) + '%'; }, 200);
}

function animateCounter(id, target, suffix = '', decimals = 0) {
  const el = document.getElementById(id);
  if (!el) return;
  const start = 0;
  const dur   = 900;
  const t0    = performance.now();
  function step(now) {
    const progress = Math.min((now - t0) / dur, 1);
    const ease     = 1 - Math.pow(1 - progress, 3);
    const current  = start + (target - start) * ease;
    el.textContent = current.toFixed(decimals) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ────────────────────────────────────────────────────────────
   Chart.js — Feature Importance
   ──────────────────────────────────────────────────────────── */
function buildFeatureChart(importances) {
  const ctx = document.getElementById('featureImportanceChart');
  if (!ctx) return;

  const sorted = Object.entries(importances)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);

  const labels = sorted.map(([k]) => k.replace(/_/g, ' '));
  const values = sorted.map(([, v]) => (v * 100).toFixed(2));

  if (featureChart) featureChart.destroy();

  featureChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Importance (%)',
        data: values,
        backgroundColor: labels.map((_, i) =>
          `rgba(59,130,246,${0.9 - i * 0.06})`),
        borderColor: 'rgba(59,130,246,0.5)',
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.parsed.x}%` },
        },
      },
      scales: {
        x: {
          grid:  { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#8892a4', font: { size: 11 } },
        },
        y: {
          grid:  { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#8892a4', font: { size: 11 } },
        },
      },
    },
  });
}

/* ────────────────────────────────────────────────────────────
   Chart.js — Per-class F1
   ──────────────────────────────────────────────────────────── */
function buildPerClassChart(perClass) {
  const ctx = document.getElementById('perClassChart');
  if (!ctx) return;

  const classNames = ['low', 'moderate', 'high', 'severe'];
  const colors     = ['#22c55e', '#f59e0b', '#ef4444', '#b91c1c'];
  const f1s        = classNames.map(c => perClass[c]?.['f1-score'] ? (perClass[c]['f1-score'] * 100).toFixed(1) : 0);

  if (perClassChart) perClassChart.destroy();

  perClassChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: classNames.map(c => c.charAt(0).toUpperCase() + c.slice(1)),
      datasets: [{
        data: f1s,
        backgroundColor: colors.map(c => c + 'cc'),
        borderColor:     colors,
        borderWidth: 2,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true,
      cutout: '68%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: '#8892a4',
            font:  { size: 12 },
            padding: 16,
          },
        },
        tooltip: {
          callbacks: { label: ctx => ` F1: ${ctx.parsed}%` },
        },
      },
    },
  });
}

/* ────────────────────────────────────────────────────────────
   Structured Predictor Form
   ──────────────────────────────────────────────────────────── */
function initPredictorForm() {
  const form    = document.getElementById('predictorForm');
  const btn     = document.getElementById('predictBtn');
  const spinner = document.getElementById('predictSpinner');
  const btnText = btn.querySelector('.btn-text');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Collect form data
    const fd = new FormData(form);
    const raw = {};
    fd.forEach((v, k) => { raw[k] = v; });

    // Checkbox handling (unchecked = 0)
    ['incident_reported','school_zone','construction_zone','public_event_nearby'].forEach(k => {
      raw[k] = form.querySelector(`#${k}`).checked ? 1 : 0;
    });

    // Set loading state
    btnText.textContent = 'Analysing...';
    spinner.classList.remove('hidden');
    btn.disabled = true;

    try {
      const res  = await fetch(API.predict, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(raw),
      });
      const data = await res.json();

      if (!data.success) throw new Error(data.error || 'Prediction failed');

      showStructuredResult(data);
      loadHistory();

    } catch (err) {
      showError('resultPanel', err.message);
    } finally {
      btnText.textContent = 'Predict Congestion Level';
      spinner.classList.add('hidden');
      btn.disabled = false;
    }
  });
}

function showStructuredResult(data) {
  const placeholder = document.getElementById('resultPlaceholder');
  const content     = document.getElementById('resultContent');
  const badge       = document.getElementById('resultBadge');
  const indicator   = document.getElementById('levelIndicator');
  const levelText   = document.getElementById('levelText');
  const confValue   = document.getElementById('confValue');
  const barsEl      = document.getElementById('confidenceBars');
  const recEl       = document.getElementById('resultRecommendation');
  const metaEl      = document.getElementById('resultMeta');
  const panel       = document.getElementById('resultPanel');

  placeholder.classList.add('hidden');
  content.classList.remove('hidden');
  panel.style.alignItems = 'flex-start';
  panel.style.justifyContent = 'flex-start';

  const level = data.prediction;

  indicator.className = `level-indicator dot-result-${level}`;
  levelText.className = `level-text level-${level}`;
  levelText.textContent = level;

  confValue.textContent = data.top_confidence + '%';

  // Confidence bars
  barsEl.innerHTML = '';
  const order = ['low', 'moderate', 'high', 'severe'];
  order.forEach(cls => {
    const pct = ((data.confidence[cls] || 0) * 100);
    const row = document.createElement('div');
    row.className = 'conf-bar-row';
    row.innerHTML = `
      <span class="conf-bar-label">${cls}</span>
      <div class="conf-bar-track">
        <div class="conf-bar-fill fill-${cls}" style="width:0%"></div>
      </div>
      <span class="conf-bar-pct">${pct.toFixed(1)}%</span>
    `;
    barsEl.appendChild(row);
    setTimeout(() => {
      row.querySelector('.conf-bar-fill').style.width = pct + '%';
    }, 80);
  });

  // Recommendation
  const rec = RECOMMENDATIONS[level] || {};
  let recHtml = `<strong>Status</strong>${rec.status}<br/><br/><strong>Recommendation</strong>${rec.tip}`;
  if (data.ai_summary) {
      recHtml += `<br/><br/><strong style="color:var(--blue-400);">AI Analysis</strong><span style="display:block;margin-top:4px;">${data.ai_summary}</span>`;
  }
  recEl.innerHTML = recHtml;

  // Meta
  const now = new Date().toLocaleTimeString();
  metaEl.textContent = `Predicted at ${now}  |  Ensemble: XGBoost + RandomForest`;

  // Animate panel border
  panel.style.borderColor = getBorderColor(level);
}

function getBorderColor(level) {
  return { low: '#22c55e44', moderate: '#f59e0b44', high: '#ef444444', severe: '#b91c1c55' }[level] || 'rgba(255,255,255,0.07)';
}

/* ────────────────────────────────────────────────────────────
   NLP Predictor
   ──────────────────────────────────────────────────────────── */
function initNLP() {
  const btn     = document.getElementById('nlpSubmit');
  const spinner = document.getElementById('nlpSpinner');
  const btnText = btn.querySelector('.btn-text');
  const input   = document.getElementById('nlpInput');

  // Example chips
  document.querySelectorAll('.example-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      input.value = chip.dataset.text;
      input.focus();
    });
  });

  btn.addEventListener('click', async () => {
    const description = input.value.trim();
    if (!description) { input.focus(); return; }

    btnText.textContent = 'Analysing with AI...';
    spinner.classList.remove('hidden');
    btn.disabled = true;

    try {
      const res  = await fetch(API.nlpPredict, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ description }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'NLP prediction failed');

      showNLPResult(data);
      loadHistory();

    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      btnText.textContent = 'Analyse with AI';
      spinner.classList.add('hidden');
      btn.disabled = false;
    }
  });
}

function showNLPResult(data) {
  const resultEl = document.getElementById('nlpResult');
  const badge    = document.getElementById('nlpBadge');
  const confEl   = document.getElementById('nlpConfidence');
  const chips    = document.getElementById('extractedChips');
  const summary  = document.getElementById('nlpSummaryText');
  const barsEl   = document.getElementById('nlpConfBars');

  const level = data.prediction;
  const colors = { low: '#22c55e', moderate: '#f59e0b', high: '#ef4444', severe: '#b91c1c' };
  const c = colors[level] || '#3b82f6';

  badge.textContent = level.toUpperCase();
  badge.style.cssText = `background:${c}22; color:${c}; border:1px solid ${c}44`;
  confEl.textContent  = `Confidence: ${data.top_confidence}%`;

  // Extracted features chips
  chips.innerHTML = '';
  const skip = ['hour_sin','hour_cos','day_sin','day_cos','month_sin','month_cos','is_weekend'];
  Object.entries(data.extracted || {}).forEach(([k, v]) => {
    if (skip.includes(k)) return;
    const chip = document.createElement('div');
    chip.className = 'extracted-chip';
    chip.innerHTML = `${k.replace(/_/g,' ')}: <span>${v}</span>`;
    chips.appendChild(chip);
  });

  summary.textContent = data.summary || 'AI summary not available.';

  // Confidence bars
  barsEl.innerHTML = '';
  const order = ['low', 'moderate', 'high', 'severe'];
  order.forEach(cls => {
    const pct = ((data.confidence[cls] || 0) * 100);
    const row = document.createElement('div');
    row.className = 'conf-bar-row';
    row.innerHTML = `
      <span class="conf-bar-label">${cls}</span>
      <div class="conf-bar-track">
        <div class="conf-bar-fill fill-${cls}" style="width:0%"></div>
      </div>
      <span class="conf-bar-pct">${pct.toFixed(1)}%</span>
    `;
    barsEl.appendChild(row);
    setTimeout(() => { row.querySelector('.conf-bar-fill').style.width = pct + '%'; }, 80);
  });

  resultEl.classList.remove('hidden');
  resultEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ────────────────────────────────────────────────────────────
   History
   ──────────────────────────────────────────────────────────── */
async function loadHistory() {
  try {
    const res  = await fetch(API.history + '?limit=20');
    const data = await res.json();
    renderHistory(data.history || [], data.total || 0);
  } catch (e) {
    console.warn('History load failed:', e.message);
  }
}

function renderHistory(records, total) {
  const tbody = document.getElementById('historyBody');
  const count = document.getElementById('historyCount');

  count.textContent = `${total} prediction${total !== 1 ? 's' : ''}`;

  if (!records.length) {
    tbody.innerHTML = `<tr class="history-empty"><td colspan="7">No predictions yet. Run the predictor above to get started.</td></tr>`;
    return;
  }

  tbody.innerHTML = records.map(r => {
    const level   = r.prediction;
    const source  = r.source === 'nlp' ? '<span class="badge-congestion badge-nlp">NLP</span>' : '<span class="badge-congestion badge-structured">Form</span>';
    const badge   = `<span class="badge-congestion badge-${level}">${level}</span>`;
    const roadType = r.input?.road_type || '--';
    const weather  = r.input?.weather_condition || '--';
    const hour     = r.input?.hour !== undefined ? `${r.input.hour}:00` : '--';
    const conf     = r.confidence ? `${r.confidence}%` : '--';
    return `
      <tr>
        <td>${r.timestamp || '--'}</td>
        <td>${source}</td>
        <td>${roadType}</td>
        <td>${weather}</td>
        <td>${hour}</td>
        <td>${badge}</td>
        <td>${conf}</td>
      </tr>
    `;
  }).join('');
}

/* ────────────────────────────────────────────────────────────
   Utilities
   ──────────────────────────────────────────────────────────── */
function showError(panelId, message) {
  const panel = document.getElementById(panelId);
  if (!panel) return;
  const errDiv = document.createElement('div');
  errDiv.style.cssText = 'color:#f87171;padding:12px;font-size:0.88rem;';
  errDiv.textContent = 'Error: ' + message;
  panel.appendChild(errDiv);
  setTimeout(() => errDiv.remove(), 6000);
}

function initRefreshHistory() {
  document.getElementById('refreshHistory').addEventListener('click', loadHistory);
}

/* ────────────────────────────────────────────────────────────
   Chart.js global defaults
   ──────────────────────────────────────────────────────────── */
function setChartDefaults() {
  Chart.defaults.color           = '#8892a4';
  Chart.defaults.borderColor     = 'rgba(255,255,255,0.05)';
  Chart.defaults.font.family     = "'Inter', sans-serif";
  Chart.defaults.plugins.tooltip.backgroundColor = '#161d2e';
  Chart.defaults.plugins.tooltip.borderColor      = 'rgba(255,255,255,0.1)';
  Chart.defaults.plugins.tooltip.borderWidth      = 1;
  Chart.defaults.plugins.tooltip.padding          = 10;
}

/* ────────────────────────────────────────────────────────────
   Boot
   ──────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  setChartDefaults();
  initHeroCanvas();
  initNavbar();
  checkHealth();
  loadModelStats();
  initPredictorForm();
  initNLP();
  initRefreshHistory();
  loadHistory();

  // Auto-refresh health every 30s
  setInterval(checkHealth, 30_000);
});

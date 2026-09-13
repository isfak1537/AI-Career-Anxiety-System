/**
 * app.js
 * Application Controller for Vercel Web Deployment
 * Manages UI reactivity, dynamic SVG charts, and seamless tab transitions.
 */

(function () {
  'use strict';

  // DOM Elements
  const tabs = document.querySelectorAll('.nav-tab');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const themeToggleIcon = document.getElementById('themeToggleIcon');

  // Form Elements
  const predictionForm = document.getElementById('predictionForm');
  const universitySelect = document.getElementById('universitySelect');
  const genderSelect = document.getElementById('genderSelect');
  const academicYearSelect = document.getElementById('academicYearSelect');
  const ageInput = document.getElementById('ageInput');
  const departmentSelect = document.getElementById('departmentSelect');
  const otherDeptWrapper = document.getElementById('otherDeptWrapper');
  const otherDepartmentInput = document.getElementById('otherDepartmentInput');
  const careerSelect = document.getElementById('careerSelect');
  const otherCareerWrapper = document.getElementById('otherCareerWrapper');
  const otherCareerInput = document.getElementById('otherCareerInput');
  const aiKnowledgeSelect = document.getElementById('aiKnowledgeSelect');
  const aiReplaceJobsSelect = document.getElementById('aiReplaceJobsSelect');
  const aiTakeoverTimeSelect = document.getElementById('aiTakeoverTimeSelect');
  const aiFuturePerspectiveSelect = document.getElementById('aiFuturePerspectiveSelect');
  const toolChipsContainer = document.getElementById('toolChipsContainer');
  const additionalToolsInput = document.getElementById('additionalToolsInput');
  const aiToolPerceptionInput = document.getElementById('aiToolPerceptionInput');
  const cohortOverrideSelect = document.getElementById('cohortOverrideSelect');
  const resolvedCohortBadge = document.getElementById('resolvedCohortBadge');

  // Results Elements
  const predictionOutcomeBanner = document.getElementById('predictionOutcomeBanner');
  const outcomeIcon = document.getElementById('outcomeIcon');
  const outcomeTitle = document.getElementById('outcomeTitle');
  const outcomeSubtitle = document.getElementById('outcomeSubtitle');
  const gaugeCircle = document.getElementById('gaugeCircle');
  const gaugePercentage = document.getElementById('gaugePercentage');
  const resModelName = document.getElementById('resModelName');
  const resCohortName = document.getElementById('resCohortName');
  const resFeatureDim = document.getElementById('resFeatureDim');
  const quickRiskFactors = document.getElementById('quickRiskFactors');
  const btnGoToExplain = document.getElementById('btnGoToExplain');

  // Explainability Elements
  const shapChartSvg = document.getElementById('shapChartSvg');
  const featuresRankTableBody = document.getElementById('featuresRankTableBody');

  // Benchmark Elements
  const benchmarkGrid = document.getElementById('benchmarkGrid');
  const benchmarkChartSvg = document.getElementById('benchmarkChartSvg');

  // State
  let lastPredictionResult = null;

  // Preset Configurations for Capstone Defense Presentation
  const PRESETS = {
    cse_high: {
      university: 'Daffodil International University',
      gender: 'Male',
      academic_year: '4th Year',
      age: 23,
      department: 'Department of Computer Science and Engineering',
      career_path: 'Software Engineer',
      ai_knowledge: 'High',
      ai_replace_jobs: 'Fully',
      ai_takeover_time: '1–5 years',
      ai_future_perspective: 'I strongly believe AI will completely take over human jobs and pose serious threat to humanity.',
      tools: ['ChatGPT', 'Copilot', 'Cursor', 'Deepseek'],
      extra_tools: 'Perplexity, Claude',
      tool_threat: 'Copilot',
    },
    pharmacy_mod: {
      university: 'American International University-Bangladesh',
      gender: 'Female',
      academic_year: '3rd Year',
      age: 22,
      department: 'Department of Pharmacy',
      career_path: 'Pharmacist',
      ai_knowledge: 'Medium',
      ai_replace_jobs: 'Partially',
      ai_takeover_time: '11–20 years',
      ai_future_perspective: 'I think AI will replace some human jobs but also create new opportunities.',
      tools: ['ChatGPT', 'Canva'],
      extra_tools: '',
      tool_threat: 'None',
    },
    english_low: {
      university: 'University of Dhaka',
      gender: 'Female',
      academic_year: '2nd Year',
      age: 20,
      department: 'Department of English',
      career_path: 'Teacher',
      ai_knowledge: 'Low',
      ai_replace_jobs: 'No',
      ai_takeover_time: 'Never',
      ai_future_perspective: 'I strongly believe AI cannot take over human activities and will be a helpful tool for humans.',
      tools: ['Quillbot', 'Grammarly'],
      extra_tools: '',
      tool_threat: 'None',
    }
  };

  /**
   * Theme Initialization
   */
  function initTheme() {
    const savedTheme = localStorage.getItem('ai_career_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    themeToggleIcon.textContent = savedTheme === 'dark' ? '☀️' : '🌙';

    themeToggleBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('ai_career_theme', next);
      themeToggleIcon.textContent = next === 'dark' ? '☀️' : '🌙';
      renderShapChart();
      renderBenchmarkChart();
    });
  }

  /**
   * Tab Navigation Setup
   */
  function initTabs() {
    function switchTab(targetId) {
      tabs.forEach(tab => {
        const isTarget = tab.dataset.tab === targetId;
        tab.classList.toggle('active', isTarget);
        tab.setAttribute('aria-selected', isTarget ? 'true' : 'false');
      });

      tabPanes.forEach(pane => {
        pane.classList.toggle('active', pane.id === targetId);
      });

      window.scrollTo({ top: 0, behavior: 'smooth' });

      // Refresh charts on tab activation
      if (targetId === 'tabExplainability') {
        renderShapChart();
      } else if (targetId === 'tabBenchmark') {
        renderBenchmarkChart();
      }
    }

    tabs.forEach(tab => {
      tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    // Footer tab links
    document.querySelectorAll('.footer-tab-link').forEach(link => {
      link.addEventListener('click', e => {
        e.preventDefault();
        switchTab(link.dataset.tab);
      });
    });

    if (btnGoToExplain) {
      btnGoToExplain.addEventListener('click', () => switchTab('tabExplainability'));
    }
  }

  /**
   * Dynamic Custom Department / Career Path Toggle
   */
  function initSelectToggles() {
    departmentSelect.addEventListener('change', () => {
      otherDeptWrapper.style.display = departmentSelect.value === 'Other Department' ? 'block' : 'none';
    });

    careerSelect.addEventListener('change', () => {
      otherCareerWrapper.style.display = careerSelect.value === 'Other Career Path' ? 'block' : 'none';
    });

    universitySelect.addEventListener('change', updateCohortCallout);
    cohortOverrideSelect.addEventListener('change', updateCohortCallout);
  }

  function updateCohortCallout() {
    const uni = universitySelect.value;
    const autoCohort = window.CareerAnxietyInference.resolveCohortFromUniversity(uni);
    const override = cohortOverrideSelect.value;

    const active = override !== 'auto' ? override : autoCohort;
    const modelLabels = {
      overall: 'Overall Cohort (Gradient Boosting)',
      public: 'Public Univ Cohort (Gradient Boosting)',
      private: 'Private Univ Cohort (Gradient Boosting)',
      daffodil: 'Daffodil Sub-Cohort (Soft Voting Ensemble)',
    };

    resolvedCohortBadge.textContent = modelLabels[active] || active.toUpperCase();

    // Update header active pill
    ['overall', 'public', 'private', 'daffodil'].forEach(c => {
      const chip = document.getElementById('chip' + c.charAt(0).toUpperCase() + c.slice(1));
      if (chip) chip.classList.toggle('active', c === active);
    });
  }

  /**
   * Tool Chips Multi-Select Setup
   */
  function initToolChips() {
    toolChipsContainer.querySelectorAll('.tool-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        chip.classList.toggle('selected');
      });
    });
  }

  function getSelectedToolsString() {
    const selected = [];
    toolChipsContainer.querySelectorAll('.tool-chip.selected').forEach(chip => {
      selected.push(chip.dataset.tool);
    });
    const extra = additionalToolsInput.value.trim();
    if (extra) {
      extra.split(',').map(t => t.trim()).filter(Boolean).forEach(t => selected.push(t));
    }
    return selected.join(', ');
  }

  /**
   * Preset Profile Handler
   */
  function initPresets() {
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const presetKey = btn.dataset.preset;
        const p = PRESETS[presetKey];
        if (!p) return;

        universitySelect.value = p.university;
        genderSelect.value = p.gender;
        academicYearSelect.value = p.academic_year;
        ageInput.value = p.age;
        departmentSelect.value = p.department;
        otherDeptWrapper.style.display = 'none';
        careerSelect.value = p.career_path;
        otherCareerWrapper.style.display = 'none';
        aiKnowledgeSelect.value = p.ai_knowledge;
        aiReplaceJobsSelect.value = p.ai_replace_jobs;
        aiTakeoverTimeSelect.value = p.ai_takeover_time;
        aiFuturePerspectiveSelect.value = p.ai_future_perspective;
        additionalToolsInput.value = p.extra_tools;
        aiToolPerceptionInput.value = p.tool_threat;

        // Tool chips
        toolChipsContainer.querySelectorAll('.tool-chip').forEach(chip => {
          const isSelected = p.tools.includes(chip.dataset.tool);
          chip.classList.toggle('selected', isSelected);
        });

        cohortOverrideSelect.value = 'auto';
        updateCohortCallout();
        executePrediction();
      });
    });
  }

  /**
   * Form Submission & Prediction Pipeline
   */
  function initPrediction() {
    predictionForm.addEventListener('submit', e => {
      e.preventDefault();
      executePrediction();
    });

    // Run initial default prediction on page load
    updateCohortCallout();
    executePrediction();
  }

  async function executePrediction() {
    const deptVal = departmentSelect.value === 'Other Department'
      ? (otherDepartmentInput.value.trim() || 'Department of General Studies')
      : departmentSelect.value;

    const careerVal = careerSelect.value === 'Other Career Path'
      ? (otherCareerInput.value.trim() || 'General Professional')
      : careerSelect.value;

    const rawInputs = {
      university: universitySelect.value,
      department: deptVal,
      age: parseInt(ageInput.value, 10) || 22,
      gender: genderSelect.value,
      academic_year: academicYearSelect.value,
      ai_knowledge: aiKnowledgeSelect.value,
      ai_tools_used: getSelectedToolsString(),
      career_path: careerVal,
      ai_tool_perception: aiToolPerceptionInput.value.trim() || 'None',
      ai_future_perspective: aiFuturePerspectiveSelect.value,
      ai_replace_jobs: aiReplaceJobsSelect.value,
      ai_takeover_time: aiTakeoverTimeSelect.value,
      cohort_override: cohortOverrideSelect.value || 'auto',
    };

    const override = cohortOverrideSelect.value;

    // 1. Try Live Python Vercel Function Endpoint
    try {
      let response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(rawInputs),
      });

      if (!response.ok && response.status === 404) {
        response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(rawInputs),
        });
      }

      if (response.ok) {
        const data = await response.json();
        const serverResult = {
          cohort: data.cohort,
          autoCohort: data.auto_cohort,
          isOverride: data.is_override,
          prediction: {
            predClass: data.prediction.predicted_class,
            probability: data.prediction.probability,
            modelName: data.model_name,
          },
          explainability: {
            featureRows: data.all_features_ranked.map(f => ({
              key: f.key,
              displayName: f.display_name,
              category: 'Feature Attribution',
              inputValue: f.input_value,
              contribution: f.contribution,
              absContribution: f.abs_contribution,
              direction: f.direction,
            })),
          },
          engineered: data.engineered_features,
          rawInputs,
          fromPythonBackend: true,
        };

        lastPredictionResult = serverResult;
        displayPredictionResults(serverResult);
        populateExplainabilityTable(serverResult);
        renderShapChart();
        return;
      }
    } catch (err) {
      // Backend not running / static mode — proceed to client inference engine
    }

    // 2. Client-Side Instant Inference Engine (Fallback / Static CDN Mode)
    const clientResult = window.CareerAnxietyInference.runPrediction(rawInputs, override);
    lastPredictionResult = clientResult;

    displayPredictionResults(clientResult);
    populateExplainabilityTable(clientResult);
    renderShapChart();
  }

  /**
   * Update UI with Prediction Results
   */
  function displayPredictionResults(result) {
    const { prediction, cohort } = result;
    const isHigh = prediction.predClass === 1;
    const pct = (prediction.probability * 100).toFixed(1);

    // Outcome Banner
    predictionOutcomeBanner.className = `prediction-outcome-banner ${isHigh ? 'outcome-banner-high' : 'outcome-banner-low'}`;
    outcomeIcon.textContent = isHigh ? '⚠️' : '✅';
    outcomeTitle.textContent = isHigh ? 'Elevated AI Career Anxiety' : 'Low / No AI Career Anxiety';
    outcomeSubtitle.textContent = isHigh
      ? `Predicted Class 1: Empirical features strongly align with elevated self-reported career concern.`
      : `Predicted Class 0: Features align with resilient, low-anxiety academic perception.`;

    // Radial SVG Gauge
    gaugePercentage.textContent = `${pct}%`;
    gaugePercentage.style.color = isHigh ? 'var(--danger)' : 'var(--success)';
    gaugeCircle.style.stroke = isHigh ? 'var(--danger)' : 'var(--success)';

    // Circumference = 2 * PI * 65 ≈ 408.4
    const circumference = 408.4;
    const offset = circumference - (prediction.probability * circumference);
    gaugeCircle.style.strokeDashoffset = offset;

    // Metadata
    resModelName.textContent = prediction.modelName.split('(')[0].trim();
    resCohortName.textContent = cohort.toUpperCase();

    const dims = { overall: 170, public: 117, private: 145, daffodil: 129 };
    resFeatureDim.textContent = `${dims[cohort] || 170} Dimensions`;

    // Quick Risk Factors Pills
    quickRiskFactors.innerHTML = '';
    const top3 = result.explainability.featureRows.slice(0, 3);
    top3.forEach(f => {
      const isPos = f.contribution >= 0;
      const pill = document.createElement('div');
      pill.className = `risk-pill ${isPos ? 'positive' : 'negative'}`;
      pill.innerHTML = `
        <span style="font-weight: 600;">${f.displayName}</span>
        <span class="pill-score ${isPos ? 'pos' : 'neg'}">${isPos ? '+' : ''}${f.contribution.toFixed(4)}</span>
      `;
      quickRiskFactors.appendChild(pill);
    });
  }

  /**
   * Populate Ranked Feature Table in Tab 2
   */
  function populateExplainabilityTable(result) {
    if (!featuresRankTableBody) return;
    featuresRankTableBody.innerHTML = '';

    const badgeClasses = {
      'Demographic': 'badge-demo',
      'AI Literacy & Beliefs': 'badge-literacy',
      'AI Tool Adoption': 'badge-tools',
      'Construct Interaction': 'badge-construct',
      'Academic & Career': 'badge-academic',
    };

    result.explainability.featureRows.forEach((row, idx) => {
      const tr = document.createElement('tr');
      const isPos = row.contribution >= 0;
      const badgeCls = badgeClasses[row.category] || 'badge-demo';

      tr.innerHTML = `
        <td><strong style="font-family: var(--font-mono); color: var(--text-muted);">#${idx + 1}</strong></td>
        <td><strong>${row.displayName}</strong></td>
        <td><span class="badge-tag ${badgeCls}">${row.category}</span></td>
        <td><code style="font-family: var(--font-mono);">${row.inputValue}</code></td>
        <td><span class="stat-val" style="color: ${isPos ? 'var(--danger)' : 'var(--info)'};">${isPos ? '+' : ''}${row.contribution.toFixed(4)}</span></td>
        <td>
          <span style="font-size: 0.8rem; font-weight: 600; color: ${isPos ? '#f87171' : '#7dd3fc'};">
            ${isPos ? '▲ Pushes Toward Class 1' : '▼ Mitigates Anxiety'}
          </span>
        </td>
      `;
      featuresRankTableBody.appendChild(tr);
    });
  }

  /**
   * Dynamic SVG SHAP Horizontal Bar Chart
   */
  function renderShapChart() {
    if (!shapChartSvg || !lastPredictionResult) return;

    const rows = [...lastPredictionResult.explainability.featureRows].reverse(); // reverse so #1 is top
    const width = 900;
    const height = Math.max(500, rows.length * 28 + 60);
    shapChartSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    shapChartSvg.innerHTML = '';

    const maxAbs = Math.max(...rows.map(r => r.absContribution), 0.05);
    const domainLimit = Math.ceil(maxAbs * 12) / 10; // round up to nice tick

    const margin = { top: 30, right: 90, bottom: 40, left: 280 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    const zeroX = margin.left + chartWidth / 2;

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const textColor = isLight ? '#334155' : '#cbd5e1';
    const gridColor = isLight ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255, 255, 255, 0.08)';

    // Background Grid & Baseline
    const svgNS = 'http://www.w3.org/2000/svg';

    // Zero Baseline
    const zeroLine = document.createElementNS(svgNS, 'line');
    zeroLine.setAttribute('x1', zeroX);
    zeroLine.setAttribute('y1', margin.top);
    zeroLine.setAttribute('x2', zeroX);
    zeroLine.setAttribute('y2', height - margin.bottom);
    zeroLine.setAttribute('stroke', isLight ? '#64748b' : '#94a3b8');
    zeroLine.setAttribute('stroke-width', '1.5');
    zeroLine.setAttribute('stroke-dasharray', '4 4');
    shapChartSvg.appendChild(zeroLine);

    // Bar rows
    const rowHeight = chartHeight / rows.length;

    rows.forEach((r, i) => {
      const y = margin.top + i * rowHeight + rowHeight * 0.15;
      const barH = rowHeight * 0.7;

      // Label text
      const label = document.createElementNS(svgNS, 'text');
      label.setAttribute('x', margin.left - 12);
      label.setAttribute('y', y + barH * 0.7);
      label.setAttribute('text-anchor', 'end');
      label.setAttribute('fill', textColor);
      label.setAttribute('font-size', '11.5');
      label.setAttribute('font-weight', '500');
      label.setAttribute('font-family', 'var(--font-body)');
      label.textContent = r.displayName;
      shapChartSvg.appendChild(label);

      // Bar geometry
      const barLen = (r.contribution / domainLimit) * (chartWidth / 2);
      const barX = r.contribution >= 0 ? zeroX : zeroX + barLen;
      const barW = Math.max(Math.abs(barLen), 2);

      const rect = document.createElementNS(svgNS, 'rect');
      rect.setAttribute('x', barX);
      rect.setAttribute('y', y);
      rect.setAttribute('width', barW);
      rect.setAttribute('height', barH);
      rect.setAttribute('rx', '3');
      rect.setAttribute('fill', r.contribution >= 0 ? '#ef4444' : '#38bdf8');
      rect.setAttribute('opacity', '0.9');
      shapChartSvg.appendChild(rect);

      // Value annotation text
      const valText = document.createElementNS(svgNS, 'text');
      const valX = r.contribution >= 0 ? zeroX + barW + 6 : zeroX + barLen - 6;
      valText.setAttribute('x', valX);
      valText.setAttribute('y', y + barH * 0.7);
      valText.setAttribute('text-anchor', r.contribution >= 0 ? 'start' : 'end');
      valText.setAttribute('fill', textColor);
      valText.setAttribute('font-size', '10.5');
      valText.setAttribute('font-weight', '600');
      valText.setAttribute('font-family', 'var(--font-mono)');
      valText.textContent = `${r.contribution >= 0 ? '+' : ''}${r.contribution.toFixed(4)}`;
      shapChartSvg.appendChild(valText);
    });

    // Axis label
    const axisLabel = document.createElementNS(svgNS, 'text');
    axisLabel.setAttribute('x', zeroX);
    axisLabel.setAttribute('y', height - 10);
    axisLabel.setAttribute('text-anchor', 'middle');
    axisLabel.setAttribute('fill', isLight ? '#64748b' : '#94a3b8');
    axisLabel.setAttribute('font-size', '11');
    axisLabel.setAttribute('font-weight', '600');
    axisLabel.setAttribute('font-family', 'var(--font-heading)');
    axisLabel.textContent = 'SHAP Contribution (Marginal Effect on Output)';
    shapChartSvg.appendChild(axisLabel);
  }

  /**
   * Benchmark Cards and Metric Divergence Chart
   */
  function initBenchmarkData() {
    if (!benchmarkGrid || !window.AI_CAREER_DATA) return;
    const benchmarks = window.AI_CAREER_DATA.research_results.frozen_benchmark;

    benchmarkGrid.innerHTML = '';
    for (const [key, b] of Object.entries(benchmarks)) {
      const card = document.createElement('div');
      card.className = 'cohort-stat-card';
      card.innerHTML = `
        <div class="cohort-card-title">
          <span>${b.cohort} Cohort</span>
          <span class="badge-tag badge-academic">Verified</span>
        </div>
        <div class="cohort-card-model">${b.model_name}</div>
        <div class="stat-metric-row">
          <span class="stat-key">Test F1 Score</span>
          <span class="stat-val" style="color: #818cf8;">${b.test_f1.toFixed(4)}</span>
        </div>
        <div class="stat-metric-row">
          <span class="stat-key">ROC-AUC</span>
          <span class="stat-val" style="color: ${b.test_roc_auc < 0.6 ? '#f87171' : '#34d399'};">${b.test_roc_auc.toFixed(4)}</span>
        </div>
        <div class="stat-metric-row">
          <span class="stat-key">MCC</span>
          <span class="stat-val" style="color: ${b.test_mcc < 0 ? '#f87171' : '#34d399'};">${b.test_mcc > 0 ? '+' : ''}${b.test_mcc.toFixed(4)}</span>
        </div>
        <div class="stat-metric-row">
          <span class="stat-key">10-Fold CV F1</span>
          <span class="stat-val">${b.cv_f1.toFixed(4)}</span>
        </div>
      `;
      benchmarkGrid.appendChild(card);
    }
  }

  function renderBenchmarkChart() {
    if (!benchmarkChartSvg) return;
    const svgNS = 'http://www.w3.org/2000/svg';
    const width = 900;
    const height = 340;
    benchmarkChartSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    benchmarkChartSvg.innerHTML = '';

    const cohorts = [
      { name: 'Overall', f1: 0.7994, auc: 0.5759, mcc: 0.0984 },
      { name: 'Public', f1: 0.7758, auc: 0.5199, mcc: -0.0671 },
      { name: 'Private', f1: 0.8065, auc: 0.6841, mcc: 0.1943 },
      { name: 'Daffodil', f1: 0.8019, auc: 0.7418, mcc: 0.2405 },
    ];

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const textColor = isLight ? '#1e293b' : '#f8fafc';
    const mutedColor = isLight ? '#64748b' : '#94a3b8';

    const colWidth = (width - 120) / cohorts.length;

    cohorts.forEach((c, idx) => {
      const x = 60 + idx * colWidth + 20;

      // Cohort Name
      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('x', x + 70);
      text.setAttribute('y', 40);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', textColor);
      text.setAttribute('font-family', 'var(--font-heading)');
      text.setAttribute('font-size', '14');
      text.setAttribute('font-weight', '700');
      text.textContent = c.name;
      benchmarkChartSvg.appendChild(text);

      // ROC-AUC Bar (Sky Blue)
      const aucH = c.auc * 180;
      const rectAuc = document.createElementNS(svgNS, 'rect');
      rectAuc.setAttribute('x', x + 15);
      rectAuc.setAttribute('y', 260 - aucH);
      rectAuc.setAttribute('width', 45);
      rectAuc.setAttribute('height', aucH);
      rectAuc.setAttribute('rx', '4');
      rectAuc.setAttribute('fill', '#38bdf8');
      benchmarkChartSvg.appendChild(rectAuc);

      const labelAuc = document.createElementNS(svgNS, 'text');
      labelAuc.setAttribute('x', x + 37);
      labelAuc.setAttribute('y', 250 - aucH);
      labelAuc.setAttribute('text-anchor', 'middle');
      labelAuc.setAttribute('fill', textColor);
      labelAuc.setAttribute('font-family', 'var(--font-mono)');
      labelAuc.setAttribute('font-size', '10.5');
      labelAuc.setAttribute('font-weight', '600');
      labelAuc.textContent = c.auc.toFixed(4);
      benchmarkChartSvg.appendChild(labelAuc);

      // MCC Bar (Violet/Crimson)
      const mccNorm = (c.mcc + 0.1) * 250; // shift scale for visual
      const rectMcc = document.createElementNS(svgNS, 'rect');
      rectMcc.setAttribute('x', x + 75);
      rectMcc.setAttribute('y', 260 - Math.max(mccNorm, 8));
      rectMcc.setAttribute('width', 45);
      rectMcc.setAttribute('height', Math.max(mccNorm, 8));
      rectMcc.setAttribute('rx', '4');
      rectMcc.setAttribute('fill', c.mcc < 0 ? '#ef4444' : '#a855f7');
      benchmarkChartSvg.appendChild(rectMcc);

      const labelMcc = document.createElementNS(svgNS, 'text');
      labelMcc.setAttribute('x', x + 97);
      labelMcc.setAttribute('y', 250 - Math.max(mccNorm, 8));
      labelMcc.setAttribute('text-anchor', 'middle');
      labelMcc.setAttribute('fill', textColor);
      labelMcc.setAttribute('font-family', 'var(--font-mono)');
      labelMcc.setAttribute('font-size', '10.5');
      labelMcc.setAttribute('font-weight', '600');
      labelMcc.textContent = (c.mcc > 0 ? '+' : '') + c.mcc.toFixed(4);
      benchmarkChartSvg.appendChild(labelMcc);
    });

    // Baseline axis
    const axis = document.createElementNS(svgNS, 'line');
    axis.setAttribute('x1', 50);
    axis.setAttribute('y1', 260);
    axis.setAttribute('x2', width - 50);
    axis.setAttribute('y2', 260);
    axis.setAttribute('stroke', mutedColor);
    axis.setAttribute('stroke-width', '1.5');
    benchmarkChartSvg.appendChild(axis);

    // Chart Legend
    const legend = document.createElementNS(svgNS, 'text');
    legend.setAttribute('x', width / 2);
    legend.setAttribute('y', 310);
    legend.setAttribute('text-anchor', 'middle');
    legend.setAttribute('fill', mutedColor);
    legend.setAttribute('font-family', 'var(--font-body)');
    legend.setAttribute('font-size', '12');
    legend.textContent = '■ Blue: ROC-AUC Discrimination  |  ■ Violet / Red: Matthews Correlation Coefficient (MCC)';
    benchmarkChartSvg.appendChild(legend);
  }

  /**
   * Accordions Setup
   */
  function initAccordions() {
    document.querySelectorAll('.modern-accordion').forEach(acc => {
      const header = acc.querySelector('.accordion-header');
      if (header) {
        header.addEventListener('click', () => {
          acc.classList.toggle('open');
        });
      }
    });
  }

  /**
   * DOM Ready Initializer
   */
  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initTabs();
    initSelectToggles();
    initToolChips();
    initPresets();
    initPrediction();
    initBenchmarkData();
    initAccordions();
  });

})();

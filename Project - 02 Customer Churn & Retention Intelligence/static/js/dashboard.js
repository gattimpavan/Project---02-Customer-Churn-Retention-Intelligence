/**
 * Project - 02: Customer Churn & Retention Intelligence
 * Dashboard Frontend Controller & Interactive Plotly Charts
 */

document.addEventListener('DOMContentLoaded', () => {
  initDashboard();
  setupEventListeners();
});

let globalOverviewData = null;

async function initDashboard() {
  try {
    const res = await fetch('/api/overview');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    globalOverviewData = data;
    
    // 1. Render KPIs
    renderKPIs(data);
    
    // 2. Render Charts
    renderDistributionChart(data.churn_distribution);
    renderFeatureImportanceChart(data.feature_importances);
    renderSegmentChart(data.segments);
    renderSurvivalChart(data.survival_curves);
    renderEconomicsChart(data.playbooks, data.simulation);
    
    // 3. Render Component Cards
    renderSegmentCards(data.segments);
    renderRootCauseCards(data.root_causes);
    renderPlaybookCards(data.playbooks);
    renderSimulationResults(data.simulation);
    
    // 4. Populate Quick Search Dropdown
    populateQuickSubscribers();
    
  } catch (err) {
    console.error('Failed to load overview data:', err);
  }
}

function renderKPIs(data) {
  const k = data.kpis;
  document.getElementById('kpi-cohort-size').textContent = Number(k.total_subscribers).toLocaleString() + ' Users';
  document.getElementById('kpi-churn-rate').textContent = k.churn_rate_pct.toFixed(1) + '%';
  document.getElementById('kpi-mrr-risk').textContent = '$' + Number(k.mrr_at_risk).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
  document.getElementById('kpi-model-auc').textContent = k.model_auc.toFixed(3);
  document.getElementById('kpi-retained-revenue').textContent = '$' + Number(data.simulation.annual_revenue_saved).toLocaleString(undefined, {maximumFractionDigits: 0}) + ' / yr';
  document.getElementById('kpi-campaign-roi').textContent = '+' + data.simulation.campaign_roi_pct.toFixed(0) + '%';
}

function renderDistributionChart(distData) {
  const trace = {
    x: distData.probabilities,
    type: 'histogram',
    nbinsx: 30,
    marker: {
      color: distData.probabilities.map(p => {
        if (p >= 0.70) return '#E50914';
        if (p >= 0.45) return '#F97316';
        if (p >= 0.25) return '#FBBF24';
        return '#10B981';
      }),
      line: { color: '#0F172A', width: 0.5 }
    },
    opacity: 0.88,
    hoverinfo: 'x+y'
  };

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { t: 20, r: 20, b: 45, l: 45 },
    xaxis: {
      title: { text: 'Predicted Churn Probability P(Churn)', font: { size: 11, color: '#94A3B8' } },
      gridcolor: 'rgba(255,255,255,0.06)',
      tickfont: { color: '#94A3B8', size: 10 }
    },
    yaxis: {
      title: { text: 'Subscriber Count', font: { size: 11, color: '#94A3B8' } },
      gridcolor: 'rgba(255,255,255,0.06)',
      tickfont: { color: '#94A3B8', size: 10 }
    },
    shapes: [{
      type: 'line',
      x0: 0.50,
      x1: 0.50,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: '#FFFFFF', width: 1.5, dash: 'dash' }
    }],
    annotations: [{
      x: 0.52,
      y: 0.95,
      yref: 'paper',
      text: 'Threshold: 0.50',
      showarrow: false,
      font: { color: '#FFFFFF', size: 10, weight: 'bold' }
    }]
  };

  Plotly.newPlot('chart-churn-dist', [trace], layout, { responsive: true, displayModeBar: false });
}

function renderFeatureImportanceChart(features) {
  const top10 = features.slice(0, 10).reverse();
  const yNames = top10.map(f => f.displayName);
  const xVals = top10.map(f => f.importance * 100);

  const trace = {
    x: xVals,
    y: yNames,
    type: 'bar',
    orientation: 'h',
    marker: {
      color: xVals.map(v => v >= 10 ? '#E50914' : '#3B82F6'),
      line: { color: '#0F172A', width: 0.5 }
    },
    text: xVals.map(v => v.toFixed(1) + '%'),
    textposition: 'outside',
    textfont: { color: '#F8FAFC', size: 10 }
  };

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { t: 20, r: 40, b: 40, l: 160 },
    xaxis: {
      title: { text: 'Relative Importance (%)', font: { size: 11, color: '#94A3B8' } },
      gridcolor: 'rgba(255,255,255,0.06)',
      tickfont: { color: '#94A3B8', size: 10 }
    },
    yaxis: {
      tickfont: { color: '#F8FAFC', size: 10.5 }
    }
  };

  Plotly.newPlot('chart-feature-imp', [trace], layout, { responsive: true, displayModeBar: false });
}

function renderSegmentChart(segments) {
  const segNames = segments.map(s => s.segment);
  const counts = segments.map(s => s.count);
  const mrrRisk = segments.map(s => s.mrr_at_risk);

  const traceCount = {
    x: segNames,
    y: counts,
    name: 'Subscribers Count',
    type: 'bar',
    marker: { color: '#475569' }
  };

  const traceMRR = {
    x: segNames,
    y: mrrRisk,
    name: 'MRR at Risk ($)',
    type: 'bar',
    yaxis: 'y2',
    marker: { color: '#E50914' }
  };

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { t: 25, r: 50, b: 60, l: 50 },
    barmode: 'group',
    legend: { orientation: 'h', x: 0.1, y: 1.15, font: { color: '#F8FAFC', size: 10 } },
    xaxis: {
      tickfont: { color: '#F8FAFC', size: 10 },
      tickangle: -15
    },
    yaxis: {
      title: { text: 'Cohort Size', font: { size: 11, color: '#94A3B8' } },
      gridcolor: 'rgba(255,255,255,0.06)',
      tickfont: { color: '#94A3B8', size: 10 }
    },
    yaxis2: {
      title: { text: 'MRR at Risk ($)', font: { size: 11, color: '#E50914' } },
      overlaying: 'y',
      side: 'right',
      tickfont: { color: '#E50914', size: 10 }
    }
  };

  Plotly.newPlot('chart-segment-mrr', [traceCount, traceMRR], layout, { responsive: true, displayModeBar: false });
}

function renderSurvivalChart(survival) {
  const months = survival.tenure_months;
  
  const traces = [
    {
      x: months,
      y: survival.overall.map(v => v * 100),
      name: 'Overall Portfolio S(t)',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#FFFFFF', width: 2.5 }
    },
    {
      x: months,
      y: survival.tiers['Standard'].map(v => v * 100),
      name: 'Standard ($15.49)',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#3B82F6', width: 1.8, dash: 'dot' }
    },
    {
      x: months,
      y: survival.tiers['Premium'].map(v => v * 100),
      name: 'Premium 4K ($22.99)',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#E50914', width: 1.8, dash: 'dash' }
    },
    {
      x: months,
      y: survival.tiers['Standard with Ads'].map(v => v * 100),
      name: 'Standard with Ads ($6.99)',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#10B981', width: 1.8 }
    }
  ];

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { t: 25, r: 25, b: 45, l: 50 },
    legend: { orientation: 'h', x: 0.05, y: 1.15, font: { color: '#F8FAFC', size: 10 } },
    xaxis: {
      title: { text: 'Tenure in Months', font: { size: 11, color: '#94A3B8' } },
      gridcolor: 'rgba(255,255,255,0.06)',
      tickfont: { color: '#94A3B8', size: 10 }
    },
    yaxis: {
      title: { text: 'Retention Probability S(t) %', font: { size: 11, color: '#94A3B8' } },
      gridcolor: 'rgba(255,255,255,0.06)',
      tickfont: { color: '#94A3B8', size: 10 },
      range: [35, 102]
    }
  };

  Plotly.newPlot('chart-survival-curve', traces, layout, { responsive: true, displayModeBar: false });
}

function renderEconomicsChart(playbooks, sim) {
  const pbs = sim.playbook_breakdown;
  const names = pbs.map(p => p.playbook_name.split(' & ')[0]);
  const costs = pbs.map(p => p.cost_spent);
  const annual = pbs.map(p => p.annual_saved);
  const net = pbs.map(p => p.annual_saved - p.cost_spent);

  const traceCost = {
    x: names,
    y: costs,
    name: 'Campaign Cost ($)',
    type: 'bar',
    marker: { color: '#64748B' }
  };

  const traceRev = {
    x: names,
    y: annual,
    name: 'Gross Saved ($/yr)',
    type: 'bar',
    marker: { color: '#3B82F6' }
  };

  const traceNet = {
    x: names,
    y: net,
    name: 'Net Profit ($/yr)',
    type: 'bar',
    marker: { color: '#10B981' }
  };

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { t: 25, r: 25, b: 60, l: 55 },
    barmode: 'group',
    legend: { orientation: 'h', x: 0.1, y: 1.15, font: { color: '#F8FAFC', size: 10 } },
    xaxis: {
      tickfont: { color: '#F8FAFC', size: 10 },
      tickangle: -15
    },
    yaxis: {
      title: { text: 'Financial Impact ($ USD)', font: { size: 11, color: '#94A3B8' } },
      gridcolor: 'rgba(255,255,255,0.06)',
      tickfont: { color: '#94A3B8', size: 10 }
    }
  };

  Plotly.newPlot('chart-retention-waterfall', [traceCost, traceRev, traceNet], layout, { responsive: true, displayModeBar: false });
}

function renderSegmentCards(segments) {
  const container = document.getElementById('segments-container');
  container.innerHTML = '';

  segments.forEach(s => {
    const card = document.createElement('div');
    card.className = 'segment-card';
    card.style.borderTop = `3px solid ${s.color}`;

    card.innerHTML = `
      <div>
        <div class="segment-header">
          <div class="segment-name">${s.segment}</div>
          <span class="segment-badge" style="background:${s.color}22; color:${s.color}">${s.risk_level}</span>
        </div>
        <div class="segment-metric-row">
          <span class="segment-metric-label">Cohort Size:</span>
          <span class="segment-metric-val">${s.count.toLocaleString()} (${s.percentage}%)</span>
        </div>
        <div class="segment-metric-row">
          <span class="segment-metric-label">MRR at Risk:</span>
          <span class="segment-metric-val red">$${s.mrr_at_risk.toLocaleString()} / mo</span>
        </div>
        <div class="segment-metric-row">
          <span class="segment-metric-label">Avg Churn Probability:</span>
          <span class="segment-metric-val">${s.avg_churn_prob_pct}%</span>
        </div>
        <div class="segment-metric-row">
          <span class="segment-metric-label">Usage Velocity:</span>
          <span class="segment-metric-val">${s.avg_usage_velocity}x</span>
        </div>
        <div class="segment-desc">${s.description}</div>
      </div>
      <div class="segment-action">🎯 ${s.prescribed_strategy}</div>
    `;
    container.appendChild(card);
  });
}

function renderRootCauseCards(rootCauses) {
  const container = document.getElementById('root-causes-container');
  container.innerHTML = '';

  rootCauses.forEach(rc => {
    const card = document.createElement('div');
    card.className = 'root-cause-card';

    card.innerHTML = `
      <div class="root-cause-top">
        <span class="root-cause-title">${rc.cause}</span>
        <span class="root-cause-pct" style="color:${rc.color}">${rc.pct_of_churn}%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" style="width: ${rc.pct_of_churn}%; background: ${rc.color}"></div>
      </div>
      <div class="root-cause-metric-row">
        <span>Affected Subscribers:</span>
        <b>${rc.affected_subscribers.toLocaleString()} Users</b>
      </div>
      <div class="root-cause-metric-row">
        <span>Lost MRR:</span>
        <b style="color:${rc.color}">$${Number(rc.lost_mrr).toLocaleString(undefined, {minimumFractionDigits: 2})} / mo</b>
      </div>
      <div class="root-cause-benchmark">
        <b>Trigger Condition:</b> ${rc.benchmark_metric}
      </div>
    `;
    container.appendChild(card);
  });
}

function renderPlaybookCards(playbooks) {
  const container = document.getElementById('playbooks-container');
  container.innerHTML = '';

  playbooks.forEach(pb => {
    const card = document.createElement('div');
    card.className = 'playbook-card';

    card.innerHTML = `
      <div>
        <span class="playbook-badge">${pb.target_segment}</span>
        <h4 class="playbook-title">${pb.name}</h4>
        <div class="playbook-target">Channel: ${pb.channel}</div>
        <div class="playbook-metrics">
          <div class="playbook-metric-item">
            <span style="color:#94A3B8">Unit Cost:</span>
            <div class="val">$${pb.intervention_cost.toFixed(2)}</div>
          </div>
          <div class="playbook-metric-item">
            <span style="color:#94A3B8">Expected Lift:</span>
            <div class="val" style="color:#10B981">-${(pb.relative_churn_reduction * 100).toFixed(0)}% Churn</div>
          </div>
        </div>
      </div>
      <div class="playbook-copy-box">"${pb.message_copy}"</div>
    `;
    container.appendChild(card);
  });
}

function renderSimulationResults(sim) {
  document.getElementById('sim-saved-subs').textContent = sim.projected_saved_subscribers.toLocaleString();
  document.getElementById('sim-churn-before-after').textContent = `${sim.baseline_churn_rate_pct.toFixed(1)}% → ${sim.projected_churn_rate_pct.toFixed(1)}%`;
  document.getElementById('sim-mrr-saved').textContent = '$' + Number(sim.monthly_mrr_saved).toLocaleString(undefined, {minimumFractionDigits: 2});
  document.getElementById('sim-annual-saved').textContent = '$' + Number(sim.annual_revenue_saved).toLocaleString(undefined, {maximumFractionDigits: 0});
  document.getElementById('sim-net-profit').textContent = '$' + Number(sim.net_annual_profit).toLocaleString(undefined, {maximumFractionDigits: 0});
  document.getElementById('sim-roi').textContent = '+' + sim.campaign_roi_pct.toFixed(0) + '%';
}

function setupEventListeners() {
  const slider = document.getElementById('budget-slider');
  const sliderVal = document.getElementById('slider-val-display');

  if (slider) {
    slider.addEventListener('input', (e) => {
      sliderVal.textContent = '$' + Number(e.target.value).toLocaleString();
    });
  }

  const btnSim = document.getElementById('btn-run-simulation');
  if (btnSim) {
    btnSim.addEventListener('click', runSimulation);
  }

  const btnLookup = document.getElementById('btn-lookup-cust');
  if (btnLookup) {
    btnLookup.addEventListener('click', lookupSubscriber);
  }

  const quickSelect = document.getElementById('quick-select-dropdown');
  if (quickSelect) {
    quickSelect.addEventListener('change', (e) => {
      if (e.target.value) {
        document.getElementById('customer-search-input').value = e.target.value;
        lookupSubscriber();
      }
    });
  }
}

async function runSimulation() {
  const budget = parseFloat(document.getElementById('budget-slider').value);
  const checkboxes = document.querySelectorAll('input[name="sim-playbook"]:checked');
  const activePlaybooks = Array.from(checkboxes).map(c => c.value);

  try {
    const res = await fetch('/api/simulate-retention', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ budget, active_playbooks: activePlaybooks })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const sim = await res.json();
    renderSimulationResults(sim);
    if (globalOverviewData) {
      renderEconomicsChart(globalOverviewData.playbooks, sim);
    }
  } catch (err) {
    console.error('Simulation error:', err);
    alert('Simulation error. Please check backend.');
  }
}

async function populateQuickSubscribers() {
  try {
    const res = await fetch('/api/quick-subscribers');
    if (!res.ok) return;
    const subs = await res.json();
    const dropdown = document.getElementById('quick-select-dropdown');
    dropdown.innerHTML = '<option value="">-- Quick Select Sample Subscriber --</option>';
    subs.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.customer_id;
      opt.textContent = `${s.customer_id} | ${s.segment} (${s.risk_tier} Risk - P: ${(s.churn_probability*100).toFixed(0)}%)`;
      dropdown.appendChild(opt);
    });

    // Default select first customer
    if (subs.length > 0) {
      document.getElementById('customer-search-input').value = subs[0].customer_id;
      lookupSubscriber();
    }
  } catch (err) {
    console.error('Failed to populate quick subscribers:', err);
  }
}

async function lookupSubscriber() {
  const custId = document.getElementById('customer-search-input').value.trim();
  if (!custId) return;

  try {
    const res = await fetch(`/api/customer/${encodeURIComponent(custId)}`);
    if (!res.ok) {
      alert(`Customer ${custId} not found.`);
      return;
    }
    const data = await res.json();
    renderSubscriberDetails(data);
  } catch (err) {
    console.error('Customer lookup failed:', err);
  }
}

function renderSubscriberDetails(data) {
  const c = data.profile;
  const p = data.prescription;

  // 1. Profile Telemetry
  document.getElementById('sub-id-display').textContent = c.customer_id;
  document.getElementById('sub-tier-display').textContent = c.subscription_tier + ` ($${c.monthly_fee}/mo)`;
  document.getElementById('sub-tenure-display').textContent = c.tenure_months + ' Months';
  document.getElementById('sub-device-display').textContent = c.primary_device;
  document.getElementById('sub-watch-hours').textContent = c.watch_hours_last_30d + ' hrs (30d)';
  document.getElementById('sub-velocity').textContent = c.usage_velocity + 'x';
  document.getElementById('sub-days-inactive').textContent = c.days_since_last_watch + ' Days';
  document.getElementById('sub-buffering').textContent = c.buffering_incidents_per_month + ' Events';
  document.getElementById('sub-tickets').textContent = c.customer_support_tickets + ' Tickets';

  // 2. Risk & Diagnostics
  const riskColor = c.churn_probability >= 0.70 ? '#E50914' : (c.churn_probability >= 0.45 ? '#F97316' : (c.churn_probability >= 0.25 ? '#FBBF24' : '#10B981'));
  const scoreElem = document.getElementById('sub-risk-score');
  scoreElem.textContent = (c.churn_probability * 100).toFixed(1) + '%';
  scoreElem.style.color = riskColor;
  document.getElementById('sub-risk-tier').textContent = c.risk_tier + ' Risk';
  document.getElementById('sub-segment-tag').textContent = c.segment;

  const driversContainer = document.getElementById('sub-drivers-container');
  driversContainer.innerHTML = '';
  c.churn_drivers.forEach(d => {
    const pElem = document.createElement('div');
    pElem.className = 'driver-pill';
    pElem.innerHTML = `
      <div class="cat">${d.category} [${d.severity}]</div>
      <div class="desc">${d.description}</div>
    `;
    driversContainer.appendChild(pElem);
  });

  // 3. Prescription
  document.getElementById('sub-rx-strategy').textContent = p.strategy_name;
  document.getElementById('sub-rx-channel').textContent = p.recommended_channel;
  document.getElementById('sub-rx-offer').textContent = p.offer_type;
  document.getElementById('sub-rx-copy').textContent = `"${p.personalized_copy}"`;
  document.getElementById('sub-rx-cost').textContent = '$' + p.intervention_cost.toFixed(2);
  document.getElementById('sub-rx-saved').textContent = '$' + p.expected_annual_saved.toFixed(2) + ' / yr';
  document.getElementById('sub-rx-roi').textContent = '+' + p.roi_pct.toFixed(0) + '%';
}

/**
 * Retail Product Demand Estimation & Stockout Risk Analysis
 * Frontend Client Controller
 */

// Global state and chart references
const state = {
  products: [],
  productsMap: {},
  dashboardData: null,
  salesTrends: null,
  modelMetrics: null,
  charts: {}
};

// Section header text mapping
const sectionMeta = {
  dashboard: {
    title: "Dashboard Overview",
    desc: "Real-time retail inventory intelligence & sales telemetry"
  },
  forecasting: {
    title: "Demand Forecasting",
    desc: "Autoregressive multi-step supervised machine learning predictions"
  },
  stockout: {
    title: "Stockout Risk Analysis",
    desc: "Forecast-driven inventory simulation and replenishment optimization"
  },
  performance: {
    title: "Model Performance & Evaluation",
    desc: "Chronological validation comparison of Linear Regression, Random Forest & Gradient Boosting"
  },
  products: {
    title: "Product Catalog",
    desc: "Search and inspect active retail SKUs with historical statistics"
  },
  eda: {
    title: "Exploratory Data Analysis Gallery",
    desc: "Statistical insights extracted from the cleaned UCI Online Retail dataset"
  }
};

// Initialization on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupForecastingEvents();
  setupStockoutEvents();
  setupCatalogSearch();
  loadAllApplicationData();
});

// ---------------- NAVIGATION ----------------
function setupNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const section = item.dataset.section;
      switchToSection(section);
    });
  });
}

function switchToSection(sectionName) {
  // Update sidebar active class
  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.section === sectionName);
  });

  // Switch visible section
  document.querySelectorAll(".page-section").forEach(sec => {
    sec.classList.toggle("active", sec.id === `section-${sectionName}`);
  });

  // Update header text
  const meta = sectionMeta[sectionName] || sectionMeta.dashboard;
  document.getElementById("current-section-title").innerText = meta.title;
  document.getElementById("current-section-desc").innerText = meta.desc;

  // Trigger chart resize if needed
  window.dispatchEvent(new Event("resize"));
}

// ---------------- DATA LOADING ----------------
async function loadAllApplicationData() {
  try {
    checkHealth();
    await Promise.all([
      fetchDashboardMetrics(),
      fetchSalesTrends(),
      fetchProductCatalog(),
      fetchModelMetrics()
    ]);
  } catch (err) {
    showToast(`Initialization notice: ${err.message}`, true);
  }
}

async function checkHealth() {
  try {
    const res = await fetch("/status");
    if (res.ok) {
      const data = await res.json();
      document.getElementById("api-status-text").innerText = `Engine: ${data.best_model_name} Online`;
    }
  } catch (e) {
    document.getElementById("api-status-text").innerText = "API Offline";
  }
}

// 1. Dashboard Metrics
async function fetchDashboardMetrics() {
  try {
    const res = await fetch("/dashboard");
    if (!res.ok) throw new Error("Could not load dashboard statistics");
    const data = await res.json();
    state.dashboardData = data;

    // Update KPI numbers
    document.getElementById("kpi-total-products").innerText = Number(data.total_products).toLocaleString();
    document.getElementById("kpi-total-transactions").innerText = Number(data.total_transactions).toLocaleString();
    document.getElementById("kpi-total-units").innerText = Number(data.total_sales_quantity).toLocaleString();
    document.getElementById("kpi-avg-daily").innerText = Number(data.avg_daily_retail_demand).toLocaleString();
    document.getElementById("kpi-high-risk").innerText = Number(data.simulated_high_risk_skus).toLocaleString();
    document.getElementById("kpi-total-revenue").innerText = "£" + Number(data.total_revenue).toLocaleString(undefined, { minimumFractionDigits: 2 });

    // Populate Best Sellers Table
    populateDashboardBestsellers(data.best_selling_products);
  } catch (err) {
    console.error("Dashboard error:", err);
  }
}

function populateDashboardBestsellers(products) {
  const tbody = document.getElementById("dashboard-bestsellers-tbody");
  if (!tbody || !products) return;

  tbody.innerHTML = products.map((p, idx) => `
    <tr>
      <td><span class="badge-rank">#${idx + 1}</span></td>
      <td><strong>${p.StockCode}</strong></td>
      <td>${p.Description}</td>
      <td>${Number(p.TotalQuantity).toLocaleString()} units</td>
      <td>£${Number(p.TotalRevenue).toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
      <td>
        <button class="action-btn-sm" onclick="quickForecast('${p.StockCode}')">Forecast</button>
      </td>
    </tr>
  `).join("");
}

// 2. Sales Trends Charts
async function fetchSalesTrends() {
  try {
    const res = await fetch("/sales-trends");
    if (!res.ok) throw new Error("Could not load sales trend telemetry");
    const data = await res.json();
    state.salesTrends = data;

    renderMonthlySalesChart(data.monthly_sales);
    renderDayOfWeekChart(data.day_of_week);
    renderRecentDemandChart(data.recent_daily_trajectory);
  } catch (err) {
    console.error("Sales trend error:", err);
  }
}

function renderMonthlySalesChart(monthly) {
  const ctx = document.getElementById("monthlySalesChart");
  if (!ctx) return;
  if (state.charts.monthly) state.charts.monthly.destroy();

  state.charts.monthly = new Chart(ctx, {
    type: "bar",
    data: {
      labels: monthly.labels,
      datasets: [
        {
          label: "Total Units Sold",
          data: monthly.quantity,
          backgroundColor: "rgba(59, 130, 246, 0.7)",
          borderColor: "#3b82f6",
          borderWidth: 1,
          borderRadius: 4,
          yAxisID: "y"
        },
        {
          label: "Total Gross Revenue (£)",
          data: monthly.revenue,
          type: "line",
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245, 158, 11, 0.15)",
          pointBackgroundColor: "#f59e0b",
          borderWidth: 2.5,
          tension: 0.3,
          yAxisID: "y1"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8" } }
      },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } },
        y: {
          type: "linear",
          position: "left",
          ticks: { color: "#94a3b8" },
          grid: { color: "#1e293b" },
          title: { display: true, text: "Units Sold", color: "#60a5fa" }
        },
        y1: {
          type: "linear",
          position: "right",
          ticks: { color: "#f59e0b" },
          grid: { drawOnChartArea: false },
          title: { display: true, text: "Gross Revenue (£)", color: "#f59e0b" }
        }
      }
    }
  });
}

function renderDayOfWeekChart(dow) {
  const ctx = document.getElementById("dowChart");
  if (!ctx) return;
  if (state.charts.dow) state.charts.dow.destroy();

  state.charts.dow = new Chart(ctx, {
    type: "bar",
    data: {
      labels: dow.labels,
      datasets: [{
        label: "Total Demand",
        data: dow.quantity,
        backgroundColor: [
          "#3b82f6", "#60a5fa", "#38bdf8", "#2dd4bf", "#34d399", "#818cf8", "#c084fc"
        ],
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } }
      }
    }
  });
}

function renderRecentDemandChart(recent) {
  const ctx = document.getElementById("recentDemandChart");
  if (!ctx) return;
  if (state.charts.recent) state.charts.recent.destroy();

  state.charts.recent = new Chart(ctx, {
    type: "line",
    data: {
      labels: recent.dates,
      datasets: [{
        label: "Daily Demand Quantity",
        data: recent.quantity,
        borderColor: "#10b981",
        backgroundColor: "rgba(16, 185, 129, 0.12)",
        fill: true,
        borderWidth: 2,
        tension: 0.25,
        pointRadius: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#94a3b8", maxTicksLimit: 8 }, grid: { color: "#1e293b" } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } }
      }
    }
  });
}

// 3. Product Catalog
async function fetchProductCatalog() {
  try {
    const res = await fetch("/products?limit=250");
    if (!res.ok) throw new Error("Could not load product catalog");
    const data = await res.json();
    state.products = data.products || [];
    state.productsMap = {};
    state.products.forEach(p => { state.productsMap[p.StockCode] = p; });

    populateProductDropdowns(state.products);
    populateCatalogTable(state.products);

    // Default select first product
    if (state.products.length > 0) {
      updateProductQuickSpecs(state.products[0].StockCode);
    }
  } catch (err) {
    console.error("Catalog error:", err);
  }
}

function populateProductDropdowns(products) {
  const forecastSelect = document.getElementById("forecast-product-select");
  const stockoutSelect = document.getElementById("stockout-product-select");

  const optionsHtml = products.map(p => {
    const shortDesc = p.Description ? (p.Description.length > 35 ? p.Description.slice(0, 35) + "..." : p.Description) : "RETAIL PRODUCT";
    return `<option value="${p.StockCode}">${p.StockCode} - ${shortDesc} (Avg: ${Math.round(p.AvgDailyDemand || 0)}/day)</option>`;
  }).join("");

  if (forecastSelect) forecastSelect.innerHTML = optionsHtml;
  if (stockoutSelect) stockoutSelect.innerHTML = optionsHtml;
}

function populateCatalogTable(products) {
  const tbody = document.getElementById("catalog-table-tbody");
  if (!tbody) return;

  if (products.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="text-center">No products matching search criteria.</td></tr>`;
    return;
  }

  tbody.innerHTML = products.slice(0, 100).map(p => `
    <tr>
      <td><strong>${p.StockCode}</strong></td>
      <td>${p.Description || 'N/A'}</td>
      <td>${Number(p.TotalDemand || 0).toLocaleString()}</td>
      <td>${p.ActiveDays || 0} days</td>
      <td>${Math.round(p.AvgDailyDemand || 0)} units/day</td>
      <td>£${Number(p.AvgPrice || 0).toFixed(2)}</td>
      <td>
        <button class="action-btn-sm" onclick="quickForecast('${p.StockCode}')">Forecast</button>
        <button class="action-btn-sm" onclick="quickStockout('${p.StockCode}')">Stockout Risk</button>
      </td>
    </tr>
  `).join("");
}

function updateProductQuickSpecs(stockCode) {
  const prod = state.productsMap[stockCode];
  const specsBox = document.getElementById("forecast-quick-specs");
  if (!prod || !specsBox) return;

  document.getElementById("spec-code").innerText = prod.StockCode;
  document.getElementById("spec-desc").innerText = prod.Description || "N/A";
  document.getElementById("spec-price").innerText = "£" + Number(prod.AvgPrice || 0).toFixed(2);
  document.getElementById("spec-days").innerText = `${prod.ActiveDays || 0} active days (${Math.round(prod.AvgDailyDemand || 0)} units/day)`;
  specsBox.style.display = "block";
}

function setupCatalogSearch() {
  const searchInput = document.getElementById("catalog-search-input");
  if (!searchInput) return;

  searchInput.addEventListener("input", (e) => {
    const term = e.target.value.trim().toUpperCase();
    if (!term) {
      populateCatalogTable(state.products);
      return;
    }
    const filtered = state.products.filter(p => 
      (p.StockCode && p.StockCode.toUpperCase().includes(term)) ||
      (p.Description && p.Description.toUpperCase().includes(term))
    );
    populateCatalogTable(filtered);
  });
}

// 4. Model Performance Comparison
async function fetchModelMetrics() {
  try {
    const res = await fetch("/model-comparison");
    if (!res.ok) throw new Error("Could not load model comparison data");
    const data = await res.json();
    state.modelMetrics = data;

    const m = data.metrics || {};
    // Populate cards
    if (m["Linear Regression"]) {
      document.getElementById("m-lr-mae").innerText = m["Linear Regression"].MAE;
      document.getElementById("m-lr-rmse").innerText = m["Linear Regression"].RMSE;
      document.getElementById("m-lr-r2").innerText = m["Linear Regression"].R2;
    }
    if (m["Random Forest"]) {
      document.getElementById("m-rf-mae").innerText = m["Random Forest"].MAE;
      document.getElementById("m-rf-rmse").innerText = m["Random Forest"].RMSE;
      document.getElementById("m-rf-r2").innerText = m["Random Forest"].R2;
    }
    if (m["Gradient Boosting"]) {
      document.getElementById("m-gb-mae").innerText = m["Gradient Boosting"].MAE;
      document.getElementById("m-gb-rmse").innerText = m["Gradient Boosting"].RMSE;
      document.getElementById("m-gb-r2").innerText = m["Gradient Boosting"].R2;
    }
    if (m["Tuned Gradient Boosting (Best)"]) {
      document.getElementById("m-tuned-mae").innerText = m["Tuned Gradient Boosting (Best)"].MAE;
      document.getElementById("m-tuned-rmse").innerText = m["Tuned Gradient Boosting (Best)"].RMSE;
      document.getElementById("m-tuned-r2").innerText = m["Tuned Gradient Boosting (Best)"].R2;
    }

    renderModelComparisonCharts(m);
  } catch (err) {
    console.error("Model metrics error:", err);
  }
}

function renderModelComparisonCharts(metrics) {
  const ctx = document.getElementById("modelComparisonChart");
  const ctxR2 = document.getElementById("r2ComparisonChart");
  if (!ctx || !ctxR2) return;

  const labels = Object.keys(metrics);
  const maeData = labels.map(k => metrics[k].MAE);
  const rmseData = labels.map(k => metrics[k].RMSE);
  const r2Data = labels.map(k => metrics[k].R2);

  // 1. Error Metrics Chart
  if (state.charts.modelComp) state.charts.modelComp.destroy();
  state.charts.modelComp = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Mean Absolute Error (MAE)",
          data: maeData,
          backgroundColor: "#3b82f6",
          borderRadius: 4
        },
        {
          label: "Root Mean Squared Error (RMSE)",
          data: rmseData,
          backgroundColor: "#ef4444",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94a3b8" } } },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" }, title: { display: true, text: "Error Magnitude", color: "#94a3b8" } }
      }
    }
  });

  // 2. R2 Goodness of Fit Chart
  if (state.charts.r2Comp) state.charts.r2Comp.destroy();
  state.charts.r2Comp = new Chart(ctxR2, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "R² Coefficient of Determination",
        data: r2Data,
        backgroundColor: [
          "#94a3b8", "#60a5fa", "#10b981", "#34d399"
        ],
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } },
        y: {
          min: 0,
          max: 0.45,
          ticks: { color: "#94a3b8" },
          grid: { color: "#1e293b" },
          title: { display: true, text: "R² Score", color: "#94a3b8" }
        }
      }
    }
  });
}

// ---------------- FORECASTING CONTROLLER ----------------
function setupForecastingEvents() {
  const productSelect = document.getElementById("forecast-product-select");
  if (productSelect) {
    productSelect.addEventListener("change", (e) => {
      updateProductQuickSpecs(e.target.value);
    });
  }

  // Duration chip buttons
  const chips = document.querySelectorAll(".duration-buttons .btn-chip");
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      chips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      document.getElementById("forecast-days-input").value = chip.dataset.days;
    });
  });

  // Form submission
  const form = document.getElementById("forecast-form");
  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      await executeDemandForecast();
    });
  }
}

async function executeDemandForecast() {
  const stockCode = document.getElementById("forecast-product-select").value;
  const days = parseInt(document.getElementById("forecast-days-input").value);
  const spinner = document.getElementById("forecast-spinner");
  const btn = document.getElementById("btn-run-forecast");

  if (!stockCode) {
    showToast("Please select a product SKU first", true);
    return;
  }

  try {
    spinner.classList.remove("d-none");
    btn.disabled = true;

    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ product_id: stockCode, forecast_days: days })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction request failed");
    }

    const payload = await res.json();
    displayForecastResults(payload.data);
    showToast(`Forecast generated for ${stockCode} across ${days} days!`);
  } catch (err) {
    showToast(err.message, true);
  } finally {
    spinner.classList.add("d-none");
    btn.disabled = false;
  }
}

function displayForecastResults(data) {
  // Reveal summaries
  document.getElementById("forecast-summary-cards").style.display = "grid";
  document.getElementById("forecast-schedule-card").style.display = "block";

  // KPIs
  document.getElementById("f-total-demand").innerText = Number(data.total_forecast_demand).toLocaleString();
  document.getElementById("f-avg-daily").innerText = Number(data.avg_daily_demand).toLocaleString();
  document.getElementById("f-peak-demand").innerText = Number(data.peak_daily_demand).toLocaleString();

  const trendBadge = document.getElementById("f-trend-badge");
  trendBadge.innerText = data.trend_direction;
  trendBadge.className = "mini-value trend-badge " + data.trend_direction.toLowerCase();
  document.getElementById("f-trend-sub").innerText = `Baseline: ${data.recent_historical_avg}/day`;

  // Render Forecast Chart
  renderForecastChart(data);

  // Render Schedule Table
  const tbody = document.getElementById("forecast-schedule-tbody");
  tbody.innerHTML = data.forecast_schedule.map(row => `
    <tr>
      <td>${row.date}</td>
      <td><strong>${row.day_name}</strong></td>
      <td><span class="forecast-value">${row.predicted_demand} units</span></td>
    </tr>
  `).join("");
}

function renderForecastChart(data) {
  const ctx = document.getElementById("forecastChart");
  if (!ctx) return;
  if (state.charts.forecast) state.charts.forecast.destroy();

  const history = data.historical_baseline || [];
  const future = data.forecast_schedule || [];

  const labels = [
    ...history.map(h => h.date.slice(5)),
    ...future.map(f => f.date.slice(5) + " (F)")
  ];

  const historicalValues = [
    ...history.map(h => h.actual_demand),
    ...future.map(() => null)
  ];

  // Bridge last historical point to forecast
  const lastActual = history.length > 0 ? history[history.length - 1].actual_demand : null;
  const forecastValues = [
    ...history.slice(0, -1).map(() => null),
    lastActual,
    ...future.map(f => f.predicted_demand)
  ];

  state.charts.forecast = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Recent Historical Demand",
          data: historicalValues,
          borderColor: "#94a3b8",
          backgroundColor: "rgba(148, 163, 184, 0.1)",
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.2
        },
        {
          label: "Machine Learning Forecast",
          data: forecastValues,
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.15)",
          borderWidth: 2.5,
          borderDash: [5, 5],
          pointRadius: 3,
          pointBackgroundColor: "#10b981",
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8" } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw} units`
          }
        }
      },
      scales: {
        x: { ticks: { color: "#94a3b8", maxTicksLimit: 12 }, grid: { color: "#1e293b" } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" }, title: { display: true, text: "Daily Demand Units", color: "#94a3b8" } }
      }
    }
  });
}

// ---------------- STOCKOUT RISK CONTROLLER ----------------
function setupStockoutEvents() {
  const form = document.getElementById("stockout-form");
  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      await executeStockoutAnalysis();
    });
  }
}

async function executeStockoutAnalysis() {
  const stockCode = document.getElementById("stockout-product-select").value;
  const inventory = parseInt(document.getElementById("stockout-inventory-input").value);
  const leadTime = parseInt(document.getElementById("stockout-leadtime-input").value);
  const safetyBuf = parseFloat(document.getElementById("stockout-safety-buffer").value);

  const spinner = document.getElementById("stockout-spinner");
  const btn = document.getElementById("btn-run-stockout");

  if (!stockCode) {
    showToast("Please choose a target product SKU", true);
    return;
  }

  try {
    spinner.classList.remove("d-none");
    btn.disabled = true;

    const res = await fetch("/stockout-risk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        product_id: stockCode,
        available_inventory: inventory,
        supplier_lead_time: leadTime,
        safety_buffer_pct: safetyBuf
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Stockout calculation failed");
    }

    const payload = await res.json();
    displayStockoutResults(payload.data);
    showToast(`Stockout assessment completed: ${payload.data.stockout_status} RISK`);
  } catch (err) {
    showToast(err.message, true);
  } finally {
    spinner.classList.add("d-none");
    btn.disabled = false;
  }
}

function displayStockoutResults(data) {
  // Update Risk Pill & Explanation
  const pill = document.getElementById("risk-status-pill");
  pill.innerText = `${data.stockout_status} STOCKOUT RISK`;
  pill.className = "risk-pill-large " + data.stockout_status.toLowerCase();

  document.getElementById("risk-explanation-text").innerText = data.risk_analysis_summary;

  // Reveal metrics
  document.getElementById("stockout-metrics-box").style.display = "grid";
  document.getElementById("stockout-action-card").style.display = "block";

  // Values
  document.getElementById("so-lead-demand").innerText = Number(data.expected_lead_time_demand).toLocaleString();
  document.getElementById("so-avail-inv").innerText = Number(data.available_inventory).toLocaleString();
  document.getElementById("so-remain-inv").innerText = Number(data.inventory_remaining_after_lead_time).toLocaleString();
  document.getElementById("so-reorder-qty").innerText = Number(data.suggested_replenishment).toLocaleString();
  document.getElementById("so-runway-days").innerText = data.estimated_runway_days > 365 ? "> 1 Year" : `${data.estimated_runway_days} Days`;
  document.getElementById("so-safety-buf").innerText = `${data.safety_stock_buffer} units (${data.safety_threshold_pct}%)`;

  // Operational Action Box
  const actionCard = document.getElementById("stockout-action-card");
  const actionTitle = document.getElementById("action-title-text");
  const actionBody = document.getElementById("action-body-text");

  if (data.stockout_status === "HIGH") {
    actionCard.className = "action-card danger mt-4";
    actionTitle.innerText = "CRITICAL ACTION REQUIRED: Immediate Replenishment Order";
    actionBody.innerText = `Product ${data.stock_code} will face a warehouse stock deficit during the ${data.supplier_lead_time_days}-day supplier lead window. Issue a replenishment order for ${data.suggested_replenishment} units immediately to maintain uninterrupted customer fulfillment and uphold safety buffer levels.`;
  } else {
    actionCard.className = "action-card mt-4";
    actionTitle.innerText = "INVENTORY HEALTHY: Optimal Coverage Maintained";
    actionBody.innerText = `Current available inventory of ${data.available_inventory} units is sufficient to fulfill all ${data.expected_lead_time_demand} projected customer orders throughout the ${data.supplier_lead_time_days}-day replenishment period, with ${data.inventory_remaining_after_lead_time} reserve units remaining above the safety threshold.`;
  }
}

// ---------------- QUICK SHORTCUT HELPERS ----------------
window.quickForecast = function(stockCode) {
  switchToSection("forecasting");
  const select = document.getElementById("forecast-product-select");
  if (select) {
    select.value = stockCode;
    updateProductQuickSpecs(stockCode);
  }
  executeDemandForecast();
};

window.quickStockout = function(stockCode) {
  switchToSection("stockout");
  const select = document.getElementById("stockout-product-select");
  if (select) {
    select.value = stockCode;
  }
  document.getElementById("stockout-inventory-input").focus();
};

// ---------------- TOAST NOTIFICATION ----------------
function showToast(message, isError = false) {
  const toast = document.getElementById("toast");
  const msg = document.getElementById("toast-message");
  if (!toast || !msg) return;

  msg.innerText = message;
  toast.className = "toast" + (isError ? " error" : "");
  toast.classList.remove("d-none");

  setTimeout(() => {
    toast.classList.add("d-none");
  }, 4000);
}

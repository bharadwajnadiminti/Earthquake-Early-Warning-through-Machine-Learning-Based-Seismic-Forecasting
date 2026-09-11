(function () {
  "use strict";

  const initData = JSON.parse(document.getElementById("init-data").textContent);

  const map = L.map("map", { worldCopyJump: true }).setView([15, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 10,
  }).addTo(map);

  const cellLayer = L.layerGroup().addTo(map);
  const quakeLayer = L.layerGroup().addTo(map);

  let forecastData = null;
  let currentModelKey = initData.bestModelKey;
  let threshold = 0.5;

  const modelSelect = document.getElementById("model-select");
  const thresholdSlider = document.getElementById("threshold-slider");
  const thresholdValue = document.getElementById("threshold-value");
  const toggleRecent = document.getElementById("toggle-recent");
  const refreshBtn = document.getElementById("refresh-btn");
  const alertBanner = document.getElementById("alert-banner");

  function riskColor(p) {
    // green -> yellow -> red
    if (p < 0.33) return "#2ecc71";
    if (p < 0.66) return "#f4c430";
    return "#e74c3c";
  }

  function fmtPct(p) {
    return (p * 100).toFixed(1) + "%";
  }

  function probFor(cell, modelKey) {
    return cell["proba_" + modelKey];
  }

  function renderForecastInfo() {
    const tbody = document.querySelector("#forecast-info tbody");
    tbody.innerHTML = "";
    const rows = [
      ["Forecast as of", forecastData.asof_date],
      ["Horizon end", forecastData.horizon_end_date + " (" + forecastData.horizon_days + " days out)"],
      ["Magnitude threshold", "M " + forecastData.mag_threshold + "+"],
      ["Active cells modeled", forecastData.n_cells],
      ["Model shown", forecastData.model_names[currentModelKey]],
    ];
    for (const [k, v] of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = "<td>" + k + "</td><td>" + v + "</td>";
      tbody.appendChild(tr);
    }
  }

  function renderMapCells() {
    cellLayer.clearLayers();
    for (const cell of forecastData.cells) {
      const p = probFor(cell, currentModelKey);
      if (p === null || p === undefined) continue;
      const marker = L.circleMarker([cell.cell_lat, cell.cell_lon], {
        radius: 4 + p * 14,
        color: riskColor(p),
        fillColor: riskColor(p),
        fillOpacity: 0.55,
        weight: 1,
      });

      const modelRows = Object.keys(forecastData.model_names)
        .map((k) => {
          const pv = probFor(cell, k);
          return "<tr><td>" + forecastData.model_names[k] + "</td><td>" +
            (pv === null || pv === undefined ? "n/a" : fmtPct(pv)) + "</td></tr>";
        })
        .join("");

      marker.bindPopup(
        "<b>Cell (" + cell.cell_lat.toFixed(1) + ", " + cell.cell_lon.toFixed(1) + ")</b><br>" +
        "<table>" + modelRows + "</table>" +
        "<table>" +
        "<tr><td>Events, last 7d</td><td>" + cell.rolling_count_7d + "</td></tr>" +
        "<tr><td>Events, last 30d</td><td>" + cell.rolling_count_30d + "</td></tr>" +
        "<tr><td>Events, last 90d</td><td>" + cell.rolling_count_90d + "</td></tr>" +
        "<tr><td>Max mag, last 365d</td><td>" + (cell.rolling_max_mag_365d != null ? cell.rolling_max_mag_365d.toFixed(2) : "n/a") + "</td></tr>" +
        "<tr><td>Days since last event</td><td>" + cell.days_since_last_event + "</td></tr>" +
        "<tr><td>Completeness mag. (Mc)</td><td>" + cell.mc_cell.toFixed(2) + "</td></tr>" +
        "<tr><td>b-value (90d)</td><td>" + (cell.b_value_90d != null ? cell.b_value_90d.toFixed(2) : "n/a") + "</td></tr>" +
        "<tr><td>Historical events</td><td>" + cell.historical_event_count + "</td></tr>" +
        "</table>"
      );
      marker.addTo(cellLayer);
    }
  }

  function renderAlertBanner() {
    const highRisk = forecastData.cells
      .map((c) => ({ cell: c, p: probFor(c, currentModelKey) }))
      .filter((x) => x.p !== null && x.p !== undefined && x.p >= threshold)
      .sort((a, b) => b.p - a.p);

    if (highRisk.length === 0) {
      alertBanner.hidden = true;
      return;
    }
    const top = highRisk.slice(0, 6)
      .map((x) => "(" + x.cell.cell_lat.toFixed(1) + ", " + x.cell.cell_lon.toFixed(1) + ") " + fmtPct(x.p))
      .join(" &middot; ");
    alertBanner.innerHTML =
      "&#9888; Early warning: <b>" + highRisk.length + "</b> cell(s) at or above " +
      fmtPct(threshold) + " probability of M&ge;" + forecastData.mag_threshold +
      "+ by " + forecastData.horizon_end_date + " &mdash; " + top +
      (highRisk.length > 6 ? " &hellip;" : "");
    alertBanner.hidden = false;
  }

  function renderRiskTable() {
    const tbody = document.querySelector("#risk-table tbody");
    tbody.innerHTML = "";
    const ranked = forecastData.cells
      .map((c) => ({ cell: c, p: probFor(c, currentModelKey) }))
      .filter((x) => x.p !== null && x.p !== undefined)
      .sort((a, b) => b.p - a.p)
      .slice(0, 15);

    ranked.forEach((x, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + (i + 1) + "</td>" +
        "<td>" + x.cell.cell_lat.toFixed(1) + ", " + x.cell.cell_lon.toFixed(1) + "</td>" +
        '<td><span class="risk-badge" style="color:' + riskColor(x.p) + '">' + fmtPct(x.p) + "</span></td>" +
        "<td>" + x.cell.rolling_count_7d + "</td>" +
        "<td>" + x.cell.days_since_last_event + "</td>";
      tbody.appendChild(tr);
    });
  }

  function renderQuakes(quakes) {
    quakeLayer.clearLayers();
    for (const q of quakes) {
      const marker = L.circleMarker([q.latitude, q.longitude], {
        radius: 5,
        color: "#111",
        fillColor: "#fff",
        fillOpacity: 0.9,
        weight: 2,
      });
      marker.bindPopup(
        "<b>M" + q.mag.toFixed(1) + "</b> " + (q.place || "") + "<br>" +
        q.time.slice(0, 16).replace("T", " ") + " UTC<br>depth " + q.depth + " km"
      );
      if (toggleRecent.checked) marker.addTo(quakeLayer);
    }
  }

  function renderMetrics(data) {
    const tbody = document.querySelector("#metrics-table tbody");
    tbody.innerHTML = "";
    const order = Object.entries(data.metrics).sort((a, b) => b[1].roc_auc - a[1].roc_auc);
    for (const [name, m] of order) {
      const tr = document.createElement("tr");
      if (name === data.best_model) tr.classList.add("best-model");
      tr.innerHTML =
        "<td>" + name + (name === data.best_model ? " &#9733;" : "") + "</td>" +
        "<td>" + m.roc_auc.toFixed(4) + "</td>" +
        "<td>" + m.pr_auc.toFixed(4) + "</td>" +
        "<td>" + m.f1_at_best_f1_threshold.toFixed(4) + "</td>" +
        "<td>" + m.brier_score.toFixed(4) + "</td>";
      tbody.appendChild(tr);
    }
    document.getElementById("best-model-note").textContent = data.justification;
  }

  function refreshRenderAll() {
    renderForecastInfo();
    renderMapCells();
    renderAlertBanner();
    renderRiskTable();
  }

  async function loadForecast(forceRefresh) {
    refreshBtn.disabled = true;
    refreshBtn.textContent = "Refreshing...";
    try {
      const url = "/api/forecast" + (forceRefresh ? "?refresh=1" : "");
      const resp = await fetch(url);
      forecastData = await resp.json();
      refreshRenderAll();
    } finally {
      refreshBtn.disabled = false;
      refreshBtn.textContent = "Refresh";
    }
  }

  async function loadQuakes() {
    const resp = await fetch("/api/recent-quakes?days=30&min_mag=4.5");
    const data = await resp.json();
    renderQuakes(data.quakes);
  }

  async function loadMetrics() {
    const resp = await fetch("/api/metrics");
    const data = await resp.json();
    renderMetrics(data);
  }

  modelSelect.addEventListener("change", () => {
    currentModelKey = modelSelect.value;
    refreshRenderAll();
  });

  thresholdSlider.addEventListener("input", () => {
    threshold = parseFloat(thresholdSlider.value);
    thresholdValue.textContent = threshold.toFixed(2);
    renderAlertBanner();
  });

  toggleRecent.addEventListener("change", loadQuakes);

  refreshBtn.addEventListener("click", () => loadForecast(true));

  loadForecast(false);
  loadQuakes();
  loadMetrics();
})();

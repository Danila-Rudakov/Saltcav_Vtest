/**
 * SALTCAV LAB — Логика веб-приложения и визуализатора роста каверны.
 */

// Состояние приложения
const state = {
  config: null,
  simData: null,
  playback: {
    isPlaying: false,
    currentStep: 0,
    speed: 1.0,
    loop: true,
    lastFrameTime: 0,
    frameIntervalMs: 80, // базовая задержка между шагами
  },
  view: {
    xIndex: 0,
    isDraggingX: false,
  },
};

// DOM-элементы
const DOM = {
  statusIndicator: document.getElementById("statusIndicator"),
  statusText: document.getElementById("statusText"),
  form: document.getElementById("configForm"),
  btnReset: document.getElementById("btnResetDefaults"),
  presetSelect: document.getElementById("presetSelect"),
  stagesTableBody: document.getElementById("stagesTableBody"),
  btnAddStage: document.getElementById("btnAddStage"),
  btnRun: document.getElementById("btnRunSimulation"),
  btnReport: document.getElementById("btnDownloadReport"),
  errorBanner: document.getElementById("formErrorMessage"),

  // Плеер
  btnPlayPause: document.getElementById("btnPlayPause"),
  btnStepPrev: document.getElementById("btnStepPrev"),
  btnStepNext: document.getElementById("btnStepNext"),
  btnLoop: document.getElementById("btnLoop"),
  speedButtons: document.querySelectorAll(".btn-speed"),
  timeSlider: document.getElementById("timeSlider"),
  currentTimeDisplay: document.getElementById("currentTimeDisplay"),
  totalTimeDisplay: document.getElementById("totalTimeDisplay"),

  // Холсты
  crossSectionCanvas: document.getElementById("crossSectionCanvas"),
  longitudinalCanvas: document.getElementById("longitudinalCanvas"),
  telemetryCanvas: document.getElementById("telemetryCanvas"),

  // Метрики
  valXCoord: document.getElementById("valXCoord"),
  valRoofR: document.getElementById("valRoofR"),
  valFloorR: document.getElementById("valFloorR"),
  valWidth: document.getElementById("valWidth"),
  metricVolume: document.getElementById("metricVolume"),
  metricOutlet: document.getElementById("metricOutlet"),
  metricSatPct: document.getElementById("metricSatPct"),
  metricProduced: document.getElementById("metricProduced"),
  metricFlow: document.getElementById("metricFlow"),
};

// ============================================================================
// Инициализация
// ============================================================================
async function init() {
  bindEvents();
  setupCanvases();
  setupResizeObserver();
  renderFormulas();
  await loadDefaults();
}

function renderFormulas() {
  if (window.renderMathInElement && typeof window.renderMathInElement === "function") {
    try {
      window.renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
        throwOnError: false,
      });
    } catch (e) {
      console.warn("KaTeX render error:", e);
    }
  }
}

function bindEvents() {
  DOM.form.addEventListener("submit", onFormSubmit);
  DOM.btnReport.addEventListener("click", () => {
    if (!state.simData) return;
    const content = buildLabReport(state.simData, state.view.xIndex);
    const url = URL.createObjectURL(new Blob([content], { type: "text/html;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "saltcav-lab-report.html";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  DOM.btnReset.addEventListener("click", () => loadDefaults());
  DOM.btnAddStage.addEventListener("click", () => addStageRow(1.0, 10.0, 0.0));
  DOM.presetSelect.addEventListener("change", onPresetChange);
  DOM.presetSelect.addEventListener("change", syncSetupFields);
  document.getElementById("materialEnabled").addEventListener("change", syncSetupFields);
  syncSetupFields();

  // Плеер
  DOM.btnPlayPause.addEventListener("click", togglePlay);
  DOM.btnStepPrev.addEventListener("click", () => step(-1));
  DOM.btnStepNext.addEventListener("click", () => step(1));
  DOM.btnLoop.addEventListener("click", toggleLoop);

  DOM.speedButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      DOM.speedButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.playback.speed = parseFloat(btn.dataset.speed);
    });
  });

  DOM.timeSlider.addEventListener("input", (e) => {
    if (state.playback.isPlaying) pause();
    setStep(parseInt(e.target.value, 10));
  });

  // Клик / перетаскивание на продольном холсте для выбора визира X
  DOM.longitudinalCanvas.addEventListener("mousedown", onLongitudinalClick);
  DOM.longitudinalCanvas.addEventListener("mousemove", (e) => {
    if (state.view.isDraggingX) onLongitudinalClick(e);
  });
  window.addEventListener("mouseup", () => {
    state.view.isDraggingX = false;
  });

  // Горячие клавиши
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.code === "Space") {
      e.preventDefault();
      togglePlay();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      step(-1);
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      step(1);
    } else if (e.code === "Home") {
      e.preventDefault();
      if (state.playback.isPlaying) pause();
      setStep(0);
    } else if (e.code === "End") {
      e.preventDefault();
      if (state.playback.isPlaying) pause();
      if (state.simData) setStep(state.simData.n_steps - 1);
    }
  });

  window.addEventListener("resize", () => {
    setupCanvases();
    renderAll();
  });
}

function setupResizeObserver() {
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(() => {
      setupCanvases();
      renderAll();
    });
    document.querySelectorAll(".canvas-wrapper, .telemetry-canvas-wrapper").forEach((el) => {
      ro.observe(el);
    });
  }
}

function onLongitudinalClick(e) {
  if (!state.simData) return;
  state.view.isDraggingX = true;
  const rect = DOM.longitudinalCanvas.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const padLeft = 40;
  const padRight = 25;
  const usableWidth = rect.width - padLeft - padRight;
  const frac = Math.min(Math.max((clickX - padLeft) / usableWidth, 0), 1);
  const nx = state.simData.x_mm.length;
  const idx = Math.min(Math.round(frac * (nx - 1)), nx - 1);
  setXIndex(idx);
}

// ============================================================================
// Загрузка дефолтов и запуск расчета
// ============================================================================
async function loadDefaults() {
  setStatus("calculating", "Загрузка параметров...");
  try {
    const res = await fetch("/api/defaults");
    const json = await res.json();
    if (json.status === "ok") {
      state.config = json.config;
      populateForm(json.config);
      setStatus("ready", "Параметры загружены");
      renderFormulas();
      // Автоматический начальный расчёт для мгновенного отображения каверны
      await runSimulation();
    }
  } catch (err) {
    console.error("Ошибка загрузки дефолтов:", err);
    setStatus("error", "Ошибка соединения с сервером");
  }
}

function populateForm(cfg) {
  document.getElementById("lengthMm").value = cfg.length_mm ?? 200;
  document.getElementById("diameterMm").value = cfg.diameter_mm ?? 2.5;
  document.getElementById("maxRadiusMm").value = cfg.max_radius_mm ?? 40;
  document.getElementById("temperatureC").value = cfg.temperature_c ?? 20;
  document.getElementById("saltDensity").value = cfg.salt_density_kg_m3 ?? 2160;
  document.getElementById("roofMult").value = cfg.roof_mult ?? 1.0;
  document.getElementById("wallMult").value = cfg.wall_mult ?? 1.0;
  document.getElementById("naturalConvection").checked = cfg.natural_convection !== false;

  DOM.stagesTableBody.innerHTML = "";
  const stages = cfg.stages || [
    { duration_h: 1.0, flow_ml_min: 5.0, inlet_saturation: 0.0 },
    { duration_h: 1.0, flow_ml_min: 20.0, inlet_saturation: 0.0 },
  ];
  stages.forEach((st) => addStageRow(st.duration_h, st.flow_ml_min, st.inlet_saturation));
}

function onPresetChange(e) {
  const val = e.target.value;
  if (val === "window") {
    // В пресете window половина сечения
    document.getElementById("lengthMm").value = 150;
    document.getElementById("diameterMm").value = 2.0;
  } else {
    document.getElementById("lengthMm").value = 200;
    document.getElementById("diameterMm").value = 2.5;
  }
}

function addStageRow(duration = 1.0, flow = 10.0, sat = 0.0) {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input type="number" step="any" min="0.001" max="1000" value="${duration}" class="stage-duration" required></td>
    <td><input type="number" step="any" min="0.001" max="5000" value="${flow}" class="stage-flow" required></td>
    <td><input type="number" step="any" min="0" max="1" value="${sat}" class="stage-sat" required></td>
    <td><button type="button" class="btn-remove-stage" title="Удалить ступень">×</button></td>
  `;
  tr.querySelector(".btn-remove-stage").addEventListener("click", () => {
    if (DOM.stagesTableBody.children.length > 1) {
      tr.remove();
    }
  });
  DOM.stagesTableBody.appendChild(tr);
}

function collectFormData() {
  const stages = [];
  const rows = DOM.stagesTableBody.querySelectorAll("tr");
  rows.forEach((row) => {
    stages.push({
      duration_h: parseFloat(row.querySelector(".stage-duration").value),
      flow_ml_min: parseFloat(row.querySelector(".stage-flow").value),
      inlet_saturation: parseFloat(row.querySelector(".stage-sat").value),
    });
  });

  const setupType = DOM.presetSelect.value;
  const maxR = document.getElementById("maxRadiusMm").value;

  return {
    name: "lab-channel-web",
    setup: setupType,
    model: "section",
    natural_convection: document.getElementById("naturalConvection").checked,
    length_mm: parseFloat(document.getElementById("lengthMm").value),
    diameter_mm: parseFloat(document.getElementById("diameterMm").value),
    max_radius_mm: maxR ? parseFloat(maxR) : null,
    temperature_c: parseFloat(document.getElementById("temperatureC").value),
    salt_density_kg_m3: parseFloat(document.getElementById("saltDensity").value),
    roof_mult: parseFloat(document.getElementById("roofMult").value),
    wall_mult: parseFloat(document.getElementById("wallMult").value),
    sh_mult: parseFloat(document.getElementById("shMult").value),
    n_cells: parseInt(document.getElementById("nCells").value, 10),
    n_theta: parseInt(document.getElementById("nTheta").value, 10),
    output_every_min: parseFloat(document.getElementById("outputEveryMin").value),
    stefan_condition: document.getElementById("stefanCondition").checked,
    material: collectMaterial(setupType),
    stages: stages,
  };
}

// Полный контур 0…2π (setup: full): без зеркала, дно — узел θ = π (середина), а не последний.
function isClosedContour(theta) {
  return theta[theta.length - 1] > Math.PI * 1.5;
}
function floorIndex(theta) {
  return isClosedContour(theta) ? (theta.length - 1) / 2 : theta.length - 1;
}

// Материал задаётся только для полного контура; иначе null (однородный).
function collectMaterial(setupType) {
  if (setupType !== "full" || !document.getElementById("materialEnabled").checked) return null;
  const offset = document.getElementById("sheetOffsetMm").value;
  return {
    seed: parseInt(document.getElementById("grainSeed").value, 10),
    grain_contrast: parseFloat(document.getElementById("grainContrast").value),
    boundary_factor: parseFloat(document.getElementById("boundaryFactor").value),
    sheet_offset_mm: offset === "" ? null : parseFloat(offset),
  };
}

// Поля, зависящие от режима: материал и кратность n_theta — только для full.
function syncSetupFields() {
  const full = DOM.presetSelect.value === "full";
  document.getElementById("materialSection").hidden = !full;
  document.getElementById("materialFields").hidden =
    !(full && document.getElementById("materialEnabled").checked);
  document.getElementById("nThetaHint").textContent = full ? "(на весь контур, кратно 4)" : "(на половину)";
  const nt = document.getElementById("nTheta");
  nt.step = full ? 4 : 1;
  if (full && nt.value % 4 !== 0) nt.value = Math.max(4, Math.round(nt.value / 4) * 4);
}

async function onFormSubmit(e) {
  e.preventDefault();
  await runSimulation();
}

async function runSimulation() {
  if (state.playback.isPlaying) pause();
  DOM.errorBanner.classList.add("hidden");
  DOM.btnRun.disabled = true;
  setStatus("calculating", "Идёт расчёт каверны...");

  const payload = collectFormData();

  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const json = await res.json();
    if (!res.ok || json.status !== "ok") {
      throw new Error(json.error || "Ошибка расчёта");
    }

    state.simData = json.data;
    DOM.btnReport.disabled = false;
    onSimulationSuccess();
  } catch (err) {
    console.error("Simulation error:", err);
    DOM.errorBanner.textContent = err.message;
    DOM.errorBanner.classList.remove("hidden");
    setStatus("error", "Ошибка расчёта");
  } finally {
    DOM.btnRun.disabled = false;
  }
}

function onSimulationSuccess() {
  const d = state.simData;
  const nSteps = d.n_steps;
  setStatus("ready", `Готов (${d.metrics.calc_duration_s} с, ${nSteps} шагов)`);

  // Обновляем слайдер времени
  DOM.timeSlider.max = nSteps - 1;
  DOM.timeSlider.value = state.playback.currentStep = 0;
  DOM.totalTimeDisplay.textContent = `/ ${d.formatted_times[nSteps - 1]}`;

  const nx = d.x_mm.length;
  const midIdx = Math.floor(nx / 2);
  setXIndex(midIdx);

  renderAll();

  // Автозапуск анимации после успешного расчета
  play();
}

function setStatus(type, text) {
  DOM.statusIndicator.className = `status-indicator ${type}`;
  DOM.statusText.textContent = text;
}

// ============================================================================
// Плеер анимации
// ============================================================================
function togglePlay() {
  if (state.playback.isPlaying) {
    pause();
  } else {
    play();
  }
}

function play() {
  if (!state.simData) return;
  state.playback.isPlaying = true;
  DOM.btnPlayPause.textContent = "❚❚";
  DOM.btnPlayPause.title = "Пауза (Пробел)";
  state.playback.lastFrameTime = performance.now();
  requestAnimationFrame(playbackLoop);
}

function pause() {
  state.playback.isPlaying = false;
  DOM.btnPlayPause.textContent = "▶";
  DOM.btnPlayPause.title = "Пуск (Пробел)";
}

function toggleLoop() {
  state.playback.loop = !state.playback.loop;
  DOM.btnLoop.classList.toggle("active", state.playback.loop);
}

function step(delta) {
  if (!state.simData) return;
  if (state.playback.isPlaying) pause();
  const nSteps = state.simData.n_steps;
  let next = state.playback.currentStep + delta;
  if (next < 0) next = 0;
  if (next >= nSteps) next = nSteps - 1;
  setStep(next);
}

function setStep(idx) {
  state.playback.currentStep = idx;
  DOM.timeSlider.value = idx;
  updateTimeAndMetrics(idx);
  renderAll();
}

function setXIndex(idx) {
  state.view.xIndex = idx;
  if (state.simData) {
    const xMm = state.simData.x_mm[idx];
    if (DOM.valXCoord) {
      DOM.valXCoord.textContent = `${xMm.toFixed(1)} мм`;
    }
  }
  renderAll();
}

function playbackLoop(timestamp) {
  if (!state.playback.isPlaying) return;

  const effectiveInterval = state.playback.frameIntervalMs / state.playback.speed;
  const elapsed = timestamp - state.playback.lastFrameTime;

  if (elapsed >= effectiveInterval) {
    state.playback.lastFrameTime = timestamp;
    const nSteps = state.simData ? state.simData.n_steps : 0;
    let nextStep = state.playback.currentStep + 1;

    if (nextStep >= nSteps) {
      if (state.playback.loop) {
        nextStep = 0;
      } else {
        pause();
        return;
      }
    }
    setStep(nextStep);
  }

  requestAnimationFrame(playbackLoop);
}

function updateTimeAndMetrics(stepIdx) {
  if (!state.simData) return;
  const d = state.simData;
  DOM.currentTimeDisplay.textContent = d.formatted_times[stepIdx];

  const m = d.metrics;
  DOM.metricVolume.textContent = m.volume_cm3[stepIdx].toFixed(2);
  const outletC = m.outlet_concentration_g_l[stepIdx];
  DOM.metricOutlet.textContent = outletC.toFixed(1);
  const satPct = Math.round((outletC / m.c_sat_g_l) * 100);
  DOM.metricSatPct.textContent = satPct;
  DOM.metricProduced.textContent = m.dissolved_g[stepIdx].toFixed(1);
  DOM.metricFlow.textContent = m.flow_ml_min[stepIdx].toFixed(1);

  // Статистика конкретного сечения
  const rSlice = d.radius_mm[stepIdx][state.view.xIndex];
  const rTop = rSlice[0]; // theta = 0
  const rBot = rSlice[floorIndex(d.theta_rad)]; // theta = pi
  // Максимальная ширина (вдоль горизонтали)
  let maxW = 0;
  if (isClosedContour(d.theta_rad)) {
    // полный контур: ширина = размах по горизонтали (с учётом несимметрии)
    const ys = Array.from(rSlice, (rv, j) => rv * Math.sin(d.theta_rad[j]));
    maxW = Math.max(...ys) - Math.min(...ys);
  } else {
    for (let j = 0; j < d.theta_rad.length; j++) {
      const w = 2.0 * rSlice[j] * Math.sin(d.theta_rad[j]);
      if (w > maxW) maxW = w;
    }
  }
  DOM.valRoofR.textContent = `${rTop.toFixed(2)} мм`;
  DOM.valFloorR.textContent = `${rBot.toFixed(2)} мм`;
  DOM.valWidth.textContent = `${maxW.toFixed(2)} мм`;
}

// ============================================================================
// Рендеринг холстов (Canvas HiDPI)
// ============================================================================
function setupCanvases() {
  [DOM.crossSectionCanvas, DOM.longitudinalCanvas, DOM.telemetryCanvas].forEach((canvas) => {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(Math.floor(rect.width), 10);
    const h = Math.max(Math.floor(rect.height), 10);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.resetTransform();
    ctx.scale(dpr, dpr);
  });
}

function renderAll() {
  if (!state.simData) {
    drawEmptyState(DOM.crossSectionCanvas, "Ожидание данных расчета...");
    drawEmptyState(DOM.longitudinalCanvas, "Ожидание данных расчета...");
    drawEmptyState(DOM.telemetryCanvas, "Динамика параметров...");
    return;
  }

  const step = state.playback.currentStep;
  const xIdx = state.view.xIndex;

  renderCrossSection(DOM.crossSectionCanvas, step, xIdx);
  renderLongitudinal(DOM.longitudinalCanvas, step, xIdx);
  renderTelemetry(DOM.telemetryCanvas, step);
}

function drawEmptyState(canvas, msg) {
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#A89F9F";
  ctx.font = "12px -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(msg, rect.width / 2, rect.height / 2);
}

/**
 * Отрисовка поперечного сечения r(theta)
 */
function renderCrossSection(canvas, stepIdx, xIdx) {
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const W = rect.width;
  const H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const d = state.simData;
  const rSlice = d.radius_mm[stepIdx][xIdx];
  const theta = d.theta_rad;
  const initialR = d.initial_diameter_mm / 2.0;

  // Масштаб (мм -> пиксели). Базируем на максимальном радиусе в опыте
  const maxR = d.max_radius_mm || Math.max(...d.radius_mm[d.n_steps - 1].flat()) * 1.2;
  const margin = 28;
  const scale = (Math.min(W, H) / 2 - margin) / maxR;

  const cx = W / 2;
  const cy = H / 2;

  // 1. Координатная сетка
  ctx.save();
  ctx.strokeStyle = "#E2E9EF";
  ctx.lineWidth = 1;

  // Окружности сетки
  const ringStep = maxR <= 10 ? 2 : maxR <= 30 ? 5 : 10;
  for (let r = ringStep; r <= maxR; r += ringStep) {
    ctx.beginPath();
    ctx.arc(cx, cy, r * scale, 0, 2 * Math.PI);
    ctx.stroke();

    ctx.fillStyle = "#8494A0";
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`${r} мм`, cx + r * scale + 3, cy - 2);
  }

  // Осевые линии
  ctx.strokeStyle = "#C6D4E1";
  ctx.beginPath();
  ctx.moveTo(cx, margin / 2);
  ctx.lineTo(cx, H - margin / 2);
  ctx.moveTo(margin / 2, cy);
  ctx.lineTo(W - margin / 2, cy);
  ctx.stroke();

  // Подписи верха и низа
  ctx.fillStyle = "#2B608A";
  ctx.font = "10px -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Свод", cx, margin / 2 + 8);
  ctx.fillStyle = "#516270";
  ctx.fillText("Дно", cx, H - margin / 2 + 2);

  // 2. Исходный просверленный канал (пунктир)
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "#7F93A3";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(cx, cy, initialR * scale, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.setLineDash([]);

  // 3. Текущая граница каверны r(theta)
  const ptsRight = [];
  const ptsLeft = [];
  const closed = isClosedContour(theta);

  for (let j = 0; j < theta.length; j++) {
    const th = theta[j];
    const r = rSlice[j] * scale;
    const px = r * Math.sin(th);
    const py = -r * Math.cos(th); // -cos дает вверх для th = 0

    ptsRight.push({ x: cx + px, y: cy + py });
    if (!closed && th > 0 && th < Math.PI) {
      ptsLeft.unshift({ x: cx - px, y: cy + py }); // зеркальная левая половина
    }
  }

  const allPts = [...ptsRight, ...ptsLeft];

  // Заливка каверны научным бледно-синим цветом
  ctx.beginPath();
  ctx.moveTo(allPts[0].x, allPts[0].y);
  for (let i = 1; i < allPts.length; i++) {
    ctx.lineTo(allPts[i].x, allPts[i].y);
  }
  ctx.closePath();

  ctx.fillStyle = "rgba(43, 96, 138, 0.16)";
  ctx.fill();

  ctx.strokeStyle = "#2B608A";
  ctx.lineWidth = 2.2;
  ctx.stroke();

  // Точки-маркеры на своде и дне
  ctx.fillStyle = "#2B608A";
  ctx.beginPath();
  ctx.arc(ptsRight[0].x, ptsRight[0].y, 3.5, 0, 2 * Math.PI);
  const fl = ptsRight[closed ? floorIndex(theta) : ptsRight.length - 1];
  ctx.arc(fl.x, fl.y, 3.5, 0, 2 * Math.PI);
  ctx.fill();

  ctx.restore();
}

/**
 * Отрисовка продольного профиля канала (вид сбоку)
 */
function renderLongitudinal(canvas, stepIdx, currentXIdx) {
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const W = rect.width;
  const H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const d = state.simData;
  const nx = d.x_mm.length;
  const stepData = d.radius_mm[stepIdx];
  const initialR = d.initial_diameter_mm / 2.0;

  const padLeft = 40;
  const padRight = 25;
  const padY = 25;

  const usableW = W - padLeft - padRight;
  const usableH = H - 2 * padY;

  const totalLength = d.x_mm[nx - 1];
  const maxR = d.max_radius_mm || Math.max(...d.radius_mm[d.n_steps - 1].flat()) * 1.2;

  const scaleX = usableW / totalLength;
  const scaleZ = (usableH / 2) / maxR;
  const midY = H / 2;

  ctx.save();

  // 1. Оси и сетка
  ctx.strokeStyle = "#E2E9EF";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padLeft, midY);
  ctx.lineTo(padLeft + usableW, midY);
  ctx.stroke();

  // Засечки по оси X
  const xStep = totalLength <= 100 ? 25 : 50;
  ctx.fillStyle = "#8494A0";
  ctx.font = "9px monospace";
  ctx.textAlign = "center";
  for (let xm = 0; xm <= totalLength; xm += xStep) {
    const px = padLeft + xm * scaleX;
    ctx.beginPath();
    ctx.moveTo(px, midY - 3);
    ctx.lineTo(px, midY + 3);
    ctx.stroke();
    ctx.fillText(`${xm}`, px, midY + 14);
  }
  ctx.fillText("x, мм", padLeft + usableW - 8, midY - 5);

  // 2. Исходный канал (горизонтальный пунктир)
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "#7F93A3";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padLeft, midY - initialR * scaleZ);
  ctx.lineTo(padLeft + usableW, midY - initialR * scaleZ);
  ctx.moveTo(padLeft, midY + initialR * scaleZ);
  ctx.lineTo(padLeft + usableW, midY + initialR * scaleZ);
  ctx.stroke();
  ctx.setLineDash([]);

  // 3. Контуры каверны: свод z_top(x) и дно z_bot(x)
  const roofPts = [];
  const floorPts = [];

  for (let i = 0; i < nx; i++) {
    const px = padLeft + d.x_mm[i] * scaleX;
    const rRoof = stepData[i][0]; // theta = 0
    const rFloor = stepData[i][floorIndex(d.theta_rad)]; // theta = pi

    roofPts.push({ x: px, y: midY - rRoof * scaleZ });
    floorPts.push({ x: px, y: midY + rFloor * scaleZ });
  }

  // Заливка каверны
  ctx.beginPath();
  ctx.moveTo(roofPts[0].x, roofPts[0].y);
  for (let i = 1; i < roofPts.length; i++) ctx.lineTo(roofPts[i].x, roofPts[i].y);
  for (let i = floorPts.length - 1; i >= 0; i--) ctx.lineTo(floorPts[i].x, floorPts[i].y);
  ctx.closePath();

  ctx.fillStyle = "rgba(43, 96, 138, 0.16)";
  ctx.fill();

  // Границы
  ctx.strokeStyle = "#2B608A";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.moveTo(roofPts[0].x, roofPts[0].y);
  for (let i = 1; i < roofPts.length; i++) ctx.lineTo(roofPts[i].x, roofPts[i].y);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(floorPts[0].x, floorPts[0].y);
  for (let i = 1; i < floorPts.length; i++) ctx.lineTo(floorPts[i].x, floorPts[i].y);
  ctx.stroke();

  // 4. Вертикальный маркер (визир) выбранного сечения X
  const markerX = padLeft + d.x_mm[currentXIdx] * scaleX;
  ctx.strokeStyle = "#2B608A";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(markerX, padY - 4);
  ctx.lineTo(markerX, H - padY + 4);
  ctx.stroke();

  // Флажок визира
  ctx.fillStyle = "#2B608A";
  ctx.beginPath();
  ctx.moveTo(markerX - 4, padY - 4);
  ctx.lineTo(markerX + 4, padY - 4);
  ctx.lineTo(markerX, padY);
  ctx.closePath();
  ctx.fill();

  ctx.restore();
}

/**
 * Отрисовка графика динамики объема V(t)
 */
function renderTelemetry(canvas, currentStepIdx) {
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const W = rect.width;
  const H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const d = state.simData;
  const nSteps = d.n_steps;
  const vols = d.metrics.volume_cm3;
  const maxV = Math.max(...vols) * 1.1 || 1.0;

  const padX = 35;
  const padY = 10;
  const usableW = W - 2 * padX;
  const usableH = H - 2 * padY;

  ctx.save();

  // Фоновая сетка
  ctx.strokeStyle = "#E2E9EF";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padX, padY);
  ctx.lineTo(padX, H - padY);
  ctx.lineTo(W - padX, H - padY);
  ctx.stroke();

  // Кривая объема V(t)
  ctx.strokeStyle = "#2B608A";
  ctx.lineWidth = 1.8;
  ctx.beginPath();

  for (let i = 0; i < nSteps; i++) {
    const px = padX + (i / (nSteps - 1)) * usableW;
    const py = H - padY - (vols[i] / maxV) * usableH;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Маркер текущего времени
  const curX = padX + (currentStepIdx / (nSteps - 1)) * usableW;
  const curY = H - padY - (vols[currentStepIdx] / maxV) * usableH;

  ctx.strokeStyle = "rgba(43, 96, 138, 0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(curX, padY);
  ctx.lineTo(curX, H - padY);
  ctx.stroke();

  ctx.fillStyle = "#2B608A";
  ctx.beginPath();
  ctx.arc(curX, curY, 3.5, 0, 2 * Math.PI);
  ctx.fill();

  // Подписи
  ctx.fillStyle = "#8494A0";
  ctx.font = "9px monospace";
  ctx.textAlign = "left";
  ctx.fillText(`V = ${vols[currentStepIdx].toFixed(2)} см³`, curX + 6, Math.max(curY - 4, padY + 8));
  ctx.fillText("0", padX - 10, H - padY + 2);
  ctx.fillText(`${maxV.toFixed(1)} см³`, padX - 30, padY + 6);

  ctx.restore();
}

// Запуск при загрузке страницы
window.addEventListener("DOMContentLoaded", init);

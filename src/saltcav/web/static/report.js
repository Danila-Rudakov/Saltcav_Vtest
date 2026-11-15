/* Автономный отчёт по данным последнего успешного расчёта веб-приложения. */
function buildLabReport(data, xIndex) {
  const cfg = data.config;
  const last = data.n_steps - 1;
  const x = Math.max(0, Math.min(xIndex, data.x_mm.length - 1));
  const m = data.metrics;
  const escape = value => String(value ?? "не задано").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[c]);
  const number = value => Number.isFinite(value) ? Number(value.toPrecision(6)).toString() : "нет данных";
  const row = (label, value, unit = "") => `<tr><th>${escape(label)}</th><td>${escape(value)} ${escape(unit)}</td></tr>`;
  const table = rows => `<table>${rows.join("")}</table>`;
  const configRows = [
    ["Название", cfg.name], ["Установка", cfg.setup === "window" ? "Полуканал у окна" : "Канал в блоке"],
    ["Модель", cfg.model], ["Длина канала", cfg.length_mm, "мм"],
    ["Начальный диаметр", cfg.diameter_mm, "мм"], ["Предел радиуса блока", cfg.max_radius_mm, "мм"],
    ["Температура на поверхности", cfg.temperature_c, "°C"], ["Температура на глубине", cfg.temperature_c + (cfg.geothermal_gradient_c_per_100m || 0) * cfg.depth_m / 100, "°C"],
    ["Плотность соли", cfg.salt_density_kg_m3, "кг/м³"], ["Глубина", cfg.depth_m, "м"],
    ["Диаметр трубок", cfg.tube_diameter_mm, "мм"], ["Геотермический градиент", cfg.geothermal_gradient_c_per_100m, "°C/100 м"],
    ["Плотность пород", cfg.rock_density_kg_m3, "кг/м³"], ["Свободная конвекция и расслоение", cfg.natural_convection ? "включены" : "выключены"],
    ["Множитель массоотдачи", cfg.sh_mult], ["Множитель потолка", cfg.roof_mult], ["Множитель стенок", cfg.wall_mult],
    ["Порог расслоения Gr/Re²", cfg.ri_crit], ["Константа реакции", cfg.k_surface_m_s, "м/с"],
    ["Теплопроводность соли", cfg.salt_conductivity_w_mk, "Вт/(м·К)"], ["Ячеек по длине", cfg.n_cells],
    ["Секторов на половину сечения", cfg.n_theta], ["Шаг записи", cfg.output_every_min, "мин"], ["CFL", cfg.cfl]
  ];
  const resultRows = [
    ["Расчётное время", number(data.times[last] / 3600), "ч"],
    ["Запланированная длительность", number(cfg.stages.reduce((sum, s) => sum + s.duration_h, 0)), "ч"],
    ["Объём каверны", number(m.volume_cm3[last]), "см³"], ["Концентрация на выходе", number(m.outlet_concentration_g_l[last]), "г/л"],
    ["Концентрация насыщения", number(m.c_sat_g_l), "г/л"], ["Вынесено соли", number(m.dissolved_g[last]), "г"],
    ["Расход на последнем записанном шаге", number(m.flow_ml_min[last]), "мл/мин"],
    ["Число записанных шагов", data.n_steps], ["Время расчёта", m.calc_duration_s, "с"]
  ];
  // Каждая кривая масштабируется отдельно, ось времени отражает реальные моменты записи.
  function chart(title, xs, ys, xLabel, yLabel) {
    const w = 620, h = 210, l = 64, r = 20, top = 18, bottom = 42;
    const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(0, ...ys), ymax = Math.max(...ys);
    const px = v => l + (v - xmin) / (xmax - xmin || 1) * (w - l - r);
    const py = v => h - bottom - (v - ymin) / (ymax - ymin || 1) * (h - top - bottom);
    const points = xs.map((v, i) => `${px(v).toFixed(2)},${py(ys[i]).toFixed(2)}`).join(" ");
    return `<figure><figcaption>${escape(title)}</figcaption><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${escape(title)}"><path d="M${l} ${top}V${h-bottom}H${w-r}" fill="none" stroke="#566"/><polyline points="${points}" fill="none" stroke="#2b608a" stroke-width="2"/><text x="${l}" y="${h-20}">${escape(number(xmin))}</text><text x="${w-r}" y="${h-20}" text-anchor="end">${escape(number(xmax))}</text><text x="${l-5}" y="${top+5}" text-anchor="end">${escape(number(ymax))}</text><text x="${l-5}" y="${h-bottom}" text-anchor="end">${escape(number(ymin))}</text><text x="${w/2}" y="${h-2}" text-anchor="middle">${escape(xLabel)}</text><text x="${l+5}" y="${top+12}">${escape(yLabel)}</text></svg></figure>`;
  }
  const hours = data.times.map(t => t / 3600);
  const radii = data.radius_mm[last][x];
  const theta = data.theta_rad;
  const coords = theta.map((t, i) => [radii[i] * Math.sin(t), -radii[i] * Math.cos(t)]);
  const outline = data.halves === 2 ? coords.concat(coords.slice(1, -1).reverse().map(([a, b]) => [-a, b])) : coords;
  const maxRadius = Math.max(...radii, data.initial_diameter_mm / 2) * 1.15 || 1;
  const points = outline.map(([a, b]) => `${(150 + a / maxRadius * 125).toFixed(2)},${(150 + b / maxRadius * 125).toFixed(2)}`).join(" ");
  const geometry = `<figure><figcaption>Итоговое сечение при x = ${escape(number(data.x_mm[x]))} мм (${data.halves === 2 ? "полный канал" : "полуканал у окна"})</figcaption><svg viewBox="0 0 300 300" role="img" aria-label="Поперечное сечение"><line x1="150" y1="15" x2="150" y2="285" stroke="#ccd"/><line x1="15" y1="150" x2="285" y2="150" stroke="#ccd"/><polygon points="${points}" fill="#d6e3ed" stroke="#2b608a" stroke-width="2"/><text x="150" y="13" text-anchor="middle">Свод; масштаб ±${escape(number(maxRadius))} мм</text><text x="150" y="298" text-anchor="middle">Дно</text></svg></figure>`;
  const roof = data.radius_mm[last].map(slice => slice[0]);
  const floor = data.radius_mm[last].map(slice => -slice[slice.length - 1]);
  const profile = chart("Продольный профиль: радиус свода", data.x_mm, roof, "x, мм", "радиус, мм") + chart("Продольный профиль: отметка дна", data.x_mm, floor, "x, мм", "отметка, мм");
  const stages = cfg.stages.map((s, i) => `<tr><td>${i+1}</td><td>${escape(s.duration_h)}</td><td>${escape(s.flow_ml_min)}</td><td>${escape(s.inlet_saturation)}</td></tr>`).join("");
  const configJson = escape(JSON.stringify(cfg, null, 2));
  return `<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Отчёт SALTCAV LAB</title><style>body{font:15px Arial,sans-serif;color:#243542;max-width:1000px;margin:32px auto;padding:0 20px}h1,h2{color:#245879}table{border-collapse:collapse;width:100%;margin:10px 0 22px}th,td{border:1px solid #cbd7df;padding:7px;text-align:left}th{background:#edf3f7}figure{margin:12px 0;break-inside:avoid}figcaption{font-weight:bold}svg{width:100%;max-height:300px}svg text{font:11px Arial;fill:#354958}.plots{display:grid;grid-template-columns:1fr 1fr;gap:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:12px}p{line-height:1.5}@media print{body{margin:0}.no-print{display:none}}</style></head><body><h1>Отчёт SALTCAV LAB</h1><p>Сформирован: ${escape(new Date().toLocaleString("ru-RU"))}. Показаны результаты последнего успешного расчёта, итоговый записанный шаг. Для PDF используйте печать браузера.</p><button class="no-print" onclick="window.print()">Печать / сохранить в PDF</button><h2>Параметры опыта</h2>${table(configRows.map(a => row(...a)))}<h2>Режим насоса</h2><table><tr><th>Ступень</th><th>Длительность, ч</th><th>Расход, мл/мин</th><th>Насыщение на входе, доля</th></tr>${stages}</table><h2>Итоги расчёта</h2>${table(resultRows.map(a => row(...a)))}<p>Раннее завершение может быть связано с достижением границы блока; точная причина сервером не передаётся. Значения и графики построены по данным, округлённым для визуализации. Вынесенная соль — показатель модели, а не экспериментальное измерение.</p><h2>Геометрия на итоговом шаге</h2><div class="plots">${geometry}${profile}</div><h2>Динамика</h2><div class="plots">${chart("Объём каверны", hours, m.volume_cm3, "время, ч", "см³")}${chart("Концентрация на выходе", hours, m.outlet_concentration_g_l, "время, ч", "г/л")}${chart("Вынесено соли", hours, m.dissolved_g, "время, ч", "г")}${chart("Расход насоса", hours, m.flow_ml_min, "время, ч", "мл/мин")}</div><h2>Ограничения и воспроизводимость</h2><p>Веб-интерфейс рассчитывает модель сечения. Давление, гидравлика трубок и ползучесть в данном отчёте не рассчитываются. Результаты зависят от параметров модели и требуют проверки по данным опыта.</p><h3>Конфигурация расчёта (JSON)</h3><pre>${configJson}</pre></body></html>`;
}

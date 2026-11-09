# Лабораторный канал в блоке NaCl

Геометрия и модели лабораторной установки: **вместо** вертикальной скважинной каверны
исходной saltcav (автор — Sergei Kizred; [разрешение](РАЗРЕШЕНИЕ_АВТОРА.md)) считается
горизонтальный канал в монолитном блоке соли. Физика Сергея (`leaching/dissolution.py`,
`leaching/brine_column.py`, `core/brine.py`, закон ползучести) осталась на месте — модели
построены на ней. Математическая модель — [МАТЕМАТИЧЕСКАЯ_МОДЕЛЬ.md](МАТЕМАТИЧЕСКАЯ_МОДЕЛЬ.md).
Что заменено, что осталось и что ждёт адаптации — [../rewrite-map.md](../rewrite-map.md).

Два варианта установки:

* `setup: channel` — канал в целом блоке;
* `setup: window` — полуканал у вертикального окна из оргстекла (на эпоксидке), видна половина каверны.

Установку можно «опустить» на глубину до 2800 м (`depth_m`): вода подаётся насосом с поверхности
по трубке, рассол поднимается по второй трубке; считаются влияние давления (и температуры пород,
если задан градиент), давление насоса, запаздывание рассола в трубке и ползучесть соли.

Две модели:

* `model: section` (основная) — сечение r(θ, x) со свободной конвекцией и расслоением рассола;
* `model: axisym` — простая модель с круглым сечением r(x), без гравитации (для сверки).

## Модули

| Что | Где |
| --- | --- |
| конфигурация опыта (`LabConfig`, `LabStage`, `load_lab_config`) | `src/saltcav/core/lab_config.py` |
| вязкость рассола, сжимаемость, влияние давления (дополнено) | `src/saltcav/core/brine.py` |
| геометрия: круглый канал r(x) | `src/saltcav/geometry/channel.py` |
| геометрия: сечение r(θ, x), объёмы клиньев, движение стенки | `src/saltcav/geometry/section.py` |
| простая модель (`simulate_lab`, `LabResult`) | `src/saltcav/leaching/axisym.py` |
| модель сечения (`simulate_section`, `SectionResult`) | `src/saltcav/leaching/section.py` |
| глубина: трубки, давление насоса, ползучесть (`hydraulics`, `creep_estimate`, `depth_study`) | `src/saltcav/leaching/depth.py` |
| проверки | `tests/verification/test_lab_channel.py`, `test_lab_section.py`, `test_lab_depth.py` |
| примеры опытов | `configs/lab/channel.yaml`, `window.yaml`, `deep_1200m.yaml` |

## Использование

```python
from saltcav.core.lab_config import load_lab_config
from saltcav.leaching.section import simulate_section
from saltcav.leaching.depth import hydraulics, creep_estimate, depth_study

cfg = load_lab_config("configs/lab/channel.yaml")
res = simulate_section(cfg)          # model: section; для model: axisym — simulate_lab(cfg)
res.times, res.outlet_concentration  # ряды результата (СИ: с, кг/м³)
res.radius[-1]                       # r(x, θ) в последний момент, м
hyd = hydraulics(res)                # давление насоса и в каверне, запаздывание в трубке
```

## Параметры (YAML)

| Поле | Что это |
| --- | --- |
| `setup` | `channel` или `window` |
| `depth_m` | глубина установки от поверхности, м (0 — на поверхности) |
| `tube_diameter_mm` | внутренний диаметр подающей и отводящей трубок |
| `geothermal_gradient_c_per_100m` | геотермический градиент; не задан — температура одна |
| `rock_density_kg_m3` | плотность пород над установкой (горное давление, ползучесть) |
| `model` | `section` или `axisym` |
| `natural_convection` | учитывать свободную конвекцию и расслоение |
| `length_mm`, `diameter_mm` | длина и начальный диаметр канала |
| `max_radius_mm` | от оси до ближайшей грани блока; расчёт останавливается, когда канал до неё дорастёт |
| `temperature_c`, `salt_density_kg_m3` | температура опыта, плотность соли |
| `sh_mult`, `roof_mult`, `wall_mult` | множители массоотдачи: вынужденной, у потолка, у боков (калибруются по опытам) |
| `ri_crit` | Gr/Re², выше которого рассол в сечении расслаивается |
| `k_surface_m_s` | константа скорости реакции на поверхности соли, м/с; не задана — транспортный контроль |
| `salt_conductivity_w_mk` | теплопроводность соли (оценка охлаждения стенки теплотой растворения) |
| `n_cells`, `n_theta` | ячеек по длине и секторов на половину сечения |
| `output_every_min` | шаг записи результатов |
| `stages` | режим насоса: `duration_h`, `flow_ml_min`, `inlet_saturation` (0 — дистиллят, 1 — насыщенный рассол) |

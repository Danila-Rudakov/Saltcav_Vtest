# Переписывание saltcav под лабораторный канал: карта

Программа Сергея Кизреда (saltcav 1.1.4) остаётся основой: её каркас, физика и все остальные
части на месте. **Заменены** её геометрия и её модели — вместо вертикальной скважинной
каверны стоят геометрия и модели горизонтального канала в блоке NaCl. Всё остальное
(консоль, окно, чтение данных, калибровка, ансамбли…) ещё работает со старыми геометрией
и моделями и **переписывается по мере адаптации** — по этой карте видно, что именно.

## Что заменено

| Было (saltcav 1.1.4) | Стало |
| --- | --- |
| `geometry/cavern.py` — каверна R(z, φ) | `geometry/channel.py` (круглый канал r(x)), `geometry/section.py` (сечение r(θ, x)) |
| `leaching/simulator.py` — вертикальная модель, `History`, `simulate` | `leaching/axisym.py` (простая модель, `simulate_lab`), `leaching/section.py` (основная, `simulate_section`), `leaching/depth.py` (глубина: трубки, давление, ползучесть) |
| `leaching/plume.py`, `insolubles.py`, `ledges.py`, `properties.py` | удалены: плюм у скважины, нерастворимые, карнизы, неоднородность породы — в лабораторном блоке не нужны |
| тесты этих модулей (`test_cavern`, `test_simulator`, `test_plume`, `test_insolubles`, `test_ledges`, `test_azimuth`, `test_geometry_analytic`, `test_wall_motion`, `test_sansmic_benchmark`) | `tests/verification/test_lab_channel.py`, `test_lab_section.py`, `test_lab_depth.py` |

Новое, чего у Сергея не было: `core/lab_config.py` (схема опыта) и дополнения в
`core/brine.py` (вязкость рассола, сжимаемость, влияние давления). Описание — [lab/README.md](lab/README.md),
формулы — [lab/МАТЕМАТИЧЕСКАЯ_МОДЕЛЬ.md](lab/МАТЕМАТИЧЕСКАЯ_МОДЕЛЬ.md).

## Что осталось как у Сергея (не тронуто, работает)

| Часть | Комментарий |
| --- | --- |
| `leaching/dissolution.py`, `leaching/brine_column.py` | физика Сергея: `wall_growth_rate`, `node_concentration`, `transport_step`, `stabilize`; на ней построены новые модели |
| `core/config.py`, `core/units.py`, `core/rng.py`, `core/parallel.py` | `core/config.py` описывает скважину; лабораторная схема лежит рядом — `core/lab_config.py` |
| `mechanics/` (включая `creep.py`: закон BGRa), `mesh/` | `leaching/depth.py` берёт закон ползучести из `mechanics/creep.py` |
| `io/fielddata.py`, `workspace.py`, `selftest.py`, `shortcut.py`, `report.py`, `surrogate/model.py`, `viz/` (кроме перечисленных ниже) | импортируются без ошибок |
| `docs/` Сергея (`user-guide`, `progress`, `software-plan` и др.) | описывают исходную вертикальную программу |

## Что ждёт адаптации

Эти модули ссылаются на `Cavern`, `CavernGrid`, `History` или `simulate` — их больше нет.
Исполнитель — в задачах репозитория, здесь не фиксируется.

| Модуль | Что в нём привязано к заменённому |
| --- | --- |
| `api.py`, `cli.py` | `simulate` (внутри функций) |
| `gui/app.py` | `simulate` и форма вертикальной модели |
| `io/results.py` | `History` (сохранение результатов) |
| `operations/schedule.py` | `CavernGrid` (ступени с башмаками и подушкой) — вместо него нужен режим насоса: `LabStage` из `core/lab_config.py` |
| `calibration/` | `History`, `simulate` |
| `stochastic/` | `Cavern`, `CavernGrid`, `simulate` |
| `surrogate/features.py`, `surrogate/study.py` | `Cavern`, `CavernGrid` |
| `viz/calibration_plot.py`, `envelope.py`, `probability_plot.py`, `surrogate_plot.py` | `History`, `Cavern` |

Пока эти модули не адаптированы, `import saltcav.operations`, `saltcav.calibration`,
`saltcav.stochastic` и отдельные `viz.*` не работают, а консоль и окно не могут запустить расчёт.
Расчёт лабораторного канала запускается из Python (пример — [lab/README.md](lab/README.md)).

## Тесты

| | |
| --- | --- |
| проходят | 127: 40 лабораторных + нетронутые тесты Сергея |
| красные | 50 прежде проходивших тестов Сергея в `test_api`, `test_calibration_cli`, `test_config` (1), `test_parallel`, `test_report`, `test_results`, `test_rng_units_cli` (1), `test_schedule`, `test_stochastic`, `test_surrogate`, `tests/verification/test_calibration` — по причине из таблицы выше, это список задач на адаптацию |
| пропущены | GUI и геомеханика: нужны PySide6, gmsh |

Каждый красный тест падает на `ImportError` заменённых имён и должен позеленеть после
адаптации соответствующего модуля.

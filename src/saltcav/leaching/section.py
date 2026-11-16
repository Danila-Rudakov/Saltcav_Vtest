"""Модель сечения лабораторного канала: форма r(θ, x), конвекция, расслоение рассола.

Геометрия
---------
Ось канала x горизонтальна. Сечение описывается радиусом от оси r(θ, x), угол θ
отсчитывается от вертикали вверх: θ = 0 — потолок, θ = 90° — бок, θ = 180° — дно.
Считается половина сечения θ ∈ [0, π] по одну сторону от вертикальной плоскости
через ось:

* ``setup: channel`` — канал в целом блоке. Задача с гравитацией симметрична
  относительно этой плоскости, поэтому вторая половина — зеркальная. Через половину
  идёт Q/2, плоскость симметрии не стенка.
* ``setup: window`` — полуканал, прижатый к вертикальному окну из оргстекла (на эпоксидной
  смоле). Окно не растворяется и не пропускает соль; через половину идёт весь расход Q,
  окно добавляется к смоченному периметру (гидравлический диаметр меньше).
* ``setup: full`` — канал в целом блоке, но считается весь контур θ ∈ [0, 2π] без
  допущения о зеркальной симметрии: неоднородность материала и выступы стенки не обязаны
  быть симметричны. Угловая производная периодическая (узел θ=2π совпадает с θ=0), дно —
  узел θ=π, через сечение идёт весь расход Q.

Сетка: узлы x_i (n_cells+1) × θ_j (n_theta+1). Ячейка (i, j) — клин между лучами θ_j и
θ_{j+1} на отрезке [x_i, x_{i+1}]; в ней концентрация c_ij.

Физика
------
Поток соли со стенки клина: j = k·(C_s − c), стенка отступает по нормали со скоростью
V = j/(ρ_s − C_s) (условие Стефана: растворившаяся соль замещается раствором, поэтому
граница отступает быстрее, чем при V = j/ρ_s, в ρ_s/(ρ_s − C_s) ≈ 1,17 раза; отключается
параметром ``stefan_condition: false`` — тогда модель без гравитации совпадает с простой).
Массоотдача — смешанная конвекция (правило Черчилля):

    k = (k_вын³ + k_св³)^(1/3),

* вынужденная (ламинарная, Левек—Грэц): Sh = max(3,66; 1,077·Gz^(1/3))·sh_mult,
  Gz = u·D_h²/(D·x), k_вын = Sh·D/D_h;
* свободная (рассол у стенки насыщен и тяжелее ядра на Δρ):
  потолок (неустойчивый слой, плюмы) k_пот = 0,15·(g·Δρ/ρ·D²/ν)^(1/3)·roof_mult;
  боковая стенка (опускающийся пограничный слой высоты H)
  k_бок = 0,59·(D/H)·Ra_H^(1/4)·wall_mult, Ra_H = g·(Δρ/ρ)·H³/(ν·D);
  дно (устойчивый слой тяжёлого рассола) — свободной конвекции нет.
  Между ними — линейно по n_z, вертикальной составляющей нормали «в соль».

Движение стенки: ∂r/∂t = V·√(1 + (∂r/∂θ / r)² + (∂r/∂x)²), монотонная схема Годунова.

Рассол: вдоль x — неявный перенос против потока (как в saltcav); каждый клин течёт на
своей высоте, поток делится между клиньями пропорционально их объёму. В сечении: если
Gr/Re² < ri_crit, вынужденное течение перемешивает сечение; иначе рассол расслаивается:
соль со стенки стекает с пограничным слоем вниз (:func:`deposit_below`), затем плотностная
устойчивость (:func:`saltcav.leaching.brine_column.stabilize`) по высоте центров клиньев:
лёгкая вода вверху, тяжёлый рассол внизу.

Без свободной конвекции (``natural_convection: false``, ``setup: channel``) модель
сводится к круглому сечению простой модели :func:`saltcav.leaching.axisym.simulate_lab`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from saltcav.core import brine
from saltcav.core.lab_config import LabConfig
from saltcav.geometry.section import Geometry, geometry, wall_speed_factor
from saltcav.leaching.axisym import MAX_DR_FRACTION, MAX_GROWTH_FRACTION, MIN_DT, SH_GRAETZ, G
from saltcav.leaching.brine_column import stabilize
from saltcav.leaching.material import build_multiplier

FloatArray = NDArray[np.float64]

WATER_ITER = 2  # итераций плотности в балансе массы раствора
C_ROOF = 0.15  # Sh = 0,15·Ra^(1/3), неустойчивый слой под горизонтальной поверхностью
C_WALL = 0.59  # Sh = 0,59·Ra^(1/4), ламинарный слой у вертикальной стенки


@dataclass
class Props:
    """Свойства рассола NaCl при температуре и давлении опыта как функции концентрации c, кг/м³.

    При 0,1 МПа: плотность и растворимость — таблицы CRC (saltcav), вязкость — Kestin и др.
    (1981), коэффициент диффузии — D(T) saltcav (разбавленный раствор). Давление: сжимаемость
    рассола — Rowe, Chou (1970); вязкость — Kestin и др. (1981); растворимость — Sawamura и др.
    (2007); диффузия — по Стоксу—Эйнштейну через вязкость воды. Для пограничного слоя у стенки
    свойства берутся при средней концентрации слоя (c + C_s)/2 («плёночные»).
    """

    t_c: float
    p_mpa: float
    c_sat: float
    diff: float
    rho_sat: float
    c_tab: FloatArray
    rho_tab: FloatArray
    nu_tab: FloatArray

    @classmethod
    def at(cls, t_c: float, p_mpa: float = brine.P_ATM_MPA) -> Props:
        m_s = float(brine.molality_from_mass_fraction(brine.saturation_mass_fraction(t_c)))
        m_s *= 1.0 + brine.SOLUBILITY_PRESSURE_COEF * (p_mpa - brine.P_ATM_MPA)
        w = np.linspace(0.0, m_s * brine.M_NACL / (1.0 + m_s * brine.M_NACL), 201)
        rho_tab = brine.brine_density(w, t_c) * brine.pressure_density_factor(w, t_c, p_mpa)
        c_tab = w * rho_tab
        mu = brine.brine_viscosity(w, t_c) * brine.viscosity_pressure_factor(w, t_c, p_mpa)
        diff = float(brine.diffusivity(t_c) / brine.viscosity_pressure_factor(0.0, t_c, p_mpa))
        return cls(t_c, p_mpa, float(c_tab[-1]), diff, float(rho_tab[-1]), c_tab, rho_tab,
                   mu / rho_tab)  # fmt: skip

    def rho(self, c: FloatArray) -> FloatArray:
        return np.asarray(np.interp(c, self.c_tab, self.rho_tab), dtype=np.float64)

    def nu(self, c: FloatArray) -> FloatArray:
        return np.asarray(np.interp(c, self.c_tab, self.nu_tab), dtype=np.float64)

    def nu_film(self, c: FloatArray) -> FloatArray:
        return self.nu(0.5 * (np.asarray(c) + self.c_sat))


def mass_transfer(cfg: LabConfig, g: Geometry, c: FloatArray, x_mid: FloatArray,
                  q: float | FloatArray, pr: Props, flow_share: FloatArray | None = None
                  ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:  # fmt: skip
    """k по ячейкам (nx, nθ), а также k_вын по ячейкам, Gr/Re² по x и k_св по ячейкам.

    ``q`` — осевой расход через половину/весь контур, м³/с: число или массив по x (nx,); в
    карманах у трубок (там нет прокачки) он равен 0 — остаётся диффузия и свободная конвекция.

    ``flow_share`` — доля осевого расхода, проходящая через клин (nx, nθ), сумма по θ равна 1;
    по умолчанию пропорциональна объёму клина (тогда скорость в клине равна средней по сечению
    и k_вын одинаков по θ, как в модели без угловой неоднородности потока).

    k — итоговый коэффициент с учётом реакции на поверхности (последовательные стадии):
    1/k = 1/k_массоотдачи + 1/k_реакции.
    """
    dh = 4.0 * g.area / g.perim
    vol_share = g.vol / np.maximum(g.vol.sum(axis=1, keepdims=True), 1e-30)
    area_sector = np.maximum(vol_share * g.area[:, None], 1e-15)  # площадь сечения клина, м²
    if flow_share is None:
        flow_share = vol_share
    q_row = np.broadcast_to(np.asarray(q, dtype=np.float64), g.area.shape)
    u_local = q_row[:, None] * flow_share / area_sector  # локальная скорость в клине, м/с
    x_dev = np.maximum(x_mid, 1e-9)[:, None]  # путь развития пограничного слоя; в карманах u = 0
    gz = u_local * dh[:, None] ** 2 / (pr.diff * x_dev)  # вязкость в Gz сокращается
    k_f = (np.maximum(SH_GRAETZ, 1.077 * gz ** (1.0 / 3.0)) * cfg.sh_mult * pr.diff
           / dh[:, None])  # fmt: skip
    rho = pr.rho(np.clip(c, 0.0, pr.c_sat))
    drel = np.maximum(pr.rho_sat - rho, 0.0) / rho
    vol_i = g.vol.sum(axis=1)
    drel_mean = np.sum(g.vol * drel, axis=1) / vol_i
    u_bulk = np.maximum(q_row / g.area, 1e-12)  # средняя по сечению скорость, м/с
    ri = G * drel_mean * dh / (u_bulk * u_bulk)  # Gr/Re²
    if not cfg.natural_convection:
        k_n = np.zeros_like(c)
    else:
        nu = pr.nu_film(np.clip(c, 0.0, pr.c_sat))
        k_roof = C_ROOF * (G * drel * pr.diff**2 / nu) ** (1.0 / 3.0) * cfg.roof_mult
        h = g.height[:, None]
        ra = G * drel * h**3 / (nu * pr.diff)
        k_wall = C_WALL * pr.diff / h * ra**0.25 * cfg.wall_mult
        up = np.clip(g.nz, 0.0, 1.0)
        down = np.clip(-g.nz, 0.0, 1.0)
        k_n = (1.0 - up - down) * k_wall + up * k_roof  # к дну k_св → 0
    k = np.cbrt(k_f**3 + k_n**3)
    if cfg.k_surface_m_s is not None:
        k = 1.0 / (1.0 / k + 1.0 / cfg.k_surface_m_s)
    return k, k_f, ri, k_n


H_SOLUTION = 3.88e3  # Дж/моль, растворение NaCl, бесконечное разбавление, 25 °C (Parker, 1965)


def wall_cooling(
    flux: FloatArray, r_cell: FloatArray, r_block: float | None, conductivity: float
) -> float:
    """Оценка охлаждения стенки теплотой растворения, К (максимум по каверне).

    Тепло q = j·ΔH/M подводится к стенке теплопроводностью соли от блока; для
    цилиндрической полости радиуса r в блоке радиуса R: ΔT ≈ q·r·ln(R/r)/λ. Отвод
    тепла рассолом не учитывается, поэтому оценка сверху.
    """
    big = np.maximum(r_block if r_block is not None else 10.0 * r_cell, 2.0 * r_cell)
    q = flux * H_SOLUTION / brine.M_NACL
    return float(np.max(q * r_cell * np.log(big / r_cell) / conductivity))


def hydraulic_flow_distribution(
    g: Geometry, r: FloatArray, dx: float, theta: FloatArray, strength: float = 0.0
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Распределение осевого потока по угловым клиньям: пропорционально объёму клина.

    Точка расширения для моделей обхода выступов потолка: феноменологическая «гидравлическая
    сеть» с заранее заданными зонами обхода убрана (она предписывала положение «рогов» сечения
    и не была обоснована физикой). Параметр ``strength`` сохранён для совместимости вызовов и
    ни на что не влияет; ``r``, ``dx``, ``theta`` оставлены для будущих замыканий.

    Возвращает доли потока по клиньям (nx, nθ), индекс блокировки потолка (нули) и высоту
    выступов (нули), чтобы не менять сигнатуру для остального кода.
    """
    share = g.vol / np.maximum(g.vol.sum(axis=1, keepdims=True), 1e-30)
    return share, np.zeros(g.vol.shape[0]), np.zeros_like(g.vol)


def wall_rate(cfg: LabConfig, g: Geometry, r: FloatArray, c: FloatArray, x_mid: FloatArray,
              q: float | FloatArray, pr: Props, flow_share: FloatArray | None = None
              ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:  # fmt: skip
    """Скорость роста радиуса в узлах ∂r/∂t, м/с, и (k, k_вын, Gr/Re², k_св)."""
    k, k_f, ri, k_n = mass_transfer(cfg, g, c, x_mid, q, pr, flow_share)
    flux = k * np.maximum(pr.c_sat - c, 0.0)  # кг/(м²·с), по ячейкам
    if cfg.setup == "full":
        # θ периодичен: ячейки по краям замыкаются друг на друга (θ=2π ≡ 0), по x — край
        ftheta = np.concatenate([flux[:, -1:], flux, flux[:, :1]], axis=1)
        fp = np.pad(ftheta, ((1, 1), (0, 0)), mode="edge")
    else:
        fp = np.pad(flux, 1, mode="edge")
    v_node = 0.25 * (fp[:-1, :-1] + fp[1:, :-1] + fp[:-1, 1:] + fp[1:, 1:])
    # Условие Стефана: нормальная скорость границы V = j/(ρ_s − C_s), а не j/ρ_s
    v_node = v_node / (cfg.salt_density_kg_m3 - (pr.c_sat if cfg.stefan_condition else 0.0))
    dx = float(x_mid[1] - x_mid[0]) if x_mid.size > 1 else 1.0
    dtheta_total = 2.0 * np.pi if cfg.setup == "full" else np.pi
    dth = dtheta_total / (r.shape[1] - 1)
    return v_node * wall_speed_factor(r, dx, dth, periodic=cfg.setup == "full"), k, k_f, ri, k_n


@dataclass
class SectionResult:
    """История прогона модели сечения. Объёмы и массы — для всего канала (обе половины)."""

    cfg: LabConfig
    x: FloatArray  # узлы по x, м
    theta: FloatArray  # узлы по θ, рад
    c_sat: float
    halves: int  # 2 — канал (половина зеркальная), 1 — окно или полный контур (full)
    n_pocket: int = 0  # ячеек кармана у каждой трубки: x содержит их слева и справа от 0…L
    times: list[float] = field(default_factory=list)
    radius: list[FloatArray] = field(default_factory=list)  # (nx+1, nθ+1), м
    concentration: list[FloatArray] = field(default_factory=list)  # (nx, nθ)
    outlet_concentration: list[float] = field(default_factory=list)
    volume: list[float] = field(default_factory=list)
    dissolved: list[float] = field(default_factory=list)
    produced: list[float] = field(default_factory=list)
    supplied: list[float] = field(default_factory=list)
    in_channel: list[float] = field(default_factory=list)
    flow: list[float] = field(default_factory=list)  # м³/с, расход насоса
    ri_max: list[float] = field(default_factory=list)
    ri_mid: list[float] = field(default_factory=list)
    stratified: list[float] = field(default_factory=list)  # доля ячеек по x с расслоением
    k_forced_mid: list[float] = field(default_factory=list)  # м/с, середина канала
    k_roof_mid: list[float] = field(default_factory=list)  # k_св у потолка, середина
    k_wall_mid: list[float] = field(default_factory=list)  # k_св у бока, середина
    by_part: list[FloatArray] = field(default_factory=list)  # кг: потолок, бока, дно
    # середина канала: скорость роста радиуса потолка, бока, дна, м/с
    v_mid: list[FloatArray] = field(default_factory=list)
    c_roof_mid: list[float] = field(default_factory=list)  # c/C_s у потолка
    c_floor_mid: list[float] = field(default_factory=list)  # c/C_s у дна
    by_mechanism: list[FloatArray] = field(default_factory=list)  # кг: вынужденная, свободная
    outflow: list[float] = field(default_factory=list)  # м³/с, рассол на выходе
    water_in: list[float] = field(default_factory=list)  # кг воды, накоплено: пришло
    water_out: list[float] = field(default_factory=list)  # кг воды, ушло
    water_in_channel: list[float] = field(default_factory=list)  # кг воды в каверне
    wall_cooling: list[float] = field(default_factory=list)  # К, оценка сверху
    damkohler: list[float] = field(default_factory=list)  # max k_массоотдачи/k_реакции
    re_mid: list[float] = field(default_factory=list)  # Re в середине канала
    pocket_dissolved: list[float] = field(default_factory=list)  # соль из карманов у трубок, кг
    # диагностика распределения потока по углу, середина канала (подробнее — в record())
    flow_share_mid: list[FloatArray] = field(default_factory=list)  # доли расхода по клиньям
    roof_blockage_mid: list[float] = field(default_factory=list)  # индекс блокировки потолка
    roof_flow_share_mid: list[float] = field(default_factory=list)  # доля потока в зоне ±22°
    side_flow_share_mid: list[float] = field(default_factory=list)  # доля в полосах около ±38°
    n_steps: int = 0
    stopped: str = ""

    @property
    def x_mid(self) -> FloatArray:
        return 0.5 * (self.x[:-1] + self.x[1:])


PART_NAMES = ("потолок", "боковые стенки", "дно")

# зоны диагностики распределения потока по углу (от вертикали вверх), градусы
ROOF_CORE_DEG = 22.0  # центральная верхняя зона ±22°
FLANK_DEG = 38.0  # верхние боковые полосы около 38° и 322°
FLANK_HALF_WIDTH_DEG = 21.0  # полуширина полосы


def _angle_dist(th: FloatArray, center: float) -> FloatArray:
    """Угловое расстояние до ``center`` с учётом периода 2π, рад."""
    return np.asarray(np.abs((th - center + np.pi) % (2.0 * np.pi) - np.pi), dtype=np.float64)


def deposit_below(base: FloatArray, dm: FloatArray, zc: FloatArray, v: FloatArray,
                  c_sat: float) -> FloatArray:  # fmt: skip
    """Масса соли по клиньям (n, nθ) после стока рассола со стенок.

    Пограничный слой у стенки насыщен и тяжелее ядра: он стекает вниз и перемешивается
    с рассолом на пути. Соль со стенки клина j распределяется по клиньям не выше его
    центра пропорционально объёму («заполняющийся объём»). Что не помещается до C_s,
    поднимается в клин выше.
    """
    below = zc[:, None, :] <= zc[:, :, None]  # [ячейка, источник j, приёмник k]
    w = below * v[:, None, :]
    w = w / w.sum(axis=2, keepdims=True)
    return overflow_up(base + np.einsum("ij,ijk->ik", dm, w), zc, c_sat * v)


def pool_equal_height(
    c: FloatArray, v: FloatArray, zc: FloatArray, tol: float = 1e-9
) -> FloatArray:
    """Выравнивает концентрацию клиньев одной высоты (в пределах ``tol`` от размаха высот).

    Клинья на одной высоте не разделены плотностной устойчивостью и перемешиваются по
    горизонтали; без этого порядок обработки равных высот (по индексу) вносил бы в полный
    контур ложную несимметрию. Масса соли сохраняется (среднее по объёму).
    """
    out = c.copy()
    span = float(np.ptp(zc)) or 1.0
    order = np.argsort(-zc)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and zc[order[start]] - zc[order[end]] <= tol * span:
            end += 1
        if end - start > 1:
            idx = order[start:end]
            out[idx] = float(np.dot(v[idx], c[idx]) / v[idx].sum())
        start = end
    return out


def overflow_up(mass: FloatArray, zc: FloatArray, cap: FloatArray) -> FloatArray:
    """Соль сверх ёмкости клина (C_s·V) поднимается в клин выше; масса сохраняется."""
    mass = mass.copy()
    order = np.argsort(zc, axis=1)  # снизу вверх
    for i in range(mass.shape[0]):
        carry = 0.0
        for k in order[i]:
            m = mass[i, k] + carry
            mass[i, k] = min(m, cap[i, k])
            carry = m - mass[i, k]
        mass[i, order[i][-1]] += carry  # сечение целиком насыщено — остаток вверху
    return mass


def pocket_side(
    idx: NDArray[np.intp], iw: int, mass: FloatArray, v_old: FloatArray, v_new: FloatArray,
    c: FloatArray, rho_old: FloatArray, rho_new: FloatArray, salt_rate: FloatArray, dt: float,
    diff: float, dx: float, exchange_mult: float,
) -> tuple[float, FloatArray]:  # fmt: skip
    """Карман у трубки: ячейки без прокачки, связанные с рабочей зоной через зазор.

    ``idx`` — строки кармана от стыка с рабочей зоной вглубь (до глухого торца), ``iw`` — строка
    рабочей зоны у стыка. Закрытый карман не вытесняет раствор, а всасывает воду: кавитационный
    объём растёт на 1/ρ_s на кг соли, раствор занимает меньше. Поток массы через стык
    F₀ = Σ(Δ(Vρ)/dt − ṁ) идёт от рабочей зоны вглубь (F₀ < 0 — наружу), по ячейкам он
    накапливается как F_{k+1} = F_k + ṁ_k − Δ(Vρ)_k/dt, на глухом торце равен нулю. Соль несут
    потоком (против потока, по клиньям) и диффузией через кольцевой зазор у стыка.

    Меняет ``mass`` кармана на месте. Возвращает F₀ (кг/с, поток массы раствора из рабочей зоны
    в карман) и поток соли через стык по клиньям (кг/с, из рабочей зоны в карман: перенос потоком
    плюс диффузия) — их вызывающий код вычитает из рабочей ячейки у стыка.
    """
    n = idx.size
    d_mass = (v_new[idx].sum(axis=1) * rho_new[idx] - v_old[idx].sum(axis=1) * rho_old[idx]) / dt
    flux = np.empty(n + 1)  # F_k, кг/с, k = 0 — стык, k = n — глухой торец (0)
    flux[0] = float(np.sum(d_mass - salt_rate[idx]))
    flux[1:] = flux[0] + np.cumsum(salt_rate[idx] - d_mass)
    flux[n] = 0.0
    share = v_new[idx] / v_new[idx].sum(axis=1, keepdims=True)  # доли клиньев в ячейках кармана
    salt = np.zeros((n + 1, c.shape[1]))  # поток соли через границу k, кг/с по клиньям
    for k in range(n):
        up = iw if k == 0 else int(idx[k - 1])
        dn = int(idx[k])
        src = up if flux[k] > 0 else dn
        rho_src = float(rho_old[src])
        salt[k] = flux[k] / rho_src * share[k] * c[src]
    # обмен через кольцевой зазор: на стыке и между соседними ячейками кармана (по клиньям,
    # масса сохраняется); поток ограничен половиной объёма меньшей из двух ячеек за шаг
    junction_ex = np.zeros(c.shape[1])
    for k in range(n):
        up = iw if k == 0 else int(idx[k - 1])
        dn = int(idx[k])
        v_gap = np.minimum(v_new[up], v_new[dn])  # зазор ограничен меньшей ячейкой
        q_ex = np.minimum(exchange_mult * diff * (v_gap / dx) / dx, 0.5 * v_gap / dt)
        ex = np.clip(q_ex * dt * (c[up] - c[dn]), -mass[dn], mass[up]) / dt  # кг/с вглубь
        mass[dn] += dt * ex
        if k == 0:
            junction_ex = ex
        else:
            mass[up] -= dt * ex
    for k in range(n):
        mass[idx[k]] += dt * (salt[k] - salt[k + 1])
    return float(flux[0]), salt[0] + junction_ex


def transport_mass(mass: FloatArray, v_old: FloatArray, v_new: FloatArray, a: FloatArray,
                   rho_old: FloatArray, rho_new: FloatArray, salt_rate: FloatArray, q_in: float,
                   rho_in: float, c_in: float,
                   dt: float) -> tuple[FloatArray, FloatArray]:  # fmt: skip
    """Неявный шаг переноса вдоль канала с расходом из баланса массы раствора.

    Расход из ячейки i: Q_i = (ρ_{i−1}Q_{i−1} + ṁ_i − (V′_iρ′_i − V_iρ_i)/dt)/ρ′_i, где ṁ_i —
    растворённая за шаг соль в ячейке, кг/с. Соль: V′c′ = Vc + Δm + dt·(приток − отток),
    клин j ячейки i получает долю a_{i−1,j} притока и отдаёт долю a_{ij} оттока.
    Возвращает концентрацию (nx, nθ) и расходы Q_i (nx,), м³/с.
    """
    nx = mass.shape[0]
    cn, qs = np.empty_like(mass), np.empty(nx)
    q_up, rho_up, inflow = q_in, rho_in, q_in * a[0] * c_in
    for i in range(nx):
        dmass = (v_new[i].sum() * rho_new[i] - v_old[i].sum() * rho_old[i]) / dt
        qi = (rho_up * q_up + salt_rate[i] - dmass) / rho_new[i]
        if qi <= 0:
            raise RuntimeError("расход в канале стал неположительным — уменьшите шаг")
        cn[i] = (mass[i] + dt * inflow) / (v_new[i] + dt * qi * a[i])
        inflow, q_up, rho_up, qs[i] = qi * a[i] * cn[i], qi, rho_new[i], qi
    return cn, qs


def _parts(nz: FloatArray) -> list[NDArray[np.bool_]]:
    return [nz > 0.5, np.abs(nz) <= 0.5, nz < -0.5]


def conditions(cfg: LabConfig) -> tuple[float, float]:
    """Температура (°C) и давление (МПа, абс.) в каверне на глубине установки.

    Давление — атмосферное плюс столб воды в подающей трубке (средняя температура и давление
    столба). Насос добавляет ещё перепад столбов рассола и воды (до ~2 МПа на 1200 м) и
    трение — их считает :func:`saltcav.lab_depth.hydraulics`; на свойства это влияет < 0,1 %.
    """
    t = cfg.temperature_at_depth
    if cfg.depth_m == 0:
        return t, brine.P_ATM_MPA
    t_col = 0.5 * (cfg.temperature_c + t)
    rho0 = float(brine.water_density(t_col))
    p_mid = brine.P_ATM_MPA + 0.5 * rho0 * G * cfg.depth_m * 1e-6
    rho = rho0 * float(brine.pressure_density_factor(0.0, t_col, p_mid))
    return t, brine.P_ATM_MPA + rho * G * cfg.depth_m * 1e-6


def simulate_section(
    cfg: LabConfig,
    progress: Callable[[float, str], None] | None = None,
    flow_coupling_strength: float = 0.0,
    material_multiplier: Callable[[FloatArray], FloatArray] | None = None,
) -> SectionResult:
    """Прогнать режим насоса в модели сечения.

    ``progress(доля, текст)`` вызывается в моменты записи (для окна: полоса хода и отмена).

    ``material_multiplier(r)`` — необязательный множитель скорости растворения в узлах
    (nx+1, nθ+1) как функция текущей формы стенки r (м): неоднородность материала (зёрна,
    слои). Не задан — берётся из ``cfg.material`` (см. :mod:`saltcav.leaching.material`),
    а если и там не задано, материал однороден. ``flow_coupling_strength`` передаётся в
    :func:`hydraulic_flow_distribution` и сейчас ни на что не влияет.
    """
    pr = Props.at(*conditions(cfg))
    if material_multiplier is None:
        material_multiplier = build_multiplier(cfg)
    nx_w, nth = cfg.n_cells, cfg.n_theta
    n_p = cfg.n_pocket  # ячеек кармана у каждой трубки (0 — трубок нет)
    nx = nx_w + 2 * n_p  # всего ячеек: карман на входе, рабочая зона, карман на выходе
    w0, w1 = n_p, n_p + nx_w  # рабочая зона — строки w0…w1−1 (x = 0…L)
    window = cfg.setup == "window"
    full = cfg.setup == "full"  # полный контур 0…2π, без допущения о симметрии
    halves = 1 if window or full else 2
    dx = cfg.length_mm * 1e-3 / nx_w
    x = (np.arange(nx + 1) - n_p) * dx  # x = 0 — торец подающей трубки, x = L — отводящей
    x_mid = 0.5 * (x[:-1] + x[1:])
    theta = np.linspace(0.0, 2.0 * np.pi if full else np.pi, nth + 1)
    dth = float(theta[1] - theta[0])
    r = np.full((nx + 1, nth + 1), 0.5e-3 * cfg.diameter_mm)
    c = np.zeros((nx, nth))
    # трубки: наружный радиус = радиус отверстия − зазор, непроницаемы, в карманах вытесняют объём
    tube_r = np.zeros(nx)
    tube_r[:w0] = tube_r[w1:] = max(0.5e-3 * (cfg.diameter_mm - 2.0 * cfg.tube_gap_mm), 0.0)

    def geom(rr: FloatArray) -> Geometry:
        return geometry(rr, dx, theta, window, tube_r if n_p else None)

    def q_rows(flow: float) -> FloatArray:
        """Осевой расход по строкам: прокачка только на рабочем участке между трубками."""
        out = np.zeros(nx)
        out[w0:w1] = flow
        return out

    rho_s = cfg.salt_density_kg_m3
    r_lim = None if cfg.max_radius_mm is None else 1e-3 * cfg.max_radius_mm
    dt_out = 60.0 * cfg.output_every_min
    mid = w0 + nx_w // 2  # середина рабочей зоны

    res = SectionResult(cfg, x, theta, pr.c_sat, halves, n_p)

    def q_half(stage_flow: float) -> float:
        return stage_flow * 1e-6 / 60.0 / halves

    acc = {"pocket": 0.0, "dissolved": 0.0, "produced": 0.0, "supplied": 0.0, "water_in": 0.0,
           "water_out": 0.0, "q_out": q_half(cfg.stages[0].flow_ml_min)}  # fmt: skip
    parts = np.zeros(3)
    mech = np.zeros(2)

    def record(t: float, q_h: float) -> None:
        g = geom(r)
        flow_share, roof_blockage, _ = hydraulic_flow_distribution(
            g, r, dx, theta, flow_coupling_strength
        )
        rate, k, k_f, ri, k_n = wall_rate(cfg, g, r, c, x_mid, q_rows(q_h), pr, flow_share)
        if material_multiplier is not None:
            rate = rate * np.asarray(material_multiplier(r), dtype=np.float64)
        q = halves * q_h
        flux = k * np.maximum(pr.c_sat - c, 0.0)
        r_cell = 0.25 * (r[:-1, :-1] + r[1:, :-1] + r[:-1, 1:] + r[1:, 1:])
        r_block = None if cfg.max_radius_mm is None else 1e-3 * cfg.max_radius_mm
        res.wall_cooling.append(wall_cooling(flux, r_cell, r_block, cfg.salt_conductivity_w_mk))
        k_mt = np.cbrt(k_f**3 + k_n**3)
        res.damkohler.append(
            float(np.max(k_mt) / cfg.k_surface_m_s) if cfg.k_surface_m_s is not None else 0.0
        )
        c_bar = float(np.sum(g.vol[mid] * c[mid]) / g.vol[mid].sum())
        dh = 4.0 * g.area[mid] / g.perim[mid]
        res.re_mid.append(float(q_h / g.area[mid] * dh / pr.nu(c_bar)))
        res.flow_share_mid.append(flow_share[mid].copy())
        res.roof_blockage_mid.append(float(roof_blockage[mid]))
        # доля потока в центральной верхней зоне и в верхних боковых полосах (диагностика)
        thm = 0.5 * (theta[:-1] + theta[1:])
        roof_core = np.cos(thm) > np.cos(np.deg2rad(ROOF_CORE_DEG))
        flank = np.minimum(_angle_dist(thm, np.deg2rad(FLANK_DEG)),
                           _angle_dist(thm, np.deg2rad(360.0 - FLANK_DEG))) < np.deg2rad(
                               FLANK_HALF_WIDTH_DEG)  # fmt: skip
        res.roof_flow_share_mid.append(float(flow_share[mid, roof_core].sum()))
        res.side_flow_share_mid.append(float(flow_share[mid, flank].sum()))
        res.outflow.append(halves * acc["q_out"])
        res.water_in.append(halves * acc["water_in"])
        res.water_out.append(halves * acc["water_out"])
        res.water_in_channel.append(
            halves * float(np.sum(g.vol * (pr.rho(np.clip(c, 0, pr.c_sat)) - c)))
        )
        # узел бока и дна: при половинном сечении бок — θ=90° (середина), дно — последний узел;
        # при полном контуре бок — θ=90° (четверть), дно — θ=180° (середина)
        side_idx = nth // 4 if full else nth // 2
        floor_idx = nth // 2 if full else -1
        res.v_mid.append(rate[mid, [0, side_idx, floor_idx]].copy())
        res.c_roof_mid.append(float(c[mid, 0] / pr.c_sat))
        res.c_floor_mid.append(float(c[mid, floor_idx] / pr.c_sat))
        res.by_mechanism.append(halves * mech.copy())
        res.times.append(t)
        res.radius.append(r.copy())
        res.concentration.append(c.copy())
        res.outlet_concentration.append(
            float(np.dot(g.vol[w1 - 1], c[w1 - 1]) / g.vol[w1 - 1].sum())
        )
        res.pocket_dissolved.append(halves * acc["pocket"])
        res.volume.append(halves * float(g.vol.sum()))
        res.dissolved.append(halves * acc["dissolved"])
        res.produced.append(halves * acc["produced"])
        res.supplied.append(halves * acc["supplied"])
        res.in_channel.append(halves * float(np.sum(g.vol * c)))
        res.flow.append(q)
        res.ri_max.append(float(np.max(ri)))
        res.ri_mid.append(float(ri[mid]))
        res.stratified.append(float(np.mean(ri >= cfg.ri_crit)) if cfg.natural_convection else 0.0)
        res.k_forced_mid.append(float(np.mean(k_f[mid])))
        res.k_roof_mid.append(float(k_n[mid, 0]))
        res.k_wall_mid.append(float(k_n[mid, side_idx]))
        res.by_part.append(halves * parts.copy())

    t_total = 3600.0 * sum(s.duration_h for s in cfg.stages)
    record(0.0, q_half(cfg.stages[0].flow_ml_min))
    t, next_out, t_end = 0.0, dt_out, 0.0
    for stage in cfg.stages:
        q = q_half(stage.flow_ml_min)
        c_in = stage.inlet_saturation * pr.c_sat
        t_end += 3600.0 * stage.duration_h
        while t < t_end - 1e-9:
            g = geom(r)
            flow_share, _, _ = hydraulic_flow_distribution(g, r, dx, theta, flow_coupling_strength)
            rate, _, k_f, ri, k_n = wall_rate(cfg, g, r, c, x_mid, q_rows(q), pr, flow_share)
            if material_multiplier is not None:
                rate = rate * np.asarray(material_multiplier(r), dtype=np.float64)
            dt = min(t_end - t, next_out - t)
            v_max = float(np.max(rate))
            if v_max > 0:
                ds = min(dx, float(np.min(r)) * dth)
                dt = min(dt, cfg.cfl * ds / v_max, MAX_DR_FRACTION / float(np.max(rate / r)))
            v_old = g.vol
            # пересыщение проверяется по сечению целиком: внутри него соль перераспределяется
            salt_old = np.sum(v_old * c, axis=1)
            cap = np.maximum(salt_old, pr.c_sat * v_old.sum(axis=1)) * (1.0 + 1e-9)
            while True:
                r_new = r + dt * rate
                g_new = geom(r_new)
                dm = rho_s * (g_new.vol - v_old)
                # пересыщение проверяется только в рабочей зоне; избыток в карманах уходит к стыку
                if (float(g_new.vol.sum() - v_old.sum()) <= MAX_GROWTH_FRACTION * q * dt
                        and bool(np.all((salt_old + dm.sum(axis=1) <= cap)[w0:w1]))):  # fmt: skip
                    break
                if dt < MIN_DT:
                    raise RuntimeError(f"шаг по времени упал ниже {MIN_DT} с при t = {t:.0f} с")
                dt *= 0.5
            v_new = g_new.vol
            mass = v_old * c + dm
            strat = (ri >= cfg.ri_crit) if cfg.natural_convection else np.zeros(nx, dtype=bool)
            if strat.any():  # тяжёлый рассол со стенки стекает и копится внизу
                mass[strat] = deposit_below(v_old[strat] * c[strat], dm[strat], g.zc[strat],
                                            v_new[strat], pr.c_sat)  # fmt: skip
            # перенос вдоль x: неявная схема против потока; слои по высоте (клинья) текут
            # каждый на своей высоте, поток делится по их объёмам. Расход между ячейками —
            # из баланса массы раствора: каверна растёт на ΔV, но растворённая соль сама
            # добавляет объём раствору (плотность рассола из таблицы CRC это учитывает)
            a, _, _ = hydraulic_flow_distribution(g_new, r_new, dx, theta, flow_coupling_strength)
            rho_in = float(pr.rho(c_in))
            rho_old = np.sum(v_old * pr.rho(c), axis=1) / v_old.sum(axis=1)
            mass_in = dm.sum(axis=1) / dt
            if n_p:
                # узел на стыке общий для кармана и рабочей зоны: соль, растворённая сверх
                # насыщения ячейки кармана, передаётся по цепочке к рабочей зоне (с массой раствора)
                for rows, step in ((range(w0), 1), (range(nx - 1, w1 - 1, -1), -1)):
                    for i in rows:
                        over = float(mass[i].sum() - pr.c_sat * v_new[i].sum())
                        room = float(pr.c_sat * v_new[i + step].sum() - mass[i + step].sum())
                        send = min(over, max(room, 0.0))  # принимающая ячейка не пересыщается
                        if send > 0.0:
                            moved = mass[i] * (send / float(mass[i].sum()))
                            mass[i] -= moved
                            mass[i + step] += moved
                            mass_in[i] -= send / dt
                            mass_in[i + step] += send / dt
            rho_new = rho_old  # плотность нового состояния: уточняется итерациями
            pocket_rows = (np.arange(w0 - 1, -1, -1), np.arange(w1, nx))  # от стыка вглубь
            junction = (w0, w1 - 1)  # рабочие строки у стыков
            c_eff = np.minimum(c, pr.c_sat)  # остаток пересыщения в кармане дальше не передаётся
            for _ in range(WATER_ITER):
                mass_it, rate_it = mass.copy(), mass_in.copy()
                if n_p:  # карманы: всасывание воды и обмен через зазор, поправки у стыков
                    for idx, iw in zip(pocket_rows, junction, strict=True):
                        f_in, s_in = pocket_side(idx, iw, mass_it, v_old, v_new, c_eff, rho_old,
                                                 rho_new, mass_in, dt, pr.diff, dx,
                                                 cfg.pocket_exchange_mult)  # fmt: skip
                        mass_it[iw] -= dt * s_in
                        rate_it[iw] -= f_in
                c_work, qs_work = transport_mass(
                    mass_it[w0:w1], v_old[w0:w1], v_new[w0:w1], a[w0:w1], rho_old[w0:w1],
                    rho_new[w0:w1], rate_it[w0:w1], q, rho_in, c_in, dt)  # fmt: skip
                c_new = mass_it / v_new  # карманы: сразу из массы, рабочая зона — перенос
                c_new[w0:w1] = c_work
                rho_new = np.sum(v_new * pr.rho(np.clip(c_new, 0, pr.c_sat)), axis=1) / v_new.sum(1)
            qs = qs_work
            q_out = float(qs[-1])
            c_out = float(np.dot(a[w1 - 1], c_new[w1 - 1]))
            acc["produced"] += dt * q_out * c_out
            acc["supplied"] += dt * q * c_in
            acc["dissolved"] += float(dm.sum())
            acc["pocket"] += float(dm[:w0].sum() + dm[w1:].sum())
            acc["water_in"] += dt * q * (rho_in - c_in)
            acc["water_out"] += dt * q_out * (float(rho_new[w1 - 1]) - c_out)
            acc["q_out"] = q_out
            for p, mask in enumerate(_parts(g.nz)):
                parts[p] += float(dm[mask].sum())
            # доля потока, которую дала бы одна вынужденная конвекция
            forced = float(np.sum(dm * k_f / np.cbrt(k_f**3 + k_n**3)))
            mech += (forced, float(dm.sum()) - forced)
            # сечение: перемешивание (вынужденный режим) или расслоение (свободная конвекция)
            if strat.any():  # слои соседних ячеек разного объёма — без пересыщения
                c_new[strat] = overflow_up(c_new[strat] * v_new[strat], g_new.zc[strat],
                                           pr.c_sat * v_new[strat]) / v_new[strat]  # fmt: skip
            for i in range(nx):
                if strat[i]:
                    order = np.argsort(-g_new.zc[i])  # сверху вниз
                    c_eq = pool_equal_height(c_new[i], v_new[i], g_new.zc[i])
                    c_new[i, order] = stabilize(c_eq[order], v_new[i, order])
                else:
                    c_new[i] = np.dot(v_new[i], c_new[i]) / v_new[i].sum()
            r, c = r_new, c_new
            t += dt
            res.n_steps += 1
            broke = r_lim is not None and float(np.max(r)) >= r_lim
            if broke:
                res.stopped = f"канал дорос до грани блока на {t / 3600:.2f} ч"
            if broke or t >= next_out - 1e-9 or t >= t_end - 1e-9:
                record(t, q)
                if progress is not None:
                    progress(t / t_total, f"{t / 3600:.2f} из {t_total / 3600:g} ч")
                next_out = t + dt_out
            if broke:
                return res
    return res

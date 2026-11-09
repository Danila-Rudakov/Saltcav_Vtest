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

Сетка: узлы x_i (n_cells+1) × θ_j (n_theta+1). Ячейка (i, j) — клин между лучами θ_j и
θ_{j+1} на отрезке [x_i, x_{i+1}]; в ней концентрация c_ij.

Физика
------
Поток соли со стенки клина: j = k·(C_s − c), стенка отступает по нормали со скоростью
V = j/ρ_s. Массоотдача — смешанная конвекция (правило Черчилля):

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


def mass_transfer(cfg: LabConfig, g: Geometry, c: FloatArray, x_mid: FloatArray, q: float,
                  pr: Props) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:  # fmt: skip
    """k по ячейкам (nx, nθ), а также k_вын, Gr/Re² по x и k_св по ячейкам.

    k — итоговый коэффициент с учётом реакции на поверхности (последовательные стадии):
    1/k = 1/k_массоотдачи + 1/k_реакции.
    """
    dh = 4.0 * g.area / g.perim
    u = q / g.area
    gz = u * dh * dh / (pr.diff * x_mid)  # вязкость в Gz сокращается
    k_f = np.maximum(SH_GRAETZ, 1.077 * gz ** (1.0 / 3.0)) * cfg.sh_mult * pr.diff / dh
    rho = pr.rho(np.clip(c, 0.0, pr.c_sat))
    drel = np.maximum(pr.rho_sat - rho, 0.0) / rho
    vol_i = g.vol.sum(axis=1)
    drel_mean = np.sum(g.vol * drel, axis=1) / vol_i
    ri = G * drel_mean * dh / (u * u)  # Gr/Re²
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
    k = np.cbrt(k_f[:, None] ** 3 + k_n**3)
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


def wall_rate(cfg: LabConfig, g: Geometry, r: FloatArray, c: FloatArray, x_mid: FloatArray,
              q: float, pr: Props) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray,
                                            FloatArray]:  # fmt: skip
    """Скорость роста радиуса в узлах ∂r/∂t, м/с, и (k, k_вын, Gr/Re², k_св)."""
    k, k_f, ri, k_n = mass_transfer(cfg, g, c, x_mid, q, pr)
    flux = k * np.maximum(pr.c_sat - c, 0.0)  # кг/(м²·с), по ячейкам
    fp = np.pad(flux, 1, mode="edge")
    v_node = 0.25 * (fp[:-1, :-1] + fp[1:, :-1] + fp[:-1, 1:] + fp[1:, 1:]) / cfg.salt_density_kg_m3
    dx = float(x_mid[1] - x_mid[0]) if x_mid.size > 1 else 1.0
    dth = np.pi / (r.shape[1] - 1)
    return v_node * wall_speed_factor(r, dx, dth), k, k_f, ri, k_n


@dataclass
class SectionResult:
    """История прогона модели сечения. Объёмы и массы — для всего канала (обе половины)."""

    cfg: LabConfig
    x: FloatArray  # узлы по x, м
    theta: FloatArray  # узлы по θ, рад
    c_sat: float
    halves: int  # 2 — канал (половина зеркальная), 1 — окно
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
    n_steps: int = 0
    stopped: str = ""

    @property
    def x_mid(self) -> FloatArray:
        return 0.5 * (self.x[:-1] + self.x[1:])


PART_NAMES = ("потолок", "боковые стенки", "дно")


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
    cfg: LabConfig, progress: Callable[[float, str], None] | None = None
) -> SectionResult:
    """Прогнать режим насоса в модели сечения.

    ``progress(доля, текст)`` вызывается в моменты записи (для окна: полоса хода и отмена).
    """
    pr = Props.at(*conditions(cfg))
    nx, nth = cfg.n_cells, cfg.n_theta
    window = cfg.setup == "window"
    halves = 1 if window else 2
    x = np.linspace(0.0, cfg.length_mm * 1e-3, nx + 1)
    dx = float(x[1] - x[0])
    x_mid = 0.5 * (x[:-1] + x[1:])
    theta = np.linspace(0.0, np.pi, nth + 1)
    dth = float(theta[1] - theta[0])
    r = np.full((nx + 1, nth + 1), 0.5e-3 * cfg.diameter_mm)
    c = np.zeros((nx, nth))
    rho_s = cfg.salt_density_kg_m3
    r_lim = None if cfg.max_radius_mm is None else 1e-3 * cfg.max_radius_mm
    dt_out = 60.0 * cfg.output_every_min
    mid = nx // 2

    res = SectionResult(cfg, x, theta, pr.c_sat, halves)

    def q_half(stage_flow: float) -> float:
        return stage_flow * 1e-6 / 60.0 / halves

    acc = {"dissolved": 0.0, "produced": 0.0, "supplied": 0.0, "water_in": 0.0,
           "water_out": 0.0, "q_out": q_half(cfg.stages[0].flow_ml_min)}  # fmt: skip
    parts = np.zeros(3)
    mech = np.zeros(2)

    def record(t: float, q_h: float) -> None:
        g = geometry(r, dx, theta, window)
        rate, k, k_f, ri, k_n = wall_rate(cfg, g, r, c, x_mid, q_h, pr)
        q = halves * q_h
        flux = k * np.maximum(pr.c_sat - c, 0.0)
        r_cell = 0.25 * (r[:-1, :-1] + r[1:, :-1] + r[:-1, 1:] + r[1:, 1:])
        r_block = None if cfg.max_radius_mm is None else 1e-3 * cfg.max_radius_mm
        res.wall_cooling.append(wall_cooling(flux, r_cell, r_block, cfg.salt_conductivity_w_mk))
        k_mt = np.cbrt(k_f[:, None] ** 3 + k_n**3)
        res.damkohler.append(
            float(np.max(k_mt) / cfg.k_surface_m_s) if cfg.k_surface_m_s is not None else 0.0
        )
        c_bar = float(np.sum(g.vol[mid] * c[mid]) / g.vol[mid].sum())
        dh = 4.0 * g.area[mid] / g.perim[mid]
        res.re_mid.append(float(q_h / g.area[mid] * dh / pr.nu(c_bar)))
        res.outflow.append(halves * acc["q_out"])
        res.water_in.append(halves * acc["water_in"])
        res.water_out.append(halves * acc["water_out"])
        res.water_in_channel.append(
            halves * float(np.sum(g.vol * (pr.rho(np.clip(c, 0, pr.c_sat)) - c)))
        )
        res.v_mid.append(rate[mid, [0, nth // 2, nth]].copy())
        res.c_roof_mid.append(float(c[mid, 0] / pr.c_sat))
        res.c_floor_mid.append(float(c[mid, -1] / pr.c_sat))
        res.by_mechanism.append(halves * mech.copy())
        res.times.append(t)
        res.radius.append(r.copy())
        res.concentration.append(c.copy())
        res.outlet_concentration.append(float(np.dot(g.vol[-1], c[-1]) / g.vol[-1].sum()))
        res.volume.append(halves * float(g.vol.sum()))
        res.dissolved.append(halves * acc["dissolved"])
        res.produced.append(halves * acc["produced"])
        res.supplied.append(halves * acc["supplied"])
        res.in_channel.append(halves * float(np.sum(g.vol * c)))
        res.flow.append(q)
        res.ri_max.append(float(np.max(ri)))
        res.ri_mid.append(float(ri[mid]))
        res.stratified.append(float(np.mean(ri >= cfg.ri_crit)) if cfg.natural_convection else 0.0)
        res.k_forced_mid.append(float(k_f[mid]))
        res.k_roof_mid.append(float(k_n[mid, 0]))
        res.k_wall_mid.append(float(k_n[mid, nth // 2]))
        res.by_part.append(halves * parts.copy())

    t_total = 3600.0 * sum(s.duration_h for s in cfg.stages)
    record(0.0, q_half(cfg.stages[0].flow_ml_min))
    t, next_out, t_end = 0.0, dt_out, 0.0
    for stage in cfg.stages:
        q = q_half(stage.flow_ml_min)
        c_in = stage.inlet_saturation * pr.c_sat
        t_end += 3600.0 * stage.duration_h
        while t < t_end - 1e-9:
            g = geometry(r, dx, theta, window)
            rate, _, k_f, ri, k_n = wall_rate(cfg, g, r, c, x_mid, q, pr)
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
                g_new = geometry(r_new, dx, theta, window)
                dm = rho_s * (g_new.vol - v_old)
                if (float(g_new.vol.sum() - v_old.sum()) <= MAX_GROWTH_FRACTION * q * dt
                        and bool(np.all(salt_old + dm.sum(axis=1) <= cap))):  # fmt: skip
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
            a = v_new / v_new.sum(axis=1, keepdims=True)
            rho_in = float(pr.rho(c_in))
            rho_old = np.sum(v_old * pr.rho(c), axis=1) / v_old.sum(axis=1)
            mass_in = dm.sum(axis=1) / dt
            rho_new = rho_old  # плотность нового состояния: уточняется итерациями
            for _ in range(WATER_ITER):
                c_new, qs = transport_mass(mass, v_old, v_new, a, rho_old, rho_new, mass_in,
                                           q, rho_in, c_in, dt)  # fmt: skip
                rho_new = np.sum(v_new * pr.rho(np.clip(c_new, 0, pr.c_sat)), axis=1) / v_new.sum(1)
            q_out = float(qs[-1])
            c_out = float(np.dot(a[-1], c_new[-1]))
            acc["produced"] += dt * q_out * c_out
            acc["supplied"] += dt * q * c_in
            acc["dissolved"] += float(dm.sum())
            acc["water_in"] += dt * q * (rho_in - c_in)
            acc["water_out"] += dt * q_out * (float(rho_new[-1]) - c_out)
            acc["q_out"] = q_out
            for p, mask in enumerate(_parts(g.nz)):
                parts[p] += float(dm[mask].sum())
            # доля потока, которую дала бы одна вынужденная конвекция
            forced = float(np.sum(dm * k_f[:, None] / np.cbrt(k_f[:, None] ** 3 + k_n**3)))
            mech += (forced, float(dm.sum()) - forced)
            # сечение: перемешивание (вынужденный режим) или расслоение (свободная конвекция)
            if strat.any():  # слои соседних ячеек разного объёма — без пересыщения
                c_new[strat] = overflow_up(c_new[strat] * v_new[strat], g_new.zc[strat],
                                           pr.c_sat * v_new[strat]) / v_new[strat]  # fmt: skip
            for i in range(nx):
                if strat[i]:
                    order = np.argsort(-g_new.zc[i])  # сверху вниз
                    c_new[i, order] = stabilize(c_new[i, order], v_new[i, order])
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

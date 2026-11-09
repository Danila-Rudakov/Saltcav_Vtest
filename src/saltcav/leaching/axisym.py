"""Лабораторная установка: горизонтальный канал в монолитном блоке NaCl.

Дистиллят (или рассол) подаётся насосом в один торец канала, рассол выходит из
другого. Канал делится вдоль оси x на ячейки; в узлах хранится радиус r(x, t), в
ячейках — концентрация c(x, t). Модель первой версии (этап 1 руководства по
адаптации): сечение круглое, гравитация на форму не влияет.

Физика
------
* Течение ламинарное. Массоотдача по Левеку для развивающегося профиля
  концентрации, с пределом Грэца для развитого:
  Sh = max(3,66; 1,077·Gz^(1/3))·sh_mult, Gz = Re·Sc·d/x = 4Q/(π·D·x),
  k = Sh·D/d. Вязкость в Gz сокращается, поэтому вязкость рассола
  (в saltcav её нет) на растворение в этой версии не влияет.
* Поток соли со стенки j = k·(C_s − c), отступание стенки V = j/ρ_s.
* Движение стенки и перенос рассола — функции saltcav:
  :func:`~saltcav.leaching.dissolution.wall_growth_rate` и
  :func:`~saltcav.leaching.brine_column.transport_step` (неявная схема против потока,
  концентрация на входе ``c_in``). Плотностное перемешивание по оси не нужно:
  ось горизонтальна.
* Критерий плавучести Gr/Re² = g·(Δρ/ρ)·π²·d⁵/(16·Q²) (вязкость тоже сокращается).
  При Gr/Re² ≳ 1 рассол расслаивается по высоте сечения и осесимметричная модель
  даёт только базовый уровень — это пишется в результат как предупреждение.

Вертикальная модель saltcav этим модулем не затрагивается.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from saltcav.core import brine
from saltcav.core.lab_config import LabConfig
from saltcav.geometry.channel import Channel
from saltcav.leaching.brine_column import transport_step
from saltcav.leaching.dissolution import DissolutionLaw, node_concentration, wall_growth_rate

FloatArray = NDArray[np.float64]

G = 9.81  # м/с²
SH_GRAETZ = 3.66  # развитое течение, постоянная концентрация на стенке
MAX_GROWTH_FRACTION = 0.5  # рост объёма за шаг не больше этой доли подачи
MAX_DR_FRACTION = 0.02  # радиус за шаг растёт не больше чем на 2 % (k ∝ 1/d — явная схема)
MIN_DT = 1e-3  # с


# --- конфигурация -------------------------------------------------------------


def laminar_k(r: FloatArray, x: FloatArray, q: float, diff: float, sh_mult: float,
              x_min: float) -> FloatArray:  # fmt: skip
    """Коэффициент массоотдачи k(x) = Sh·D/d в узлах, м/с."""
    gz = 4.0 * q / (np.pi * diff * np.maximum(x, x_min))
    sh = np.maximum(SH_GRAETZ, 1.077 * gz ** (1.0 / 3.0)) * sh_mult
    return np.asarray(sh * diff / (2.0 * r), dtype=np.float64)


def buoyancy_ratio(d: FloatArray, q: float, drho_rel: FloatArray) -> FloatArray:
    """Gr/Re² = g·(Δρ/ρ)·π²·d⁵/(16·Q²): ≳ 1 — свободная конвекция сильнее потока."""
    return np.asarray(G * drho_rel * np.pi**2 * d**5 / (16.0 * q * q), dtype=np.float64)


# --- расчёт -------------------------------------------------------------------


@dataclass
class LabResult:
    """История прогона. Массивы по времени — первая ось. Единицы СИ."""

    cfg: LabConfig
    x: FloatArray  # узлы вдоль оси, м
    c_sat: float  # кг/м³
    times: list[float] = field(default_factory=list)  # с
    radius: list[FloatArray] = field(default_factory=list)  # (n+1,), м
    concentration: list[FloatArray] = field(default_factory=list)  # (n,), кг/м³
    outlet_concentration: list[float] = field(default_factory=list)  # кг/м³
    volume: list[float] = field(default_factory=list)  # м³
    dissolved: list[float] = field(default_factory=list)  # кг, накоплено
    produced: list[float] = field(default_factory=list)  # кг, вынесено с рассолом
    supplied: list[float] = field(default_factory=list)  # кг, пришло с рассолом на входе
    in_channel: list[float] = field(default_factory=list)  # кг соли в канале
    buoyancy: list[float] = field(default_factory=list)  # max Gr/Re² по каналу
    n_steps: int = 0
    stopped: str = ""  # причина досрочной остановки

    def record(self, t: float, cav: Channel, c: FloatArray, acc: tuple[float, float, float],
               q: float, t_c: float) -> None:  # fmt: skip
        r = cav.radius
        v = cav.cell_volumes()
        self.times.append(t)
        self.radius.append(r.copy())
        self.concentration.append(c.copy())
        self.outlet_concentration.append(float(c[-1]))
        self.volume.append(float(v.sum()))
        self.dissolved.append(acc[0])
        self.produced.append(acc[1])
        self.supplied.append(acc[2])
        self.in_channel.append(float(np.dot(v, c)))
        # у стенки рассол насыщен, в ядре — c: перепад плотности по сечению
        rho = brine.density_from_concentration(np.clip(c, 0.0, self.c_sat), t_c)
        rho_s = float(brine.density_from_concentration(self.c_sat, t_c))
        d = r[:-1] + r[1:]  # средний диаметр ячейки
        self.buoyancy.append(float(np.max(buoyancy_ratio(d, q, (rho_s - rho) / rho))))

    def table(self) -> str:
        """Таблица выхода во времени (для сравнения с опытом)."""
        lines = ["t, ч\tc_вых, г/л\tc_вых/C_s\tобъём, мл\tпотеря массы блока, г\tGr/Re²"]
        for i, t in enumerate(self.times):
            lines.append(
                f"{t / 3600:.3f}\t{self.outlet_concentration[i]:.2f}\t"
                f"{self.outlet_concentration[i] / self.c_sat:.3f}\t{self.volume[i] * 1e6:.3f}\t"
                f"{self.dissolved[i] * 1e3:.3f}\t{self.buoyancy[i]:.3g}"
            )
        return "\n".join(lines)

    def summary(self) -> str:
        r0, r = self.radius[0], self.radius[-1]
        balance = self.dissolved[-1] + self.supplied[-1] - self.produced[-1] - self.in_channel[-1]
        out = [
            f"«{self.cfg.name}»: {self.times[-1] / 3600:g} ч, {self.n_steps} шагов",
            f"  объём канала       {self.volume[0] * 1e6:.3f} → {self.volume[-1] * 1e6:.3f} мл",
            f"  диаметр у входа    {2e3 * r0[0]:.2f} → {2e3 * r[0]:.2f} мм",
            f"  диаметр у выхода   {2e3 * r0[-1]:.2f} → {2e3 * r[-1]:.2f} мм",
            f"  на выходе          {self.outlet_concentration[-1]:.1f} г/л "
            f"({self.outlet_concentration[-1] / self.c_sat:.0%} насыщения)",
            f"  потеря массы блока {self.dissolved[-1] * 1e3:.3f} г",
            f"  баланс соли        невязка {balance * 1e3:.2e} г",
        ]
        gr = max(self.buoyancy)
        if gr >= 1.0:
            out.append(f"  ВНИМАНИЕ: Gr/Re² до {gr:.3g} ≥ 1 — рассол расслаивается по высоте "
                       "сечения, круглое сечение модели — только базовый уровень")  # fmt: skip
        if self.stopped:
            out.append(f"  остановлен: {self.stopped}")
        return "\n".join(out)

    def plot(self, path: str | Path, n_profiles: int = 6) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
        x_mm = self.x * 1e3
        for i in np.unique(np.linspace(0, len(self.times) - 1, n_profiles).round().astype(int)):
            r_mm = self.radius[i] * 1e3
            (line,) = ax1.plot(x_mm, r_mm, label=f"{self.times[i] / 3600:.2f} ч")
            ax1.plot(x_mm, -r_mm, color=line.get_color())
        ax1.set(xlabel="x вдоль канала, мм (вход слева)", ylabel="радиус, мм",
                title=f"{self.cfg.name}: форма канала (продольный разрез)")  # fmt: skip
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        t_h = np.array(self.times) / 3600
        ax2.plot(t_h, np.array(self.outlet_concentration) / self.c_sat)
        ax2.set(xlabel="время, ч", ylabel="c_вых / C_s", ylim=(0, 1.05),
                title="насыщение рассола на выходе")  # fmt: skip
        ax2.grid(alpha=0.3)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def simulate_lab(cfg: LabConfig) -> LabResult:
    """Прогнать весь режим насоса для одного канала."""
    t_c = cfg.temperature_c
    c_sat = float(brine.saturation_concentration(t_c))
    diff = float(brine.diffusivity(t_c))
    cav = Channel.cylinder(cfg.length_mm * 1e-3, 0.5e-3 * cfg.diameter_mm, cfg.n_cells)
    x = cav.x
    dx = float(cav.dz[0])
    c = np.zeros(cav.n_cells)  # канал заполнен дистиллятом
    law = DissolutionLaw(k0=1.0, exponent=1.0, roof_ratio=1.0, floor_ratio=1.0)  # j = k·(C_s − c)
    active = np.ones(cav.n_cells + 1, dtype=bool)  # растворяются все узлы, включая торцы
    rho_s = cfg.salt_density_kg_m3
    r_lim = None if cfg.max_radius_mm is None else 1e-3 * cfg.max_radius_mm
    dt_out = 60.0 * cfg.output_every_min

    res = LabResult(cfg, x, c_sat)
    dissolved = produced = supplied = 0.0
    q0 = cfg.stages[0].flow_ml_min * 1e-6 / 60.0
    res.record(0.0, cav, c, (0.0, 0.0, 0.0), q0, t_c)
    t, next_out, t_end = 0.0, dt_out, 0.0
    for stage in cfg.stages:
        q = stage.flow_ml_min * 1e-6 / 60.0
        c_in = stage.inlet_saturation * c_sat
        t_end += 3600.0 * stage.duration_h
        while t < t_end - 1e-9:
            k = laminar_k(cav.radius, x, q, diff, cfg.sh_mult, 0.5 * dx)
            rate, v_max = wall_growth_rate(
                cav.radius[:, None], cav.dz, node_concentration(c), c_sat, law, rho_s, active,
                k[:, None],
            )  # fmt: skip
            rate = rate[:, 0]
            dt = min(t_end - t, next_out - t)
            if v_max > 0:
                dt = min(dt, cfg.cfl * dx / v_max)
            growth = float(np.max(rate / cav.radius))
            if growth > 0:
                dt = min(dt, MAX_DR_FRACTION / growth)
            v_old = cav.cell_volumes()
            c_cap = np.maximum(c, c_sat) * (1.0 + 1e-9)
            while True:  # как в saltcav: без отрицательного отбора и пересыщения
                new_cav = cav.with_radius(cav.radius + dt * rate)
                v_new = new_cav.cell_volumes()
                dm = rho_s * (v_new - v_old)
                if (float(v_new.sum() - v_old.sum()) <= MAX_GROWTH_FRACTION * q * dt
                        and bool(np.all(v_old * c + dm <= c_cap * v_old))):  # fmt: skip
                    break
                if dt < MIN_DT:
                    raise RuntimeError(f"шаг по времени упал ниже {MIN_DT} с при t = {t:.0f} с")
                dt *= 0.5
            step = transport_step(c, v_old, v_new, dm, q, 0, cav.n_cells - 1, dt, c_in=c_in)
            c, cav = step.c, new_cav
            dissolved += float(dm.sum())
            produced += step.produced_salt
            supplied += dt * q * c_in
            t += dt
            res.n_steps += 1
            broke = r_lim is not None and cav.max_radius() >= r_lim
            if broke:
                res.stopped = f"канал дорос до грани блока на {t / 3600:.2f} ч"
            if broke or t >= next_out - 1e-9 or t >= t_end - 1e-9:
                res.record(t, cav, c, (dissolved, produced, supplied), q, t_c)
                next_out = t + dt_out
            if broke:
                return res
    return res

"""Установка на глубине: трубки до поверхности, давление насоса, запаздывание, ползучесть.

Схема: блок с каналом на глубине H. Насос на поверхности подаёт воду по подающей трубке
вниз; рассол поднимается на поверхность по отводящей трубке и вытекает при атмосферном
давлении (второго насоса нет — сообщающиеся сосуды). Давление в каверне задаёт столб
рассола в отводящей трубке; насос должен преодолеть разницу столбов и трение:

    p_каверны = p_атм + ρ_отв·g·H + Δp_тр,отв,
    p_насоса (избыточное) = (ρ_отв − ρ_под)·g·H + Δp_тр,под + Δp_тр,отв,
    Δp_тр = 128·μ·H·Q/(π·d⁴)  (ламинарное течение в трубке, Хаген—Пуазейль).

Отводящая трубка вначале заполнена водой и заменяется рассолом «поршнем»: рассол из каверны
приходит на поверхность через время V_трубки/Q, и концентрация на поверхности запаздывает.
Если насос остановить, более тяжёлый столб рассола потечёт обратно: самотёком рассол не
поднимается.

Ползучесть: если блок — часть массива под горным давлением σ = ρ_пород·g·H, а в каверне
p_каверны, полость сужается. Оценка — решение для цилиндрической полости в бесконечной
среде со степенной ползучестью (закон BGRa из saltcav, плоская деформация):

    (ṙ/r) = (√3/2)^(n+1) · A·exp(−Q/RT) · (2·(σ − p_каверны)/n)^n.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saltcav.core import brine
from saltcav.core.lab_config import LabConfig
from saltcav.leaching.axisym import G
from saltcav.leaching.section import Props, SectionResult, conditions, simulate_section
from saltcav.mechanics.creep import BGRA, R_GAS

FloatArray = NDArray[np.float64]


@dataclass
class Hydraulics:
    """Трубки и давление во времени (моменты записи результата)."""

    t_h: FloatArray
    p_cavity_mpa: FloatArray  # абсолютное давление в каверне
    p_pump_mpa: FloatArray  # избыточное давление насоса
    p_column_diff_mpa: FloatArray  # (ρ_отв − ρ_под)·g·H
    p_friction_mpa: FloatArray  # трение в обеих трубках
    rho_inlet: float  # средняя плотность в подающей трубке, кг/м³
    rho_outlet: FloatArray  # средняя плотность в отводящей трубке
    c_surface: FloatArray  # концентрация рассола на поверхности, кг/м³
    transit_h: FloatArray  # время подъёма рассола по трубке, ч
    tube_volume_ml: float
    re_tube: float  # число Рейнольдса в трубке (подача)


def hydraulics(res: SectionResult) -> Hydraulics:
    """Давление насоса и в каверне, концентрация на поверхности (поршневое вытеснение)."""
    cfg = res.cfg
    h = cfg.depth_m
    d = 1e-3 * cfg.tube_diameter_mm
    t = np.array(res.times)
    q_in = np.array(res.flow)
    q_out = np.array(res.outflow)
    c_out = np.array(res.outlet_concentration)
    v_tube = 0.25 * np.pi * d * d * h
    # свойства столбов: средняя температура и давление по высоте трубки
    _, p_cav0 = conditions(cfg)
    col = Props.at(0.5 * (cfg.temperature_c + cfg.temperature_at_depth),
                   0.5 * (brine.P_ATM_MPA + p_cav0))  # fmt: skip
    c_in = cfg.stages[0].inlet_saturation * col.c_sat
    rho_in = float(col.rho(c_in))
    mu_in = float(col.nu(c_in)) * rho_in
    # накопленный объём, ушедший из каверны (трапеции)
    v_cum = np.concatenate([[0.0], np.cumsum(0.5 * (q_out[1:] + q_out[:-1]) * np.diff(t))])
    rho_out = np.empty_like(t)
    mu_out = np.empty_like(t)
    c_surf = np.empty_like(t)
    for n in range(t.size):
        if v_tube <= 0:
            rho_out[n], mu_out[n], c_surf[n] = col.rho(c_out[n]), col.nu(c_out[n]), c_out[n]
            mu_out[n] *= rho_out[n]
            continue
        vs = np.linspace(v_cum[n] - v_tube, v_cum[n], 200)
        c_col = np.clip(np.interp(vs, v_cum[: n + 1], c_out[: n + 1], left=0.0), 0, col.c_sat)
        rho_col = col.rho(c_col)
        rho_out[n] = float(rho_col.mean())
        mu_out[n] = float((col.nu(c_col) * rho_col).mean())
        c_surf[n] = c_col[0]  # у поверхности — то, что ушло из каверны V_трубки назад
    fr_in = 128.0 * mu_in * h * q_in / (np.pi * d**4)
    fr_out = 128.0 * mu_out * h * q_out / (np.pi * d**4)
    diff = (rho_out - rho_in) * G * h
    p_cav = brine.P_ATM_MPA + (rho_out * G * h + fr_out) * 1e-6
    re = 4.0 * float(q_in[0]) / (np.pi * d * float(col.nu(c_in))) if h > 0 else 0.0
    transit = np.where(q_out > 0, v_tube / np.maximum(q_out, 1e-30), 0.0) / 3600.0
    return Hydraulics(t / 3600.0, p_cav, (diff + fr_in + fr_out) * 1e-6, diff * 1e-6,
                      (fr_in + fr_out) * 1e-6, rho_in, rho_out, c_surf, transit,
                      v_tube * 1e6, re)  # fmt: skip


@dataclass
class Creep:
    """Оценка сужения каверны ползучестью соли под горным давлением."""

    sigma_rock_mpa: float
    p_cavity_mpa: float
    rate_per_s: float  # ṙ/r
    closure_fraction: float  # доля радиуса за весь опыт


def creep_estimate(res: SectionResult, p_cavity_mpa: float) -> Creep:
    cfg = res.cfg
    sigma = cfg.rock_density_kg_m3 * G * cfg.depth_m * 1e-6 + brine.P_ATM_MPA
    dsig = max(sigma - p_cavity_mpa, 0.0)
    tk = cfg.temperature_at_depth + brine.KELVIN
    n = BGRA.n
    rate = ((np.sqrt(3.0) / 2.0) ** (n + 1) * BGRA.a_per_s
            * np.exp(-BGRA.q_j_per_mol / (R_GAS * tk)) * (2.0 * dsig / n) ** n)  # fmt: skip
    return Creep(sigma, p_cavity_mpa, float(rate), float(rate * res.times[-1]))


def study_depths(cfg: LabConfig) -> list[float]:
    """Глубины для сравнения: от поверхности до заданной (не меньше 1200 м), 5 точек."""
    top = max(cfg.depth_m, 1200.0)
    return [float(x) for x in np.linspace(0.0, top, 5)]


def depth_study(cfg: LabConfig, depths: list[float] | None = None,
                progress: Callable[[float, str], None] | None = None,
                ) -> list[tuple[float, SectionResult]]:  # fmt: skip
    """Тот же опыт на разных глубинах (всё прочее одинаково)."""
    depths = study_depths(cfg) if depths is None else depths
    out = []
    for k, h in enumerate(depths):

        def sub(f: float, text: str, k: int = k, h: float = h) -> None:
            if progress is not None:
                progress((k + f) / len(depths), f"{h:.0f} м: {text}")

        out.append((h, simulate_section(cfg.model_copy(update={"depth_m": h}), sub)))
    return out

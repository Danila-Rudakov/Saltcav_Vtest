"""Установка на глубине: свойства под давлением, трубки, насос, ползучесть."""

from pathlib import Path

import numpy as np
import pytest

from saltcav.core import brine
from saltcav.core.lab_config import LabConfig, load_lab_config
from saltcav.leaching.depth import creep_estimate, hydraulics
from saltcav.leaching.section import Props, conditions, simulate_section

ROOT = Path(__file__).resolve().parents[2]


def _cfg(**kw: object) -> LabConfig:
    base: dict[str, object] = {
        "length_mm": 200, "diameter_mm": 2.5, "n_cells": 30, "n_theta": 10,
        "output_every_min": 2, "stages": [{"duration_h": 0.5, "flow_ml_min": 5.0}],
    }  # fmt: skip
    base.update(kw)
    return LabConfig.model_validate(base)


def _w(m: float) -> float:
    return m * brine.M_NACL / (1.0 + m * brine.M_NACL)


@pytest.mark.parametrize(("m", "mu_35mpa"), [(0.0, 988.2), (0.5, 1035.8)])
def test_viscosity_under_pressure_matches_kestin(m: float, mu_35mpa: float) -> None:
    mu = brine.brine_viscosity(_w(m), 20.0) * brine.viscosity_pressure_factor(_w(m), 20.0, 35.0)
    assert float(mu) * 1e6 == pytest.approx(mu_35mpa, abs=0.3)


def test_kinematic_viscosity_under_pressure_matches_kestin() -> None:
    """Табл. 26 (6 моль/кг, 20 °C): ν = 1,6297 при 0,1 МПа и 1,6479 мм²/с при 35 МПа."""
    w = _w(6.0)
    for p, nu_tab in ((0.1, 1.6297), (35.0, 1.6479)):
        rho = brine.brine_density(w, 20.0) * brine.pressure_density_factor(w, 20.0, p)
        mu = brine.brine_viscosity(w, 20.0) * brine.viscosity_pressure_factor(w, 20.0, p)
        assert float(mu / rho) * 1e6 == pytest.approx(nu_tab, rel=2e-3)


def test_compressibility_order() -> None:
    """Вода сжимается сильнее насыщенного рассола (≈ 0,55 % и 0,35 % на 12 МПа)."""
    water = float(brine.pressure_density_factor(0.0, 20.0, 12.0)) - 1
    sat = float(brine.pressure_density_factor(0.2641, 20.0, 12.0)) - 1
    assert 0.004 < water < 0.007
    assert 0.002 < sat < water


def test_surface_is_unchanged() -> None:
    t, p = conditions(_cfg())
    assert p == pytest.approx(brine.P_ATM_MPA)
    assert Props.at(t, p).c_sat == pytest.approx(float(brine.saturation_concentration(t)), rel=1e-3)


def test_depth_pressure_and_small_effect_without_heat() -> None:
    surf = simulate_section(_cfg())
    deep_cfg = _cfg(depth_m=1200)
    _, p = conditions(deep_cfg)
    assert 11.5 < p < 12.2  # ≈ 0,1 + ρ_воды·g·1200 м
    deep = simulate_section(deep_cfg)
    assert deep.c_sat / surf.c_sat - 1 == pytest.approx(0.006, abs=0.004)
    assert abs(deep.volume[-1] / surf.volume[-1] - 1) < 0.03
    bal = deep.dissolved[-1] + deep.supplied[-1] - deep.produced[-1] - deep.in_channel[-1]
    assert abs(bal) <= 1e-9 * deep.dissolved[-1]


def test_geothermal_gradient_heats_and_speeds_up() -> None:
    cfg = _cfg(depth_m=1200, geothermal_gradient_c_per_100m=2.5)
    assert cfg.temperature_at_depth == pytest.approx(50.0)
    assert simulate_section(cfg).volume[-1] > 1.1 * simulate_section(_cfg(depth_m=1200)).volume[-1]


def test_hydraulics_plug_flow_and_pump() -> None:
    """Короткая трубка: рассол доходит до поверхности; насос ≈ разница столбов + трение."""
    res = simulate_section(_cfg(depth_m=20, tube_diameter_mm=1.0,
                                stages=[{"duration_h": 1.0, "flow_ml_min": 5.0}]))  # fmt: skip
    hy = hydraulics(res)
    assert hy.tube_volume_ml == pytest.approx(0.25 * np.pi * 1e-6 * 20 * 1e6, rel=1e-9)
    assert hy.c_surface[0] == 0.0 and hy.c_surface[-1] > 0.5 * res.outlet_concentration[-1]
    assert np.allclose(hy.p_pump_mpa, hy.p_column_diff_mpa + hy.p_friction_mpa)
    assert hy.p_column_diff_mpa[-1] > 0


def test_long_tube_delays_brine() -> None:
    res = simulate_section(_cfg(depth_m=1200))
    hy = hydraulics(res)
    assert hy.transit_h[-1] > 10
    assert np.all(hy.c_surface == 0.0)


def test_creep_negligible_for_hours() -> None:
    res = simulate_section(_cfg(depth_m=1200))
    cr = creep_estimate(res, float(hydraulics(res).p_cavity_mpa.max()))
    assert cr.sigma_rock_mpa > cr.p_cavity_mpa
    assert 0 < cr.closure_fraction < 1e-4


def test_axisym_only_on_surface() -> None:
    with pytest.raises(ValueError):
        _cfg(model="axisym", natural_convection=False, depth_m=100)


def test_example_config_loads() -> None:
    cfg = load_lab_config(ROOT / "configs" / "lab" / "deep_1200m.yaml")
    assert cfg.depth_m == 1200

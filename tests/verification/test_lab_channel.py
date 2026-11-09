"""Проверки простой модели лабораторного канала."""

from pathlib import Path

import numpy as np
import pytest

from saltcav.core import brine
from saltcav.core.lab_config import LabConfig, load_lab_config
from saltcav.leaching.axisym import buoyancy_ratio, laminar_k, simulate_lab

ROOT = Path(__file__).resolve().parents[2]


def _cfg(**kw: object) -> LabConfig:
    base: dict[str, object] = {
        "length_mm": 200, "diameter_mm": 2.5, "n_cells": 100, "output_every_min": 10,
        "stages": [{"duration_h": 1.0, "flow_ml_min": 1.0}],
    }  # fmt: skip
    base.update(kw)
    return LabConfig.model_validate(base)


def test_salt_balance() -> None:
    """Растворено + пришло на входе = вынесено + осталось в канале."""
    res = simulate_lab(_cfg(stages=[
        {"duration_h": 1.0, "flow_ml_min": 1.0},
        {"duration_h": 0.5, "flow_ml_min": 5.0, "inlet_saturation": 0.3},
    ]))  # fmt: skip
    lhs = res.dissolved[-1] + res.supplied[-1]
    rhs = res.produced[-1] + res.in_channel[-1]
    assert lhs > 0
    assert abs(lhs - rhs) <= 1e-9 * lhs
    # растворённая соль равна потере объёма блока
    assert res.dissolved[-1] == pytest.approx(2160 * (res.volume[-1] - res.volume[0]), rel=1e-9)


def test_profile_matches_analytic() -> None:
    """Неподвижная стенка: c(L) = C_s·(1 − exp(−∫2πr·k(x)dx / Q))."""
    cfg = _cfg(n_cells=400, salt_density_kg_m3=1e12)
    res = simulate_lab(cfg)
    q, r = 1e-6 / 60, 1.25e-3
    diff = float(brine.diffusivity(20.0))
    xs = np.linspace(0.0, 0.2, 200_001)
    k = laminar_k(np.full_like(xs, r), xs, q, diff, 1.0, 1e-9)
    integral = float(np.sum(0.5 * (k[1:] + k[:-1]) * np.diff(xs))) * 2 * np.pi * r
    expected = res.c_sat * (1.0 - np.exp(-integral / q))
    assert res.outlet_concentration[-1] == pytest.approx(expected, rel=0.03)


def test_saturated_inlet_dissolves_nothing() -> None:
    res = simulate_lab(_cfg(stages=[{"duration_h": 1.0, "flow_ml_min": 1.0,
                                     "inlet_saturation": 1.0}]))  # fmt: skip
    c_out = np.array(res.outlet_concentration)
    assert c_out[-1] == pytest.approx(res.c_sat, rel=1e-6)
    # растворить можно не больше, чем примет начальная вода в канале
    assert 0 < res.dissolved[-1] <= res.volume[0] * res.c_sat


def test_inlet_grows_faster_and_high_flow_dilutes() -> None:
    slow = simulate_lab(_cfg())
    fast = simulate_lab(_cfg(stages=[{"duration_h": 1.0, "flow_ml_min": 50.0}]))
    r = slow.radius[-1]
    assert r[0] > r[-1] > 1.25e-3
    assert fast.outlet_concentration[-1] < slow.outlet_concentration[-1]


def test_grid_convergence() -> None:
    v1 = simulate_lab(_cfg(n_cells=50)).volume[-1]
    v2 = simulate_lab(_cfg(n_cells=100)).volume[-1]
    assert v1 == pytest.approx(v2, rel=0.03)


def test_time_step_convergence(monkeypatch: pytest.MonkeyPatch) -> None:
    import saltcav.leaching.axisym as lab

    cfg = _cfg(stages=[{"duration_h": 3.0, "flow_ml_min": 5.0}])
    v1 = simulate_lab(cfg).volume[-1]
    monkeypatch.setattr(lab, "MAX_DR_FRACTION", lab.MAX_DR_FRACTION / 4)
    v2 = simulate_lab(cfg).volume[-1]
    assert v1 == pytest.approx(v2, rel=0.01)


def test_buoyancy_matches_handbook_estimate() -> None:
    """Ручная оценка: d = 2,5 мм, Q = 1 мл/мин, Δρ/ρ = 0,1 → Gr/Re² ≈ 210."""
    assert float(buoyancy_ratio(np.array(2.5e-3), 1e-6 / 60, np.array(0.1))) == pytest.approx(
        210, rel=0.02
    )


def test_stops_at_block_face() -> None:
    res = simulate_lab(_cfg(max_radius_mm=2.0))
    assert res.stopped
    assert max(res.radius[-1]) >= 2.0e-3


def test_example_config_loads() -> None:
    cfg = load_lab_config(ROOT / "configs" / "lab" / "channel.yaml")
    assert cfg.stages[0].inlet_saturation == 0.0

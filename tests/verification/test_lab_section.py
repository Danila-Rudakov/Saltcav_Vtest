"""Проверки модели сечения: канал в блоке и полуканал у окна из оргстекла."""

import numpy as np
import pytest

from saltcav.core.lab_config import LabConfig
from saltcav.leaching.axisym import simulate_lab
from saltcav.leaching.section import simulate_section


def _cfg(**kw: object) -> LabConfig:
    base: dict[str, object] = {
        "length_mm": 200, "diameter_mm": 2.5, "n_cells": 40, "n_theta": 12,
        "stages": [{"duration_h": 1.0, "flow_ml_min": 5.0}],
    }  # fmt: skip
    base.update(kw)
    return LabConfig.model_validate(base)


def _balance(res: object) -> float:
    return res.dissolved[-1] + res.supplied[-1] - res.produced[-1] - res.in_channel[-1]  # type: ignore[attr-defined]


@pytest.mark.parametrize("setup", ["channel", "window"])
def test_salt_balance(setup: str) -> None:
    res = simulate_section(_cfg(setup=setup, stages=[
        {"duration_h": 0.5, "flow_ml_min": 5.0},
        {"duration_h": 0.5, "flow_ml_min": 2.0, "inlet_saturation": 0.4},
    ]))  # fmt: skip
    assert res.dissolved[-1] > 0
    assert abs(_balance(res)) <= 1e-9 * res.dissolved[-1]
    assert sum(res.by_part[-1]) == pytest.approx(res.dissolved[-1], rel=1e-9)


def test_without_gravity_reduces_to_simple_model() -> None:
    """Без гравитации сечение круглое и результат равен простой модели."""
    sec = simulate_section(_cfg(natural_convection=False))
    simple = simulate_lab(_cfg(model="axisym", natural_convection=False))
    r = sec.radius[-1]
    assert np.ptp(r, axis=1).max() <= 1e-12 * r.max()
    assert sec.volume[-1] == pytest.approx(simple.volume[-1], rel=0.02)
    assert sec.outlet_concentration[-1] == pytest.approx(simple.outlet_concentration[-1], rel=0.02)


def test_gravity_roof_faster_than_floor() -> None:
    res = simulate_section(_cfg())
    r = res.radius[-1]
    mid = r.shape[0] // 2
    assert r[mid, 0] > r[mid, r.shape[1] // 2] > r[mid, -1]
    roof, _, floor = res.by_part[-1]
    assert roof > floor
    assert res.stratified[-1] == 1.0


def test_brine_stratified_and_not_supersaturated() -> None:
    res = simulate_section(_cfg())
    c = res.concentration[-1]
    mid = c.shape[0] // 2
    assert c[mid, 0] < c[mid, -1]
    assert c.max() <= res.c_sat * (1 + 1e-6)


def test_window_with_half_flow_mirrors_channel() -> None:
    """Полуканал у окна при Q/2 — половина канала при Q; отличие только от стекла (D_h)."""
    full = simulate_section(_cfg())
    half = simulate_section(_cfg(setup="window", stages=[{"duration_h": 1.0, "flow_ml_min": 2.5}]))
    assert half.volume[-1] == pytest.approx(full.volume[-1] / 2, rel=0.05)
    assert half.outlet_concentration[-1] == pytest.approx(full.outlet_concentration[-1], rel=0.05)


def test_convection_increases_dissolution() -> None:
    forced = simulate_section(_cfg(natural_convection=False))
    mixed = simulate_section(_cfg())
    assert mixed.dissolved[-1] > forced.dissolved[-1]


def test_mechanism_split_sums_to_dissolved() -> None:
    """Вклад вынужденной и свободной конвекции в сумме даёт всё растворённое."""
    res = simulate_section(_cfg(output_every_min=2,
                                stages=[{"duration_h": 1.0, "flow_ml_min": 50.0}]))  # fmt: skip
    forced, natural = res.by_mechanism[-1]
    assert forced + natural == pytest.approx(res.dissolved[-1], rel=1e-9)


def test_no_natural_share_without_convection() -> None:
    res = simulate_section(_cfg(natural_convection=False))
    assert res.by_mechanism[-1][1] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    ("t_c", "m", "mu_table"),  # Kestin и др. (1981), табл. 2, 7, 13 (0,1 МПа), мкПа·с
    [(20.0, 0.5, 1043.2), (25.0, 0.5, 928.6), (20.0, 3.0, 1343.2), (25.0, 3.0, 1199.8),
     (20.0, 6.0, 1950.8), (25.0, 6.0, 1737.5)],
)  # fmt: skip
def test_brine_viscosity_matches_kestin_tables(t_c: float, m: float, mu_table: float) -> None:
    from saltcav.core import brine

    w = m * brine.M_NACL / (1.0 + m * brine.M_NACL)
    assert float(brine.brine_viscosity(w, t_c)) * 1e6 == pytest.approx(mu_table, abs=0.3)


def test_surface_kinetics_limits() -> None:
    """Быстрая реакция → транспортный контроль; медленная — растворение замедляется."""
    base = simulate_section(_cfg())
    fast = simulate_section(_cfg(k_surface_m_s=1e3))
    slow = simulate_section(_cfg(k_surface_m_s=1e-6))
    assert fast.dissolved[-1] == pytest.approx(base.dissolved[-1], rel=1e-3)
    assert slow.dissolved[-1] < 0.5 * base.dissolved[-1]
    assert max(slow.damkohler) > 1 > max(fast.damkohler)


def test_water_balance_and_outflow() -> None:
    """Вода: пришло = ушло + в каверне (точность — объём смешения рассолов, < 1 %)."""
    res = simulate_section(_cfg())
    resid = res.water_in[-1] - res.water_out[-1] - res.water_in_channel[-1]
    assert abs(resid) < 0.01 * res.water_in[-1]
    q_ratio = np.array(res.outflow[1:]) / np.array(res.flow[1:])
    assert np.all((q_ratio > 0.8) & (q_ratio < 1.2))


def test_wall_cooling_small() -> None:
    res = simulate_section(_cfg())
    assert 0 < max(res.wall_cooling) < 5.0


def test_axisym_rejects_window() -> None:
    with pytest.raises(ValueError):
        _cfg(model="axisym", setup="window", natural_convection=False)

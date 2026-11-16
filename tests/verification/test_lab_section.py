"""Проверки модели сечения: канал в блоке, полуканал у окна из оргстекла, полный контур 360°."""

import numpy as np
import pytest

from saltcav.core.lab_config import LabConfig
from saltcav.geometry import geometry, wall_speed_factor
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
    sec = simulate_section(_cfg(natural_convection=False, stefan_condition=False))
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


# ---- режим full: полный контур 0…360° ----


def test_full_contour_is_periodic_and_symmetric() -> None:
    """Однородный материал: контур замкнут (θ=2π ≡ 0) и зеркален относительно вертикали."""
    res = simulate_section(_cfg(setup="full", n_theta=24))
    r = res.radius[-1]
    assert res.theta[-1] == pytest.approx(2.0 * np.pi)
    assert res.halves == 1
    assert np.max(np.abs(r[:, 0] - r[:, -1])) <= 1e-9 * r.max()
    assert np.max(np.abs(r - r[:, ::-1])) <= 1e-6 * r.max()


def test_full_matches_channel() -> None:
    """Без неоднородностей полный контур даёт тот же объём и концентрацию, что половина ×2."""
    kw = {"n_theta": 24}
    half = simulate_section(_cfg(**kw))
    full = simulate_section(_cfg(setup="full", **kw))
    assert full.volume[-1] == pytest.approx(half.volume[-1], rel=0.03)
    assert full.outlet_concentration[-1] == pytest.approx(half.outlet_concentration[-1], rel=0.03)
    roof, _, floor = full.by_part[-1]
    assert roof > floor


def test_full_salt_balance() -> None:
    res = simulate_section(_cfg(setup="full", n_theta=24))
    assert res.dissolved[-1] > 0
    assert abs(_balance(res)) <= 1e-9 * res.dissolved[-1]
    assert sum(res.by_part[-1]) == pytest.approx(res.dissolved[-1], rel=1e-9)


def test_full_height_is_roof_plus_floor() -> None:
    """Высота сечения в режиме 360° — потолок + дно (θ=π), а не потолок + потолок."""
    theta = np.linspace(0.0, 2.0 * np.pi, 25)
    r = np.full((5, 25), 2e-3)
    r[:, 0] = r[:, -1] = 3e-3  # потолок выше дна
    g = geometry(r, 1e-3, theta, window=False)
    assert g.height == pytest.approx(np.full(4, 0.5 * (3e-3 + 3e-3) + 0.5 * (2e-3 + 2e-3)), rel=0.2)
    assert np.all(g.height < 2.0 * 3e-3 - 1e-6)


def test_full_requires_n_theta_multiple_of_4() -> None:
    with pytest.raises(ValueError, match="кратно 4"):
        _cfg(setup="full", n_theta=10)


def test_axisym_rejects_full() -> None:
    with pytest.raises(ValueError, match="axisym"):
        _cfg(model="axisym", setup="full", natural_convection=False, n_theta=12)


def test_wall_speed_factor_periodic_has_no_seam() -> None:
    """Периодическая схема не создаёт шва: гладкое возмущение в районе θ=0 симметрично."""
    theta = np.linspace(0.0, 2.0 * np.pi, 49)
    r = 2e-3 * (1.0 + 0.1 * np.cos(theta))  # гладкий замкнутый контур
    f = wall_speed_factor(np.tile(r, (5, 1)), 1e-3, theta[1] - theta[0], periodic=True)
    assert f[:, 0] == pytest.approx(f[:, -1])
    assert np.max(np.abs(f[:, 1] - f[:, -2])) <= 1e-9


# ---- условие Стефана и множитель материала ----


def test_stefan_condition_speeds_up_wall() -> None:
    """V = j/(ρ_s − C_s) быстрее V = j/ρ_s: каверна больше, но не более чем в ρ_s/(ρ_s − C_s)."""
    on = simulate_section(_cfg(natural_convection=False))
    off = simulate_section(_cfg(natural_convection=False, stefan_condition=False))
    ratio = on.volume[-1] / off.volume[-1]
    assert 1.05 < ratio < 2160.0 / (2160.0 - on.c_sat) * 1.05


def test_material_multiplier_scales_dissolution() -> None:
    """Множитель 1 ничего не меняет, множитель 0,5 растворяет заметно меньше."""
    base = simulate_section(_cfg(setup="full", n_theta=12, natural_convection=False))
    one = simulate_section(
        _cfg(setup="full", n_theta=12, natural_convection=False),
        material_multiplier=lambda r: np.ones_like(r),
    )
    half = simulate_section(
        _cfg(setup="full", n_theta=12, natural_convection=False),
        material_multiplier=lambda r: np.full_like(r, 0.5),
    )
    assert one.volume[-1] == pytest.approx(base.volume[-1], rel=1e-12)
    assert half.volume[-1] < 0.8 * base.volume[-1]


# ---- материал (зёрна, пластины) ----


def test_grain_field_is_normalised_periodic_and_reproducible() -> None:
    from saltcav.leaching.material import grain_field

    a = grain_field(40.0, 4.0, seed=7, contrast=0.2, boundary_factor=1.65)
    b = grain_field(40.0, 4.0, seed=7, contrast=0.2, boundary_factor=1.65)
    c = grain_field(40.0, 4.0, seed=8, contrast=0.2, boundary_factor=1.65)
    assert a.mean() == pytest.approx(1.0)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)
    assert np.array_equal(a[:, 0], a[:, -1])


def test_material_field_is_resolution_independent_and_breaks_symmetry() -> None:
    from saltcav.leaching.material import build_multiplier

    mat = {"seed": 3, "sheet_offset_mm": None}
    coarse = build_multiplier(_cfg(setup="full", n_cells=10, n_theta=24, material=mat))
    fine = build_multiplier(_cfg(setup="full", n_cells=20, n_theta=48, material=mat))
    assert coarse is not None and fine is not None
    r1, r2 = np.full((11, 25), 2e-3), np.full((21, 49), 2e-3)
    m1, m2 = coarse(r1), fine(r2)
    assert m1[::1, ::1].shape == (11, 25)
    assert m1[0, 0] == pytest.approx(m2[0, 0], rel=1e-9)  # один узел — то же значение
    assert np.max(np.abs(m1 - m1[:, ::-1])) > 1e-3  # зёрна несимметричны


def test_material_changes_cavern_and_homogeneous_default_is_none() -> None:
    from saltcav.leaching.material import build_multiplier

    assert build_multiplier(_cfg(setup="full", n_theta=12)) is None
    kw = {"setup": "full", "n_theta": 12, "natural_convection": False}
    plain = simulate_section(_cfg(**kw))
    grainy = simulate_section(_cfg(material={"seed": 5, "grain_contrast": 0.4}, **kw))
    assert grainy.volume[-1] != pytest.approx(plain.volume[-1], rel=1e-4)
    assert grainy.volume[-1] == pytest.approx(plain.volume[-1], rel=0.25)  # заметно, но не вдвое


def test_material_requires_full_setup() -> None:
    with pytest.raises(ValueError, match="только для setup: full"):
        _cfg(setup="channel", material={"seed": 1})


def test_full_yaml_example_loads() -> None:
    from pathlib import Path

    from saltcav.core.lab_config import load_lab_config

    path = Path(__file__).resolve().parents[2] / "configs" / "lab" / "full.yaml"
    cfg = load_lab_config(path)
    assert cfg.setup == "full" and cfg.material is not None and cfg.n_theta % 4 == 0

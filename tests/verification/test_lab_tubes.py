"""Заход трубок в отверстие: карманы x < 0 и x > L, прокачка только между торцами трубок."""

from __future__ import annotations

import numpy as np
import pytest

from saltcav.core.lab_config import LabConfig
from saltcav.geometry import geometry
from saltcav.leaching.material import build_multiplier
from saltcav.leaching.section import simulate_section


def _cfg(**kw: object) -> LabConfig:
    base: dict[str, object] = {
        "setup": "full",
        "n_theta": 12,
        "length_mm": 40.0,
        "diameter_mm": 4.0,
        "n_cells": 16,
        "max_radius_mm": 40.0,
        "output_every_min": 10.0,
        "tube_insert_mm": 5.0,
        "tube_gap_mm": 0.1,
        "stages": [{"duration_h": 0.4, "flow_ml_min": 1.25}],
    }
    base.update(kw)
    return LabConfig.model_validate(base)


def test_grid_extends_to_pockets() -> None:
    res = simulate_section(_cfg())
    assert res.n_pocket == 2  # dx = 2,5 мм, заход 5 мм
    assert res.x[0] == pytest.approx(-5e-3)
    assert res.x[-1] == pytest.approx(45e-3)
    assert res.radius[-1].shape == (16 + 2 * 2 + 1, 13)


def test_no_tubes_keeps_working_grid() -> None:
    res = simulate_section(_cfg(tube_insert_mm=0.0))
    assert res.n_pocket == 0
    assert res.x[0] == 0.0 and res.x[-1] == pytest.approx(40e-3)
    assert res.pocket_dissolved[-1] == 0.0


@pytest.mark.parametrize("mult", [1.0, 100.0])
def test_salt_balance_with_pockets(mult: float) -> None:
    res = simulate_section(_cfg(pocket_exchange_mult=mult))
    balance = res.supplied[-1] + res.dissolved[-1] - res.produced[-1] - res.in_channel[-1]
    assert abs(balance) <= 1e-9 * res.dissolved[-1]
    assert 0.0 < res.pocket_dissolved[-1] < res.dissolved[-1]


def test_stronger_exchange_dissolves_pockets_deeper() -> None:
    weak = simulate_section(_cfg(pocket_exchange_mult=1.0))
    strong = simulate_section(_cfg(pocket_exchange_mult=100.0))
    assert strong.pocket_dissolved[-1] > weak.pocket_dissolved[-1]
    # глубина: радиус у потолка на торце кармана (x = −заход) не меньше, чем при слабом обмене
    assert strong.radius[-1][1, 0] >= weak.radius[-1][1, 0]


def test_full_contour_with_pockets_stays_symmetric() -> None:
    res = simulate_section(_cfg())
    r = res.radius[-1]
    assert np.max(np.abs(r - r[:, ::-1])) <= 1e-6 * r.max()


def test_tube_displaces_volume_and_adds_wetted_perimeter() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 25)
    r = np.full((5, 25), 2e-3)
    dx = 1e-3
    plain = geometry(r, dx, theta, window=False)
    tube_r = np.full(4, 1.9e-3)
    with_tube = geometry(r, dx, theta, window=False, tube_r=tube_r)
    assert with_tube.area == pytest.approx(plain.area - np.pi * 1.9e-3**2, rel=1e-9)
    assert with_tube.perim == pytest.approx(plain.perim + 2.0 * np.pi * 1.9e-3, rel=1e-9)
    assert np.all(with_tube.vol > 0)


def test_material_grid_covers_pockets() -> None:
    cfg = _cfg(material={"seed": 1, "sheet_offset_mm": 25.0})
    mult = build_multiplier(cfg)
    assert mult is not None
    m = mult(np.full((16 + 2 * cfg.n_pocket + 1, 13), 2e-3))
    assert m.shape == (21, 13) and np.all(np.isfinite(m))


def test_tube_validation() -> None:
    with pytest.raises(ValueError, match="только для model: section"):
        _cfg(model="axisym", setup="channel", natural_convection=False, n_theta=12)
    with pytest.raises(ValueError, match="меньше радиуса"):
        _cfg(tube_gap_mm=2.5)

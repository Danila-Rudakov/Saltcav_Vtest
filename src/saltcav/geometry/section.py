"""Геометрия сечения лабораторного канала r(θ, x): объёмы клиньев, нормали, периметр.

Угол θ отсчитывается от вертикали вверх: θ = 0 — потолок, θ = 90° — бок, θ = 180° — дно.
Считается половина сечения θ ∈ [0, π]; для ``setup: window`` к смоченному периметру
добавляется окно из оргстекла. Для ``setup: full`` считается полный контур θ ∈ [0, 2π]
(последний узел замыкается на первый, угловая производная — периодическая). Движение
стенки — схема Годунова (см. модель сечения, :mod:`saltcav.leaching.section`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class Geometry:
    """Геометрия сечения канала на момент времени (массивы по ячейкам (nx, nθ)).

    Для ``channel``/``window`` это половина канала, для ``full`` — весь контур 0…2π.
    """

    vol: FloatArray  # объём клина, м³
    nz: FloatArray  # вертикальная составляющая нормали «в соль»: +1 потолок, −1 дно
    zc: FloatArray  # высота центра клина над осью, м
    area: FloatArray  # (nx,) площадь расчётной части сечения (половина или весь контур), м²
    perim: FloatArray  # (nx,) смоченный периметр расчётной части (с окном для window), м
    height: FloatArray  # (nx,) высота сечения (потолок + дно), м


def geometry(
    r: FloatArray,
    dx: float,
    theta: FloatArray,
    window: bool,
    tube_r: FloatArray | None = None,
) -> Geometry:
    """Геометрия по форме стенки r (nx+1, nθ+1).

    ``tube_r`` (nx,) — наружный радиус трубки в ячейках по x (0 — трубки нет): трубка в центре
    сечения вытесняет объём (клин теряет ½·r_т²·dθ·dx), а её поверхность добавляется к
    смоченному периметру (гидравлический диаметр кольцевого зазора ≈ 2·ширина зазора).
    """
    rm = 0.5 * (r[:-1] + r[1:])  # радиус в середине ячейки по x, (nx, nθ+1)
    dth = float(theta[1] - theta[0])
    vol = dx * dth / 2.0 * 0.25 * (r[:-1, :-1] ** 2 + r[1:, :-1] ** 2 + r[:-1, 1:] ** 2
                                   + r[1:, 1:] ** 2)  # fmt: skip
    y, z = rm * np.sin(theta), rm * np.cos(theta)
    ty, tz = np.diff(y, axis=1), np.diff(z, axis=1)
    seg = np.hypot(ty, tz)
    nz = ty / np.where(seg > 0, seg, 1.0)  # нормаль (−t_z, t_y)/|t|
    th_mid = 0.5 * (theta[:-1] + theta[1:])
    zc = 2.0 / 3.0 * 0.5 * (rm[:, :-1] + rm[:, 1:]) * np.cos(th_mid)
    # Высота сечения = потолок + дно. В режиме 360° последний узел по θ — снова потолок,
    # поэтому дно берём в узле θ=π (середина массива); в половинной модели — последний узел.
    full = bool(np.isclose(theta[-1], 2.0 * np.pi))
    floor_idx = (rm.shape[1] - 1) // 2 if full else -1
    height = rm[:, 0] + rm[:, floor_idx]
    perim = seg.sum(axis=1) + (height if window else 0.0)
    if tube_r is not None:
        vol = vol - (0.5 * dx * dth * tube_r**2)[:, None]
        perim = perim + float(theta[-1]) * tube_r
    return Geometry(vol, nz, zc, vol.sum(axis=1) / dx, perim, height)


def _godunov_sq(pm: FloatArray, pp: FloatArray) -> FloatArray:
    """p² для ∂r/∂t = V·√(1+p²): max на [p⁻, p⁺] при p⁻ ≤ p⁺, иначе min."""
    sq_max = np.maximum(pm * pm, pp * pp)
    straddle = (np.minimum(pm, pp) < 0) & (np.maximum(pm, pp) > 0)
    sq_min = np.where(straddle, 0.0, np.minimum(pm * pm, pp * pp))
    return np.asarray(np.where(pm <= pp, sq_max, sq_min), dtype=np.float64)


def wall_speed_factor(r: FloatArray, dx: float, dth: float, periodic: bool = False) -> FloatArray:
    """√(1 + (r_θ/r)² + r_x²) в узлах; торцы — без наклона.

    По θ: зеркало на окне/плоскости симметрии (половина сечения) либо периодическое
    замыкание (``periodic=True``, полный контур 0…2π, где узел θ=2π совпадает с θ=0).
    """
    if periodic:
        core = r[:, :-1]  # уникальные узлы: последний (θ=2π) дублирует первый
        prev, nxt = np.roll(core, 1, axis=1), np.roll(core, -1, axis=1)
        dm_core, dp_core = (core - prev) / dth, (nxt - core) / dth
        dm_t = np.concatenate([dm_core, dm_core[:, :1]], axis=1)  # дубликат — копия первого
        dp_t = np.concatenate([dp_core, dp_core[:, :1]], axis=1)
    else:
        rt = np.pad(r, ((0, 0), (1, 1)), mode="reflect")  # θ: зеркало
        dm_t, dp_t = (rt[:, 1:-1] - rt[:, :-2]) / dth, (rt[:, 2:] - rt[:, 1:-1]) / dth
    rx = np.pad(r, ((1, 1), (0, 0)), mode="edge")  # x: нулевой наклон за торцами
    dm_x, dp_x = (rx[1:-1] - rx[:-2]) / dx, (rx[2:] - rx[1:-1]) / dx
    return np.asarray(np.sqrt(1.0 + _godunov_sq(dm_t, dp_t) / (r * r) + _godunov_sq(dm_x, dp_x)),
                      dtype=np.float64)  # fmt: skip

"""Круглый канал r(x) на осевой сетке: радиус в узлах, объёмы ячеек.

Заменяет вертикальную ``Cavern`` (сетка «глубина × азимут»): простой модели нужны только
узлы вдоль оси, радиус в узлах и объёмы ячеек. Между соседними узлами стенка — усечённый
конус, объём ячейки V_i = π·Δx/3 · (r_i² + r_i·r_{i+1} + r_{i+1}²).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Channel:
    """Форма канала в один момент времени. Изменение формы — новый объект."""

    x: FloatArray  # узлы вдоль оси, м, (n+1,)
    radius: FloatArray  # радиус в узлах, м, (n+1,)

    def __post_init__(self) -> None:
        x = np.array(self.x, dtype=float)
        r = np.array(self.radius, dtype=float)  # копия: объект не зависит от входного массива
        if x.ndim != 1 or x.size < 2:
            raise ValueError("нужно минимум два узла по оси")
        if np.any(np.diff(x) <= 0):
            raise ValueError("узлы x должны строго возрастать")
        if r.shape != x.shape:
            raise ValueError(f"форма r {r.shape}, ожидалось {x.shape}")
        if np.any(~np.isfinite(r)) or np.any(r < 0):
            raise ValueError("r должен быть конечным и неотрицательным")
        x.setflags(write=False)
        r.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "radius", r)

    @classmethod
    def cylinder(cls, length: float, radius: float, n_cells: int) -> Channel:
        """Начальное состояние: просверленный цилиндрический канал."""
        return cls(np.linspace(0.0, length, n_cells + 1), np.full(n_cells + 1, float(radius)))

    def with_radius(self, radius: FloatArray) -> Channel:
        return Channel(self.x, radius)

    @property
    def n_cells(self) -> int:
        return self.x.size - 1

    @property
    def dz(self) -> FloatArray:
        """Длины ячеек вдоль оси, м (имя как у сетки saltcav)."""
        return np.asarray(np.diff(self.x), dtype=np.float64)

    def cell_volumes(self) -> FloatArray:
        """Объём каждой ячейки, м³ (усечённый конус)."""
        r1, r2 = self.radius[:-1], self.radius[1:]
        return np.asarray(np.pi * self.dz / 3.0 * (r1 * r1 + r1 * r2 + r2 * r2), dtype=np.float64)

    def volume(self) -> float:
        return float(self.cell_volumes().sum())

    def max_radius(self) -> float:
        return float(np.max(self.radius))

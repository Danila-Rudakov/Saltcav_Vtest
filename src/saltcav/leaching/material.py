"""Неоднородность материала блока для модели сечения полного контура (``setup: full``).

Множитель скорости растворения M(x, θ) в узлах сетки (nx+1, nθ+1), среднее по полю равно 1:

* **зёрна** — случайная мозаика Вороного (шаг ~2 мм) на развёртке стенки (x, s = r₀·θ): границы
  зёрен растворяются быстрее (``boundary_factor``), контраст между зёрнами задаёт
  ``grain_contrast``, плюс мелкий шум. Поле генерируется на фиксированной базовой сетке
  24 × 72 и интерполируется на сетку расчёта, поэтому одно и то же ``seed`` даёт одну и ту же
  картину при любом разрешении;
* **наклонные пластины** — слабые слои, наклонённые к оси (смещение ``sheet_offset_mm`` на длину
  образца): ядро пластины растворяется чуть медленнее (0,97), граница — быстрее (1,08);
  множитель зависит от положения точки стенки в пространстве, а не от угла, поэтому
  пересчитывается по текущей форме r(θ, x).

Это синтетическая модель материала для численных экспериментов, а не измеренное поле свойств.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from saltcav.core.lab_config import LabConfig

FloatArray = NDArray[np.float64]

BASE_NX, BASE_NTH = 24, 72  # базовая сетка зёрен: ячеек по x и по θ (0…2π)
SHEET_CLIP = (0.90, 1.12)  # допустимый диапазон множителя пластин


def grain_field(
    length_mm: float,
    diameter_mm: float,
    seed: int,
    contrast: float,
    boundary_factor: float,
    boundary_width_mm: float = 0.16,
    grain_pitch_mm: float = 2.0,
    micro_sigma: float = 0.07,
) -> FloatArray:
    """Множитель зёрен на базовой сетке узлов (BASE_NX+1, BASE_NTH+1), среднее равно 1."""
    rng = np.random.default_rng(seed)
    nx, nt = BASE_NX, BASE_NTH
    x = np.linspace(0.0, length_mm, nx + 1)
    theta = np.linspace(0.0, 2.0 * np.pi, nt + 1)
    r0 = diameter_mm / 2.0
    s_len = 2.0 * np.pi * r0  # длина окружности исходного канала
    xx, ss = np.meshgrid(x, r0 * theta, indexing="ij")

    seeds_list: list[list[float]] = []
    for x0 in np.arange(-5, length_mm + 6, grain_pitch_mm):
        for s0 in np.arange(-5, s_len + 6, grain_pitch_mm):
            pitch = grain_pitch_mm * np.exp(rng.uniform(np.log(0.45), np.log(2.5)))
            seeds_list.append(
                [
                    x0 + rng.uniform(-0.35, 0.35) * pitch,
                    (s0 + rng.uniform(-0.35, 0.35) * pitch) % s_len,
                ]
            )
    seeds = np.asarray(seeds_list)
    shift = np.array([0.0, s_len])
    tiled = np.vstack([seeds - shift, seeds, seeds + shift])  # периодичность по s
    tree = cKDTree(tiled)
    pts = np.column_stack([xx.ravel(), ss.ravel()])
    dist2, idx = tree.query(pts, k=2)
    d1, d2 = dist2[:, 0].reshape(xx.shape), dist2[:, 1].reshape(xx.shape)
    grain_id = idx[:, 0].reshape(xx.shape) % len(seeds)
    to_boundary = 0.5 * np.maximum(d2 - d1, 0.0)
    boundary = np.exp(-0.5 * (to_boundary / boundary_width_mm) ** 2)
    grain_values = np.exp(contrast * rng.standard_normal(len(seeds)) - 0.5 * contrast**2)
    resolved = grain_values[grain_id] * (0.82 + (boundary_factor - 0.82) * boundary)

    noise = rng.standard_normal(resolved.shape)
    smooth = (
        4 * noise
        + np.roll(noise, 1, 0)
        + np.roll(noise, -1, 0)
        + np.roll(noise, 1, 1)
        + np.roll(noise, -1, 1)
    ) / 8.0
    micro = np.exp(micro_sigma * smooth - 0.5 * micro_sigma**2)
    mult = resolved * micro
    mult[:, -1] = mult[:, 0]  # θ=2π ≡ θ=0
    return np.asarray(mult / mult.mean(), dtype=np.float64)


class InclinedSheets:
    """Слоистость: наклонные пластины разной толщины, ядро медленнее, границы быстрее."""

    def __init__(
        self,
        offset_mm: float,
        length_mm: float,
        min_t: float = 0.0005,
        max_t: float = 4.0,
        core_factor: float = 0.97,
        boundary_factor: float = 1.08,
        boundary_fraction: float = 0.035,
        extent_mm: float = 100.0,
    ) -> None:
        self.offset_mm = float(offset_mm)
        self.length_mm = float(length_mm)
        seq = np.geomspace(min_t, max_t, 48)
        seq = np.r_[seq, seq[::-1]]  # толщины от тонких к толстым и обратно
        edges = [-extent_mm]
        vals: list[float] = []
        i = 0
        while edges[-1] < extent_mm:
            t = float(seq[i % len(seq)])
            a = edges[-1]
            b = min(a + t, extent_mm)
            t = b - a
            bw = min(0.025, max(0.00002, boundary_fraction * t))
            if t > 2 * bw:
                vals.extend([boundary_factor, core_factor, boundary_factor])
                edges.extend([a + bw, b - bw, b])
            else:
                vals.append(core_factor)
                edges.append(b)
            i += 1
        self.edges = np.asarray(edges, dtype=np.float64)
        self.values = np.asarray(vals, dtype=np.float64)
        widths = np.diff(self.edges)
        self.cum = np.r_[0.0, np.cumsum(widths * self.values)]
        mean = self.cum[-1] / (self.edges[-1] - self.edges[0])  # среднее поля — 1
        self.values /= mean
        self.cum /= mean

    def primitive(self, q: FloatArray) -> FloatArray:
        """Первообразная кусочно-постоянного поля вдоль координаты пластин q, мм."""
        q = np.clip(q, self.edges[0], self.edges[-1])
        k = np.clip(np.searchsorted(self.edges, q, side="right") - 1, 0, len(self.values) - 1)
        return np.asarray(self.cum[k] + (q - self.edges[k]) * self.values[k], dtype=np.float64)

    def evaluate(self, x_mm: FloatArray, theta: FloatArray, r_m: FloatArray) -> FloatArray:
        """Множитель в узлах (nx+1, nθ+1) для формы стенки r(θ, x), м: среднее по ячейке."""
        xx = np.broadcast_to(x_mm[:, None], r_m.shape)
        rr = r_m * 1000.0
        th = theta[None, :]
        q = rr * np.sin(th) + self.offset_mm * xx / self.length_mm  # координата поперёк пластин
        dth = 2.0 * np.pi / (r_m.shape[1] - 1)
        dx = self.length_mm / (r_m.shape[0] - 1)
        w_ang = np.abs(rr * np.cos(th)) * dth  # проекция шага по θ на ось q
        w_ax = abs(self.offset_mm / self.length_mm) * dx  # проекция шага по x
        half = 0.5 * np.maximum(np.sqrt(w_ang * w_ang + w_ax * w_ax), 0.0005)
        field = (self.primitive(q + half) - self.primitive(q - half)) / (2.0 * half)
        field[:, -1] = field[:, 0]
        return np.asarray(np.clip(field, *SHEET_CLIP), dtype=np.float64)


def build_multiplier(cfg: LabConfig) -> Callable[[FloatArray], FloatArray] | None:
    """Множитель скорости растворения M(r) по ``cfg.material``; ``None`` — материал однороден."""
    mat = cfg.material
    if mat is None:
        return None
    x_mm = np.linspace(0.0, cfg.length_mm, cfg.n_cells + 1)
    theta = np.linspace(0.0, 2.0 * np.pi, cfg.n_theta + 1)

    base = grain_field(cfg.length_mm, cfg.diameter_mm, mat.seed, mat.grain_contrast,
                       mat.boundary_factor)  # fmt: skip
    x_base = np.linspace(0.0, cfg.length_mm, BASE_NX + 1)
    th_base = np.linspace(0.0, 2.0 * np.pi, BASE_NTH + 1)
    xg, tg = np.meshgrid(x_mm, theta, indexing="ij")
    interp = RegularGridInterpolator((x_base, th_base), base, bounds_error=False, fill_value=None)
    grains = np.asarray(interp(np.c_[xg.ravel(), tg.ravel()]).reshape(xg.shape), dtype=np.float64)
    grains[:, -1] = grains[:, 0]

    sheets = (
        None if mat.sheet_offset_mm is None else InclinedSheets(mat.sheet_offset_mm, cfg.length_mm)
    )

    def multiplier(r: FloatArray) -> FloatArray:
        if sheets is None:
            return grains
        return np.asarray(grains * sheets.evaluate(x_mm, theta, r), dtype=np.float64)

    return multiplier

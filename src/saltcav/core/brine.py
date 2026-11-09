"""Свойства воды и рассола NaCl.

Соглашения
----------
* ``w`` — массовая доля NaCl в растворе, кг/кг (0 … w_sat).
* ``C`` — массовая концентрация, кг NaCl на м³ раствора: ``C = w · ρ(w, T)``.
  В модели выщелачивания баланс соли ведётся именно в ``C``.
* ``T`` — температура, °C (на границе ввода); внутри функций переводится в К,
  где это нужно.

Источники и точность
--------------------
* Плотность воды и растворимость NaCl — табличные значения справочника CRC
  (Handbook of Chemistry and Physics), линейная интерполяция.
* Плотность рассола — таблица CRC для 20 °C по массовой доле, пересчитанная на
  другую температуру множителем ρ_w(T)/ρ_w(20 °C). Это приближение: погрешность
  растёт с удалением от 20 °C (порядка десятых долей процента в диапазоне
  10–60 °C). Влияние давления не учитывается.
* Вязкость воды — уравнение Фогеля; коэффициент диффузии — пересчёт по
  Стоксу–Эйнштейну от опорного значения.
* Вязкость рассола — корреляция Kestin, Khalifa, Correia (1981), ±0,5 %
  (:func:`brine_viscosity`); вертикальная модель её пока не использует.

Таблицы — кандидаты на сверку при получении лабораторных данных компании.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

KELVIN = 273.15

# Плотность чистой воды при 1 атм, кг/м³ (CRC).
_T_WATER = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
_RHO_WATER = np.array(
    [999.84, 999.70, 998.21, 995.65, 992.22, 988.04, 983.20, 977.76, 971.79, 965.31, 958.35]
)

# Растворимость NaCl, массовая доля насыщенного раствора (CRC).
_T_SOL = np.array([0, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
_W_SAT = (
    np.array([26.28, 26.32, 26.41, 26.45, 26.52, 26.67, 26.84, 27.03, 27.25, 27.50, 27.78, 28.05])
    / 100.0
)

# Плотность раствора NaCl при 20 °C в зависимости от массовой доли, кг/м³ (CRC).
_W_TAB = np.array([0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,
                   0.20, 0.22, 0.24, 0.26])  # fmt: skip
_RHO20_TAB = np.array([998.21, 1005.3, 1012.5, 1026.8, 1041.3, 1055.9, 1070.7, 1085.7,
                       1100.9, 1116.2, 1131.9, 1147.8, 1164.0, 1180.4, 1197.2])  # fmt: skip

# Вязкость воды, уравнение Фогеля: μ = A·10^(B/(T−C)), T в К.
_VOGEL_A = 2.414e-5  # Па·с
_VOGEL_B = 247.8  # К
_VOGEL_C = 140.0  # К

# Коэффициент диффузии NaCl в воде при 25 °C (разбавленный раствор), м²/с.
D_NACL_25C = 1.61e-9


def _check_temperature(t: FloatArray) -> None:
    if np.any(t < 0.0) or np.any(t > 100.0):
        raise ValueError("температура вне диапазона таблиц 0–100 °C")


def water_density(t_c: ArrayLike) -> FloatArray:
    """Плотность чистой воды, кг/м³."""
    t = np.asarray(t_c, dtype=float)
    _check_temperature(t)
    return np.asarray(np.interp(t, _T_WATER, _RHO_WATER), dtype=np.float64)


def saturation_mass_fraction(t_c: ArrayLike) -> FloatArray:
    """Массовая доля NaCl в насыщенном растворе."""
    t = np.asarray(t_c, dtype=float)
    _check_temperature(t)
    return np.asarray(np.interp(t, _T_SOL, _W_SAT), dtype=np.float64)


def brine_density(w: ArrayLike, t_c: ArrayLike = 20.0) -> FloatArray:
    """Плотность рассола по массовой доле, кг/м³.

    Допускается небольшой выход за таблицу (до 0.30) линейной экстраполяцией —
    растворимость при 100 °C чуть выше последней табличной точки 0.26.
    """
    w_arr = np.asarray(w, dtype=float)
    t = np.asarray(t_c, dtype=float)
    _check_temperature(t)
    if np.any(w_arr < 0.0) or np.any(w_arr > 0.30):
        raise ValueError("массовая доля вне диапазона 0–0.30")
    slope = (_RHO20_TAB[-1] - _RHO20_TAB[-2]) / (_W_TAB[-1] - _W_TAB[-2])
    rho20 = np.where(
        w_arr <= _W_TAB[-1],
        np.interp(w_arr, _W_TAB, _RHO20_TAB),
        _RHO20_TAB[-1] + slope * (w_arr - _W_TAB[-1]),
    )
    return np.asarray(rho20 * water_density(t) / _RHO_WATER[2], dtype=np.float64)


def concentration_from_mass_fraction(w: ArrayLike, t_c: ArrayLike = 20.0) -> FloatArray:
    """Массовая доля → концентрация C, кг/м³."""
    w_arr = np.asarray(w, dtype=float)
    return w_arr * brine_density(w_arr, t_c)


def mass_fraction_from_concentration(c: ArrayLike, t_c: float = 20.0) -> FloatArray:
    """Концентрация C, кг/м³ → массовая доля (обратная функция по таблице)."""
    c_arr = np.asarray(c, dtype=float)
    w_grid = np.linspace(0.0, 0.30, 3001)
    c_grid = concentration_from_mass_fraction(w_grid, t_c)
    if np.any(c_arr < 0.0) or np.any(c_arr > c_grid[-1]):
        raise ValueError("концентрация вне допустимого диапазона")
    return np.asarray(np.interp(c_arr, c_grid, w_grid), dtype=np.float64)


def saturation_concentration(t_c: ArrayLike) -> FloatArray:
    """Концентрация насыщения C_s(T), кг/м³."""
    t = np.asarray(t_c, dtype=float)
    return concentration_from_mass_fraction(saturation_mass_fraction(t), t)


def density_from_concentration(c: ArrayLike, t_c: float = 20.0) -> FloatArray:
    """Плотность рассола по концентрации C, кг/м³ (для стратификации)."""
    return brine_density(mass_fraction_from_concentration(c, t_c), t_c)


def water_viscosity(t_c: ArrayLike) -> FloatArray:
    """Динамическая вязкость чистой воды, Па·с (уравнение Фогеля)."""
    t = np.asarray(t_c, dtype=float)
    _check_temperature(t)
    tk = t + KELVIN
    return _VOGEL_A * 10.0 ** (_VOGEL_B / (tk - _VOGEL_C))


M_NACL = 0.0584428  # кг/моль

# Kestin, Khalifa, Correia (1981), J. Phys. Chem. Ref. Data 10(1), 71–87, ур. (2)–(5):
# ±0,5 % при 20–150 °C, 0–6 моль/кг, 0,1 МПа (влияние давления до 35 МПа < 1 %, не учитывается).
_KESTIN_ALPHA = (1.2378, -1.303e-3, 3.06e-6, 2.55e-8)  # вода, ур. (3)
_KESTIN_MU20 = 1002.0e-6  # Па·с, вода при 20 °C
_KESTIN_A = (3.324e-2, 3.624e-3, -1.879e-4)  # ур. (4)
_KESTIN_B = (-3.96e-2, 1.02e-2, -7.02e-4)  # ур. (5)


# Влияние давления на вязкость (Kestin и др., 1981, ур. (1), (6)–(10)): μ = μ⁰·(1 + β·p),
# p в ГПа; β = β_s^E·β*(m/m_s) + β_w.
_KESTIN_BETA_W = (-1.297, 5.74e-2, -6.97e-4, 4.47e-6, -1.05e-8)  # ур. (7), ГПа⁻¹
_KESTIN_GAMMA = (0.545, 2.8e-3)  # ур. (8)
_KESTIN_MS = (6.044, 2.8e-3, 3.6e-5)  # ур. (9), моль/кг
_KESTIN_BSTAR = (2.5, -2.0, 0.5)  # ур. (10)

# Удельный объём раствора NaCl от давления, Rowe, Chou (1970), J. Chem. Eng. Data 15, 61,
# в записи Kestin и др. (1981), ур. (11a)–(14): v = A − B·p − C·p² + w·D + w²·E − w·F·p −
# w²·G·p − ½·H·p², p в МПа, T в К, v в м³/кг (0–150 °C, до 35 МПа).
_RC_A = (1.006741e2, -1.127522, 5.916365e-3, -1.035794e-5, 9.270048e-9)  # степени T: −2…2
_RC_B = (1.042948, -1.1933677e-2, 5.307535e-5, -1.0688768e-7, 8.492739e-11)  # −2…2
_RC_C = (1.23268e-9, -6.861928e-12)  # степени T: 0…1
_RC_D = (-2.5166e-3, 1.11766e-5, -1.70552e-8)  # 0…2
_RC_E = (2.84851e-3, -1.54305e-5, 2.23982e-8)
_RC_F = (-1.5106e-5, 8.4605e-8, -1.2715e-10)
_RC_G = (2.7676e-5, -1.5694e-7, 2.3102e-10)
_RC_H = (6.4633e-8, -4.1671e-10, 6.8599e-13)
P_ATM_MPA = 0.101325

# Растворимость NaCl растёт с давлением: Sawamura и др. (2007) — 6,61 моль/кг при 0,3 ГПа
# против ≈ 6,15 при 0,1 МПа (25 °C), в среднем d ln m_s/dp ≈ 2,5·10⁻⁴ МПа⁻¹. Та же величина
# следует из термодинамики (V̄_NaCl в насыщенном растворе меньше молярного объёма галита).
SOLUBILITY_PRESSURE_COEF = 2.5e-4  # 1/МПа; оценка, до ~200 МПа почти линейно


def _poly(coef: tuple[float, ...], x: FloatArray, start: int = 0) -> FloatArray:
    return np.asarray(sum(c * x ** (i + start) for i, c in enumerate(coef)), dtype=np.float64)


def specific_volume_rowe_chou(w: ArrayLike, t_c: ArrayLike, p_mpa: ArrayLike) -> FloatArray:
    """Удельный объём рассола NaCl, м³/кг, по Rowe и Chou (1970)."""
    w_arr = np.asarray(w, dtype=float)
    tk = np.asarray(t_c, dtype=float) + KELVIN
    p = np.asarray(p_mpa, dtype=float)
    return np.asarray(
        _poly(_RC_A, tk, -2) - _poly(_RC_B, tk, -2) * p - _poly(_RC_C, tk) * p**2
        + w_arr * _poly(_RC_D, tk) + w_arr**2 * _poly(_RC_E, tk)
        - w_arr * _poly(_RC_F, tk) * p - w_arr**2 * _poly(_RC_G, tk) * p
        - 0.5 * _poly(_RC_H, tk) * p**2,
        dtype=np.float64,
    )  # fmt: skip


def pressure_density_factor(w: ArrayLike, t_c: ArrayLike, p_mpa: ArrayLike) -> FloatArray:
    """ρ(p)/ρ(0,1 МПа) для рассола: сжимаемость по Rowe и Chou, база — таблица CRC."""
    return np.asarray(
        specific_volume_rowe_chou(w, t_c, P_ATM_MPA) / specific_volume_rowe_chou(w, t_c, p_mpa),
        dtype=np.float64,
    )


def viscosity_pressure_factor(w: ArrayLike, t_c: ArrayLike, p_mpa: ArrayLike) -> FloatArray:
    """μ(p)/μ⁰ для рассола NaCl (Kestin и др., 1981, ур. (1), (6)–(10))."""
    t = np.asarray(t_c, dtype=float)
    m = molality_from_mass_fraction(w)
    beta_w = _poly(_KESTIN_BETA_W, t)
    beta_se = _KESTIN_GAMMA[0] + _KESTIN_GAMMA[1] * t - beta_w
    beta_star = _poly(_KESTIN_BSTAR, m / _poly(_KESTIN_MS, t), 1)
    beta = beta_se * beta_star + beta_w  # ГПа⁻¹
    return np.asarray(1.0 + beta * np.asarray(p_mpa, dtype=float) * 1e-3, dtype=np.float64)


def molality_from_mass_fraction(w: ArrayLike) -> FloatArray:
    """Массовая доля NaCl → моляльность, моль/кг воды."""
    w_arr = np.asarray(w, dtype=float)
    return np.asarray(w_arr / ((1.0 - w_arr) * M_NACL), dtype=np.float64)


def brine_viscosity(w: ArrayLike, t_c: ArrayLike = 20.0) -> FloatArray:
    """Динамическая вязкость рассола NaCl, Па·с (Kestin и др., 1981).

    log₁₀(μ/μ_w) = A(m) + B(m)·log₁₀(μ_w/μ_w(20 °C)), A = Σaᵢmⁱ, B = Σbᵢmⁱ;
    log₁₀(μ_w/μ_w(20 °C)) = Σαᵢ(20 − t)ⁱ/(96 + t). Ниже 20 °C — экстраполяция.
    """
    t = np.asarray(t_c, dtype=float)
    _check_temperature(t)
    m = molality_from_mass_fraction(w)
    lw = sum(a * (20.0 - t) ** (i + 1) for i, a in enumerate(_KESTIN_ALPHA)) / (96.0 + t)
    a_m = sum(a * m ** (i + 1) for i, a in enumerate(_KESTIN_A))
    b_m = sum(b * m ** (i + 1) for i, b in enumerate(_KESTIN_B))
    return np.asarray(_KESTIN_MU20 * 10.0 ** (lw + a_m + b_m * lw), dtype=np.float64)


def diffusivity(t_c: ArrayLike) -> FloatArray:
    """Коэффициент диффузии NaCl, м²/с: пересчёт D(25 °C) по Стоксу–Эйнштейну D ∝ T/μ."""
    t = np.asarray(t_c, dtype=float)
    tk = t + KELVIN
    ref_t = 25.0 + KELVIN
    return D_NACL_25C * (tk / ref_t) * (water_viscosity(25.0) / water_viscosity(t))

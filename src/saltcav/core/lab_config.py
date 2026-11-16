"""Конфигурация лабораторной установки: канал в блоке NaCl, режим насоса, глубина.

Строгая схема (как остальные конфигурации saltcav): неизвестные поля отвергаются.
Описание параметров — docs/lab/README.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

from saltcav.core.config import _Strict


class LabStage(_Strict):
    """Ступень режима насоса."""

    duration_h: Annotated[float, Field(gt=0, description="Длительность ступени, ч")]
    flow_ml_min: Annotated[float, Field(gt=0, description="Расход насоса, мл/мин")]
    inlet_saturation: Annotated[
        float, Field(ge=0, le=1, description="Насыщение на входе, доля C_s; 0 — дистиллят")
    ] = 0.0


class LabMaterial(_Strict):
    """Синтетическая неоднородность материала блока (только ``setup: full``)."""

    seed: Annotated[int, Field(ge=0, description="Зерно генератора случайных зёрен")] = 20261112
    grain_contrast: Annotated[
        float, Field(ge=0, le=1, description="Контраст скорости растворения между зёрнами")
    ] = 0.20
    boundary_factor: Annotated[
        float, Field(gt=0, description="Скорость растворения на границах зёрен относительно ядра")
    ] = 1.65
    sheet_offset_mm: Annotated[
        float | None,
        Field(description="Наклонные пластины: смещение слоёв на длину образца, мм; "
                          "не задано — без пластин"),
    ] = 25.0  # fmt: skip


class LabConfig(_Strict):
    """Параметры установки. Всё, что можно поменять в опыте, задаётся здесь."""

    name: Annotated[str, Field(description="Название опыта")] = "lab-channel"
    setup: Annotated[
        Literal["channel", "window", "full"],
        Field(description="channel — канал в целом блоке (считается половина, вторая зеркальная); "
                          "window — полуканал у вертикального окна из оргстекла (половина "
                          "каверны, видна через стекло); full — канал в целом блоке, весь "
                          "контур 0…360° без допущения о симметрии"),
    ] = "channel"  # fmt: skip
    length_mm: Annotated[float, Field(gt=0, description="Длина канала (блока вдоль оси), мм")]
    diameter_mm: Annotated[float, Field(gt=0, description="Начальный диаметр канала, мм")]
    max_radius_mm: Annotated[
        float | None,
        Field(gt=0, description="От оси канала до ближайшей грани блока, мм; расчёт "
                                "останавливается, когда канал до неё дорастёт"),
    ] = None  # fmt: skip
    temperature_c: Annotated[
        float, Field(ge=0, le=100, description="Температура, °C (при заданном геотермическом "
                                               "градиенте — температура на поверхности)"),
    ] = 20.0  # fmt: skip
    salt_density_kg_m3: Annotated[float, Field(gt=0, description="Плотность соли блока, кг/м³")] = (
        2160.0
    )
    depth_m: Annotated[
        float, Field(ge=0, le=2800, description="Глубина установки от поверхности, м: вода "
                                                "подаётся насосом с поверхности по одной трубке, "
                                                "рассол поднимается на поверхность по другой "
                                                "(0 — установка на поверхности)"),
    ] = 0.0  # fmt: skip
    tube_diameter_mm: Annotated[
        float, Field(gt=0, description="Внутренний диаметр подающей и отводящей трубок, мм "
                                       "(при глубине > 0)"),
    ] = 4.0  # fmt: skip
    geothermal_gradient_c_per_100m: Annotated[
        float | None,
        Field(ge=0, description="Геотермический градиент, °C на 100 м: температура блока и "
                                "рассола на глубине = температура на поверхности + "
                                "градиент·глубина. Не задан — температура одна и та же"),
    ] = None  # fmt: skip
    rock_density_kg_m3: Annotated[
        float, Field(gt=0, description="Средняя плотность пород над установкой, кг/м³ "
                                       "(горное давление — для оценки ползучести соли)"),
    ] = 2300.0  # fmt: skip
    model: Annotated[
        Literal["section", "axisym"],
        Field(description="section — сечение r(θ, x) с конвекцией и расслоением; "
                          "axisym — простая модель с круглым сечением r(x)"),
    ] = "section"  # fmt: skip
    natural_convection: Annotated[
        bool, Field(description="Учитывать свободную конвекцию и расслоение рассола")
    ] = True
    tube_insert_mm: Annotated[
        float,
        Field(ge=0, description="Заход подающей и отводящей трубок в отверстие, мм. Каверна "
                                "растворяется от x = −заход до x = L + заход, а вода входит в "
                                "x = 0 и выходит в x = L (трубка непроницаема). 0 — трубок нет"),
    ] = 0.0  # fmt: skip
    tube_gap_mm: Annotated[
        float,
        Field(gt=0, description="Радиальный зазор между трубкой и стенкой отверстия, мм "
                                "(наружный радиус трубки = D₀/2 − зазор)"),
    ] = 0.1  # fmt: skip
    pocket_exchange_mult: Annotated[
        float,
        Field(ge=0, description="Множитель обмена рассолом между карманом у трубки и "
                                "рабочим участком (1 — молекулярная диффузия через зазор; "
                                "калибруется по опытам)"),
    ] = 1.0  # fmt: skip
    stefan_condition: Annotated[
        bool,
        Field(description="Условие Стефана на границе соли: скорость отступания стенки "
                          "V = j/(ρ_s − C_s) вместо V = j/ρ_s (быстрее в ρ_s/(ρ_s − C_s) ≈ 1,17 "
                          "раза). Только модель сечения; выключено — прежняя скорость"),
    ] = True  # fmt: skip
    sh_mult: Annotated[
        float, Field(gt=0, description="Калибровочный множитель массоотдачи (по опытам)")
    ] = 1.0
    roof_mult: Annotated[
        float, Field(ge=0, description="Множитель массоотдачи свободной конвекции у потолка")
    ] = 1.0
    wall_mult: Annotated[
        float, Field(ge=0, description="Множитель массоотдачи свободной конвекции у боковых стенок")
    ] = 1.0
    ri_crit: Annotated[
        float, Field(gt=0, description="Gr/Re², выше которого рассол в сечении расслаивается")
    ] = 1.0
    k_surface_m_s: Annotated[
        float | None,
        Field(gt=0, description="Константа скорости реакции на поверхности соли, м/с: поток "
                                "j = (C_s − c)/(1/k_массоотдачи + 1/k_реакции). Не задана — "
                                "растворение ограничено массоотдачей (транспортный контроль)"),
    ] = None  # fmt: skip
    salt_conductivity_w_mk: Annotated[
        float, Field(gt=0, description="Теплопроводность соли блока, Вт/(м·К), для оценки "
                                       "охлаждения стенки теплотой растворения"),
    ] = 6.0  # fmt: skip
    material: Annotated[
        LabMaterial | None,
        Field(description="Неоднородность материала (зёрна, пластины); не задано — однородный. "
                          "Только setup: full"),
    ] = None  # fmt: skip
    n_cells: Annotated[int, Field(ge=4, description="Число ячеек вдоль оси")] = 100
    n_theta: Annotated[
        int,
        Field(
            ge=4,
            description="Секторов по углу на половину сечения (full — на весь контур "
            "0…360°, кратно 4)",
        ),
    ] = 18
    output_every_min: Annotated[float, Field(gt=0, description="Шаг записи, мин")] = 10.0
    cfl: Annotated[float, Field(gt=0, le=1, description="Запас по условию Куранта")] = 0.5
    stages: Annotated[list[LabStage], Field(min_length=1, description="Режим насоса")]

    @model_validator(mode="after")
    def _check_model(self) -> LabConfig:
        if self.model == "axisym" and (self.setup != "channel" or self.natural_convection):
            raise ValueError("model: axisym — только setup: channel и natural_convection: false")
        if self.tube_insert_mm > 0 and self.model != "section":
            raise ValueError("tube_insert_mm — только для model: section")
        if self.tube_gap_mm >= 0.5 * self.diameter_mm:
            raise ValueError("tube_gap_mm должен быть меньше радиуса канала")
        if self.material is not None and self.setup != "full":
            raise ValueError("material — только для setup: full (поле задано на всём контуре 0…2π)")
        if self.setup == "full" and self.n_theta % 4:
            raise ValueError("setup: full — n_theta кратно 4 (узлы бока θ=90° и дна θ=180°)")
        if self.model == "axisym" and self.depth_m > 0:
            raise ValueError("model: axisym — только для установки на поверхности (depth_m: 0)")
        if self.temperature_at_depth > 100:
            raise ValueError("температура на глубине выше 100 °C — вне таблиц свойств рассола")
        return self

    @property
    def n_pocket(self) -> int:
        """Ячеек кармана у каждой трубки (заход трубки в целых ячейках по x, 0 — без трубок)."""
        if self.tube_insert_mm <= 0:
            return 0
        return max(1, round(self.tube_insert_mm / (self.length_mm / self.n_cells)))

    @property
    def temperature_at_depth(self) -> float:
        """Температура блока и рассола на глубине установки, °C."""
        grad = self.geothermal_gradient_c_per_100m or 0.0
        return self.temperature_c + grad * self.depth_m / 100.0


def load_lab_config(path: str | Path) -> LabConfig:
    with open(path, encoding="utf-8") as f:
        return LabConfig.model_validate(yaml.safe_load(f))


# --- физика -------------------------------------------------------------------

"""Растворение стенки и перенос рассола.

Физика Сергея Кизреда (:mod:`~saltcav.leaching.dissolution`,
:mod:`~saltcav.leaching.brine_column`) и модели лабораторного канала, построенные на ней:

* :mod:`~saltcav.leaching.axisym` — простая модель, круглое сечение r(x);
* :mod:`~saltcav.leaching.section` — модель сечения r(θ, x): конвекция, расслоение рассола;
* :mod:`~saltcav.leaching.depth` — установка на глубине: трубки, давление насоса, ползучесть.
"""

from saltcav.leaching.brine_column import TransportResult, column_step, stabilize, transport_step
from saltcav.leaching.dissolution import DissolutionLaw, node_concentration, wall_growth_rate

__all__ = [
    "DissolutionLaw",
    "TransportResult",
    "column_step",
    "node_concentration",
    "stabilize",
    "transport_step",
    "wall_growth_rate",
]

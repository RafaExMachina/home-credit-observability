"""Contratos e objetos de valor para detecção de drift."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True, slots=True)
class DriftResult:
    """Representa o resultado de um teste de drift em uma variável."""

    feature: str
    method: str
    statistic: float
    threshold: float
    drift_detected: bool
    p_value: float | None = None

    def to_dict(self) -> dict[str, str | float | bool | None]:
        """Converte o resultado em um dicionário serializável."""
        return asdict(self)


class DriftDetector(Protocol):
    """Interface implementada por detectores univariados de drift."""

    def detect(
        self,
        reference: pd.Series,
        current: pd.Series,
        *,
        feature: str,
    ) -> DriftResult:
        """Compara duas amostras e devolve o resultado do teste."""
        ...

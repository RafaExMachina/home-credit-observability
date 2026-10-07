"""Contratos dos componentes de observabilidade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


class EventLogger(Protocol):
    """Contrato para persistência de eventos estruturados."""

    def log(
        self,
        event: str,
        *,
        level: str = "INFO",
        **attributes: Any,
    ) -> None:
        """Registra um evento e seus atributos."""
        ...


class MetricsExporter(Protocol):
    """Contrato para exportadores de métricas operacionais."""

    def export(self, metrics: Mapping[str, float], output_path: Path) -> Path:
        """Persiste métricas em um formato consumível externamente."""
        ...


class DashboardRenderer(Protocol):
    """Contrato para renderização de dashboards."""

    def render(
        self,
        metrics: Mapping[str, float],
        drift_results: Sequence[Mapping[str, Any]],
        performance: Mapping[str, Any],
        alerts: Sequence[Any],
        output_path: Path,
    ) -> Path:
        """Gera um dashboard com métricas, drift e alertas."""
        ...

"""Políticas de alerta para métricas de saúde e degradação."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class MonitoringAlert:
    """Representa uma violação de uma política de monitoramento."""

    code: str
    severity: str
    message: str
    observed: float
    threshold: float

    def to_dict(self) -> dict[str, str | float]:
        """Converte o alerta em um dicionário serializável."""
        return asdict(self)


class AlertPolicyEvaluator:
    """Avalia métricas utilizando limites explícitos e auditáveis."""

    def __init__(
        self,
        *,
        max_null_ratio: float = 0.0,
        max_latency_p95_ms: float = 200.0,
        min_recall_delta: float = -0.05,
        min_roc_auc: float = 0.80,
    ) -> None:
        """Configura os limites operacionais do modelo."""
        self.max_null_ratio = max_null_ratio
        self.max_latency_p95_ms = max_latency_p95_ms
        self.min_recall_delta = min_recall_delta
        self.min_roc_auc = min_roc_auc

    def evaluate(self, metrics: Mapping[str, float]) -> list[MonitoringAlert]:
        """Retorna um alerta para cada política violada."""
        alerts: list[MonitoringAlert] = []
        self._append_if(
            alerts,
            metrics["prediction_null_ratio"] > self.max_null_ratio,
            "PREDICTION_NULL_RATIO",
            "critical",
            "Existem predições nulas no lote de produção.",
            metrics["prediction_null_ratio"],
            self.max_null_ratio,
        )
        self._append_if(
            alerts,
            metrics["prediction_latency_p95_ms"] > self.max_latency_p95_ms,
            "INFERENCE_LATENCY_P95",
            "warning",
            "A latência p95 excedeu o SLO definido.",
            metrics["prediction_latency_p95_ms"],
            self.max_latency_p95_ms,
        )
        self._append_if(
            alerts,
            metrics["drifted_features_count"] > 0,
            "DATA_DRIFT_DETECTED",
            "warning",
            "Uma ou mais variáveis apresentaram data drift.",
            metrics["drifted_features_count"],
            0.0,
        )
        self._append_if(
            alerts,
            metrics["model_recall_delta"] < self.min_recall_delta,
            "MODEL_RECALL_DEGRADATION",
            "critical",
            "A queda de recall ultrapassou o limite permitido.",
            metrics["model_recall_delta"],
            self.min_recall_delta,
        )
        self._append_if(
            alerts,
            metrics["model_roc_auc"] < self.min_roc_auc,
            "MODEL_ROC_AUC_LOW",
            "critical",
            "O ROC AUC de produção está abaixo do mínimo permitido.",
            metrics["model_roc_auc"],
            self.min_roc_auc,
        )
        return alerts

    @staticmethod
    def _append_if(
        alerts: list[MonitoringAlert],
        condition: bool,
        code: str,
        severity: str,
        message: str,
        observed: float,
        threshold: float,
    ) -> None:
        """Acrescenta um alerta somente quando a condição é verdadeira."""
        if condition:
            alerts.append(MonitoringAlert(code, severity, message, observed, threshold))

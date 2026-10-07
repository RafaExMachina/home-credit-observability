"""Cálculo e exportação de métricas de saúde do modelo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, write_to_textfile

from home_credit_observability.drift.prediction import PredictionService


class PredictionLatencyMonitor:
    """Mede a latência de inferência por registro em pequenos lotes."""

    def __init__(self, predictor: PredictionService, *, batch_size: int = 256) -> None:
        """Recebe o serviço de predição e o tamanho dos lotes medidos."""
        if batch_size < 1:
            raise ValueError("batch_size deve ser positivo")
        self.predictor = predictor
        self.batch_size = batch_size

    def measure(self, dataframe: pd.DataFrame) -> list[float]:
        """Executa inferências e retorna latências em milissegundos por registro."""
        if dataframe.empty:
            raise ValueError("o dataset de produção não pode ser vazio")
        latencies: list[float] = []
        for start in range(0, len(dataframe), self.batch_size):
            batch = dataframe.iloc[start : start + self.batch_size]
            started_at = perf_counter()
            self.predictor.predict(batch)
            elapsed_ms = (perf_counter() - started_at) * 1_000
            latencies.append(elapsed_ms / len(batch))
        return latencies


class HealthMetricsCollector:
    """Consolida métricas de predição, drift, desempenho e latência."""

    def collect(
        self,
        production: pd.DataFrame,
        drift_results: Sequence[Mapping[str, Any]],
        performance: Mapping[str, Any],
        latency_samples_ms: Sequence[float],
        *,
        pipeline_duration_seconds: float,
    ) -> dict[str, float]:
        """Produz o conjunto padronizado de métricas operacionais."""
        if not latency_samples_ms:
            raise ValueError("ao menos uma amostra de latência é necessária")
        prediction = production["prediction"]
        probability = production["prediction_probability"]
        drifted_features = {
            str(result["feature"])
            for result in drift_results
            if bool(result["drift_detected"])
        }
        drift_alerts = sum(bool(result["drift_detected"]) for result in drift_results)
        current_metrics = performance["production"]
        delta = performance["delta"]
        latency = np.asarray(latency_samples_ms, dtype=float)
        return {
            "prediction_count": float(len(production)),
            "prediction_null_ratio": float(prediction.isna().mean()),
            "prediction_positive_ratio": float(prediction.mean()),
            "prediction_confidence_mean": float(probability.mean()),
            "prediction_latency_p50_ms": float(np.percentile(latency, 50)),
            "prediction_latency_p95_ms": float(np.percentile(latency, 95)),
            "prediction_latency_p99_ms": float(np.percentile(latency, 99)),
            "drifted_features_count": float(len(drifted_features)),
            "drift_alerts_count": float(drift_alerts),
            "model_recall": float(current_metrics["recall"]),
            "model_recall_delta": float(delta["recall"]),
            "model_roc_auc": float(current_metrics["roc_auc"]),
            "pipeline_duration_seconds": float(pipeline_duration_seconds),
        }


class PrometheusTextExporter:
    """Exporta gauges no formato de exposição de texto do Prometheus."""

    def __init__(self, *, namespace: str = "home_credit") -> None:
        """Define o namespace usado como prefixo das métricas."""
        self.namespace = namespace

    def export(self, metrics: Mapping[str, float], output_path: Path) -> Path:
        """Grava as métricas em um arquivo `.prom`."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        registry = CollectorRegistry()
        for name, value in metrics.items():
            gauge = Gauge(
                name,
                f"Home Credit observability metric: {name}",
                namespace=self.namespace,
                registry=registry,
            )
            gauge.set(value)
        write_to_textfile(str(output_path), registry)
        return output_path

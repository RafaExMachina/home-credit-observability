"""Pipeline da Etapa 3 para observabilidade e monitoramento."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from home_credit_observability.config import MODELS_DIR, PROJECT_ROOT
from home_credit_observability.drift.prediction import PredictionService
from home_credit_observability.observability.alerts import AlertPolicyEvaluator
from home_credit_observability.observability.dashboard import PlotlyHealthDashboard
from home_credit_observability.observability.logger import JsonLinesEventLogger
from home_credit_observability.observability.metrics import (
    HealthMetricsCollector,
    PredictionLatencyMonitor,
    PrometheusTextExporter,
)
from home_credit_observability.pipelines.drift_pipeline import run_drift_pipeline

DriftRunner = Callable[..., dict[str, Any]]


def _read_json(path: str | Path) -> Any:
    """Lê um documento JSON em UTF-8."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _status(alerts: Sequence[Any]) -> str:
    """Calcula o estado global a partir das severidades dos alertas."""
    severities = {alert.severity for alert in alerts}
    if "critical" in severities:
        return "critical"
    if severities:
        return "warning"
    return "healthy"


def run_monitoring_pipeline(
    *,
    sample_size: int = 8_000,
    random_state: int = 42,
    monitoring_dir: Path | None = None,
    log_path: Path | None = None,
    model_path: Path | None = None,
    drift_runner: DriftRunner = run_drift_pipeline,
    latency_samples_ms: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Executa drift, centraliza logs, exporta métricas e cria o dashboard."""
    started_at = perf_counter()
    monitoring_dir = monitoring_dir or PROJECT_ROOT / "reports" / "monitoring"
    log_path = log_path or PROJECT_ROOT / "reports" / "logs" / "observability.jsonl"
    model_path = model_path or MODELS_DIR / "baseline_pipeline.joblib"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonLinesEventLogger(log_path)
    logger.log(
        "monitoring_started",
        sample_size=sample_size,
        random_state=random_state,
    )

    drift_summary = drift_runner(sample_size=sample_size, random_state=random_state)
    production = pd.read_parquet(drift_summary["production_path"])
    drift_results = _read_json(drift_summary["statistical_results_path"])
    performance = _read_json(drift_summary["performance_comparison_path"])
    if latency_samples_ms is None:
        latency_samples_ms = PredictionLatencyMonitor(
            PredictionService(model_path)
        ).measure(production)

    metrics = HealthMetricsCollector().collect(
        production,
        drift_results,
        performance,
        latency_samples_ms,
        pipeline_duration_seconds=perf_counter() - started_at,
    )
    alerts = AlertPolicyEvaluator().evaluate(metrics)
    status = _status(alerts)
    metrics_path = monitoring_dir / "health_metrics.json"
    alerts_path = monitoring_dir / "alerts.json"
    prometheus_path = monitoring_dir / "prometheus_metrics.prom"
    dashboard_path = monitoring_dir / "observability_dashboard.html"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    alerts_path.write_text(
        json.dumps(
            [alert.to_dict() for alert in alerts],
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    PrometheusTextExporter().export(metrics, prometheus_path)
    PlotlyHealthDashboard().render(
        metrics,
        drift_results,
        performance,
        alerts,
        dashboard_path,
    )
    logger.log(
        "monitoring_completed",
        status=status,
        alert_count=len(alerts),
        drifted_features=drift_summary["drifted_features"],
        metrics=metrics,
    )
    for alert in alerts:
        logger.log("alert_triggered", level=alert.severity, **alert.to_dict())
    return {
        "status": status,
        "alert_count": len(alerts),
        "metrics_path": str(metrics_path),
        "alerts_path": str(alerts_path),
        "prometheus_metrics_path": str(prometheus_path),
        "dashboard_path": str(dashboard_path),
        "log_path": str(log_path),
        "metrics": metrics,
    }

"""Testes das métricas e da exportação Prometheus."""

import pandas as pd

from home_credit_observability.observability.metrics import (
    HealthMetricsCollector,
    PrometheusTextExporter,
)


def test_health_metrics_collector_consolidates_sources(tmp_path):
    """O coletor deve combinar predição, drift, desempenho e latência."""
    production = pd.DataFrame(
        {"prediction": [0, 1, 1], "prediction_probability": [0.1, 0.8, 0.9]}
    )
    drift = [{"feature": "income", "drift_detected": True}]
    performance = {
        "production": {"recall": 0.70, "roc_auc": 0.84},
        "delta": {"recall": -0.08},
    }
    metrics = HealthMetricsCollector().collect(
        production,
        drift,
        performance,
        [1.0, 2.0, 3.0],
        pipeline_duration_seconds=4.0,
    )
    output = PrometheusTextExporter().export(metrics, tmp_path / "metrics.prom")

    assert metrics["prediction_count"] == 3
    assert metrics["prediction_null_ratio"] == 0
    assert metrics["drifted_features_count"] == 1
    assert "home_credit_model_recall" in output.read_text(encoding="utf-8")

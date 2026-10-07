"""Teste de integração do pipeline de monitoramento."""

import json
from pathlib import Path

import pandas as pd

from home_credit_observability.pipelines.monitoring_pipeline import (
    run_monitoring_pipeline,
)


def test_monitoring_pipeline_creates_all_observability_artifacts(tmp_path):
    """O pipeline deve centralizar dados e gerar os cinco entregáveis."""
    production_path = tmp_path / "production.parquet"
    drift_path = tmp_path / "drift.json"
    performance_path = tmp_path / "performance.json"
    pd.DataFrame(
        {"prediction": [0, 1, 1], "prediction_probability": [0.2, 0.8, 0.9]}
    ).to_parquet(production_path, index=False)
    drift = [
        {
            "feature": "person_income",
            "method": "psi",
            "statistic": 0.31,
            "drift_detected": True,
        }
    ]
    performance = {
        "reference": {
            "accuracy": 0.8,
            "precision": 0.8,
            "recall": 0.8,
            "f1_score": 0.8,
            "roc_auc": 0.9,
        },
        "production": {
            "accuracy": 0.8,
            "precision": 0.8,
            "recall": 0.7,
            "f1_score": 0.75,
            "roc_auc": 0.86,
        },
        "delta": {"recall": -0.1},
    }
    drift_path.write_text(json.dumps(drift), encoding="utf-8")
    performance_path.write_text(json.dumps(performance), encoding="utf-8")

    def fake_drift_runner(**_):
        return {
            "production_path": str(production_path),
            "statistical_results_path": str(drift_path),
            "performance_comparison_path": str(performance_path),
            "drifted_features": ["person_income"],
        }

    result = run_monitoring_pipeline(
        monitoring_dir=tmp_path / "monitoring",
        log_path=tmp_path / "logs" / "events.jsonl",
        drift_runner=fake_drift_runner,
        latency_samples_ms=[1.0, 2.0],
    )

    assert result["status"] == "critical"
    for key in (
        "metrics_path",
        "alerts_path",
        "prometheus_metrics_path",
        "dashboard_path",
        "log_path",
    ):
        assert tmp_path in Path(result[key]).parents
        assert Path(result[key]).exists()

"""Testes das políticas de alerta."""

from home_credit_observability.observability.alerts import AlertPolicyEvaluator


def test_alert_policy_detects_drift_and_recall_degradation():
    """Drift e queda excessiva de recall devem produzir alertas."""
    metrics = {
        "prediction_null_ratio": 0.0,
        "prediction_latency_p95_ms": 5.0,
        "drifted_features_count": 3.0,
        "model_recall_delta": -0.09,
        "model_roc_auc": 0.86,
    }

    alerts = AlertPolicyEvaluator().evaluate(metrics)

    assert {alert.code for alert in alerts} == {
        "DATA_DRIFT_DETECTED",
        "MODEL_RECALL_DEGRADATION",
    }
    assert any(alert.severity == "critical" for alert in alerts)

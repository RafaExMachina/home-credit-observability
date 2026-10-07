"""Testes do dashboard Plotly."""

from home_credit_observability.observability.alerts import MonitoringAlert
from home_credit_observability.observability.dashboard import PlotlyHealthDashboard


def test_dashboard_generates_self_contained_html(tmp_path):
    """O dashboard deve conter gráficos e tabela de alertas."""
    metrics = {"prediction_null_ratio": 0.0, "prediction_latency_p95_ms": 2.0}
    drift = [
        {
            "feature": "person_income",
            "method": "psi",
            "statistic": 0.4,
            "drift_detected": True,
        }
    ]
    scores = {
        name: 0.8 for name in ("accuracy", "precision", "recall", "f1_score", "roc_auc")
    }
    performance = {"reference": scores, "production": scores}
    alerts = [MonitoringAlert("DRIFT", "warning", "Drift detectado", 1.0, 0.0)]

    path = PlotlyHealthDashboard().render(
        metrics, drift, performance, alerts, tmp_path / "dashboard.html"
    )
    html = path.read_text(encoding="utf-8")

    assert "plotly" in html.lower()
    assert "Alertas ativos" in html
    assert "Drift detectado" in html

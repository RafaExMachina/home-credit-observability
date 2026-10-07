"""Dashboard HTML interativo para a saúde do sistema de crédito."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from home_credit_observability.observability.alerts import MonitoringAlert


class PlotlyHealthDashboard:
    """Renderiza métricas, desempenho, drift e alertas em um único HTML."""

    def render(
        self,
        metrics: Mapping[str, float],
        drift_results: Sequence[Mapping[str, Any]],
        performance: Mapping[str, Any],
        alerts: Sequence[MonitoringAlert],
        output_path: Path,
    ) -> Path:
        """Gera um relatório HTML autocontido e retorna o caminho criado."""
        figure = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}], [{}, {}]],
            subplot_titles=(
                "Predições nulas",
                "Latência p95",
                "PSI por variável",
                "Desempenho do modelo",
            ),
        )
        figure.add_trace(
            go.Indicator(
                mode="number",
                value=metrics["prediction_null_ratio"] * 100,
                number={"suffix": "%", "valueformat": ".2f"},
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Indicator(
                mode="number",
                value=metrics["prediction_latency_p95_ms"],
                number={"suffix": " ms", "valueformat": ".3f"},
            ),
            row=1,
            col=2,
        )
        psi_results = [result for result in drift_results if result["method"] == "psi"]
        figure.add_trace(
            go.Bar(
                x=[result["feature"] for result in psi_results],
                y=[result["statistic"] for result in psi_results],
                marker_color=[
                    "#ef4444" if result["drift_detected"] else "#22c55e"
                    for result in psi_results
                ],
                name="PSI",
            ),
            row=2,
            col=1,
        )
        metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        for dataset, color in (("reference", "#2563eb"), ("production", "#f97316")):
            figure.add_trace(
                go.Bar(
                    x=metric_names,
                    y=[performance[dataset][name] for name in metric_names],
                    name=dataset.capitalize(),
                    marker_color=color,
                ),
                row=2,
                col=2,
            )
        figure.update_layout(
            title="Home Credit — Observabilidade e Monitoramento",
            template="plotly_white",
            height=850,
            barmode="group",
        )
        rows = "".join(
            "<tr>"
            f"<td>{escape(alert.severity)}</td><td>{escape(alert.code)}</td>"
            f"<td>{escape(alert.message)}</td><td>{alert.observed:.4f}</td>"
            f"<td>{alert.threshold:.4f}</td></tr>"
            for alert in alerts
        )
        if not rows:
            rows = '<tr><td colspan="5">Nenhum alerta ativo.</td></tr>'
        alert_table = (
            "<h2>Alertas ativos</h2><table><thead><tr>"
            "<th>Severidade</th><th>Código</th><th>Mensagem</th>"
            "<th>Observado</th><th>Limite</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
        html = figure.to_html(full_html=False, include_plotlyjs=True)
        document = (
            "<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
            "<title>Home Credit Observability</title>"
            "<style>body{font-family:Arial;margin:24px;color:#172033}"
            "table{border-collapse:collapse;width:100%}th,td{padding:10px;"
            "border:1px solid #d8dee9;text-align:left}th{background:#eef2ff}"
            "</style></head><body>"
            f"{html}{alert_table}</body></html>"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document, encoding="utf-8")
        return output_path

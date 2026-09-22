"""Pipeline da Etapa 2 para simulação, predição e diagnóstico de drift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from home_credit_observability.config import MODELS_DIR, PROJECT_ROOT, REFERENCE_DIR
from home_credit_observability.data.prepare import MODEL_FEATURES, NUMERIC_FEATURES
from home_credit_observability.drift.detector import DriftDetectionService
from home_credit_observability.drift.ks import KolmogorovSmirnovDetector
from home_credit_observability.drift.prediction import PredictionService
from home_credit_observability.drift.psi import PSIDetector
from home_credit_observability.drift.simulator import ProductionDataSimulator
from home_credit_observability.models.evaluate import evaluate_binary_classifier
from home_credit_observability.reporting.evidently_report import EvidentlyDriftReport
from home_credit_observability.validation.validator import validate_credit_risk

PERFORMANCE_METRICS = ("accuracy", "precision", "recall", "f1_score", "roc_auc")


def _evaluate_scored_data(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Calcula métricas supervisionadas para um dataset já pontuado."""
    return evaluate_binary_classifier(
        dataframe["loan_status"],
        dataframe["prediction"],
        dataframe["prediction_probability"],
    )


def _compare_performance(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict[str, Any]:
    """Compara o desempenho atual com o desempenho na referência."""
    reference_metrics = _evaluate_scored_data(reference)
    production_metrics = _evaluate_scored_data(current)
    delta = {
        metric: float(production_metrics[metric] - reference_metrics[metric])
        for metric in PERFORMANCE_METRICS
    }
    return {
        "reference": reference_metrics,
        "production": production_metrics,
        "delta": delta,
    }


def run_drift_pipeline(
    *,
    sample_size: int = 8_000,
    random_state: int = 42,
    reference_path: Path | None = None,
    model_path: Path | None = None,
    production_dir: Path | None = None,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    """Executa toda a Etapa 2 e devolve caminhos e resumo estatístico."""
    reference_path = reference_path or REFERENCE_DIR / "reference.parquet"
    model_path = model_path or MODELS_DIR / "baseline_pipeline.joblib"
    production_dir = production_dir or PROJECT_ROOT / "data" / "production"
    report_dir = report_dir or PROJECT_ROOT / "reports" / "drift"
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Referência não encontrada: {reference_path}. Execute o comando train."
        )

    reference = pd.read_parquet(reference_path)
    simulator = ProductionDataSimulator(
        sample_size=sample_size,
        random_state=random_state,
    )
    current = validate_credit_risk(simulator.simulate(reference), min_rows=1)

    predictor = PredictionService(model_path)
    reference_scored = predictor.predict(reference)
    current_scored = predictor.predict(current)

    service = DriftDetectionService(
        [PSIDetector(threshold=0.25), KolmogorovSmirnovDetector(alpha=0.05)]
    )
    results = service.analyze(reference, current, NUMERIC_FEATURES)
    serialized = [result.to_dict() for result in results]

    production_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    production_path = production_dir / "production_drifted.parquet"
    results_path = report_dir / "statistical_drift_results.json"
    performance_path = report_dir / "model_performance_comparison.json"
    html_path = report_dir / "evidently_drift_report.html"
    current_scored.to_parquet(production_path, index=False)
    results_path.write_text(
        json.dumps(serialized, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    performance = _compare_performance(reference_scored, current_scored)
    performance_path.write_text(
        json.dumps(performance, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    report_columns = MODEL_FEATURES + ["prediction", "prediction_probability"]
    EvidentlyDriftReport().generate(
        reference_scored.loc[:, report_columns],
        current_scored.loc[:, report_columns],
        html_path,
    )
    return {
        "production_path": str(production_path),
        "statistical_results_path": str(results_path),
        "performance_comparison_path": str(performance_path),
        "evidently_report_path": str(html_path),
        "reference_rows": len(reference),
        "production_rows": len(current),
        "drift_tests": len(results),
        "detected_tests": sum(result.drift_detected for result in results),
        "drifted_features": sorted(
            {result.feature for result in results if result.drift_detected}
        ),
        "performance_delta": performance["delta"],
    }

"""Experimento sintético de Concept Drift com P(X) exatamente preservada."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binomtest

from home_credit_observability.config import PROJECT_ROOT
from home_credit_observability.data.prepare import MODEL_FEATURES, NUMERIC_FEATURES
from home_credit_observability.drift.detector import DriftDetectionService
from home_credit_observability.drift.ks import KolmogorovSmirnovDetector
from home_credit_observability.drift.psi import PSIDetector
from home_credit_observability.models.train import train_baseline
from home_credit_observability.observability.alerts import AlertPolicyEvaluator
from home_credit_observability.pipelines.drift_pipeline import _compare_performance
from home_credit_observability.reporting.evidently_report import EvidentlyDriftReport
from home_credit_observability.validation.validator import validate_credit_risk


def synthetic_cohort(rows: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Gera features financeiras fictícias e risco conhecido antes da mudança."""
    rng = np.random.default_rng(seed)
    income = rng.lognormal(np.log(60_000), 0.4, rows)
    ratio = rng.uniform(0.05, 0.55, rows)
    interest = rng.uniform(8, 20, rows)
    data = pd.DataFrame(
        {
            "person_age": rng.integers(20, 70, rows).astype(float),
            "person_income": income,
            "person_emp_length": rng.uniform(0, 25, rows),
            "loan_amnt": income * ratio,
            "loan_int_rate": interest,
            "loan_percent_income": ratio,
            "cb_person_cred_hist_length": rng.uniform(0, 15, rows),
            "person_home_ownership": rng.choice(["RENT", "OWN"], rows),
            "loan_intent": "PERSONAL",
            "loan_grade": "B",
            "cb_person_default_on_file": "N",
        }
    )
    z = (interest - 14) / 3.5 + (ratio - 0.30) / 0.15
    probability_before = expit(-1 + 1.8 * z)
    data["loan_status"] = (rng.random(rows) < probability_before).astype(int)
    return data, z


def run_concept_pipeline(
    *,
    sample_size: int = 8_000,
    random_state: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Treina em coorte independente e muda apenas P(Y|X) na avaliação.

    logit(p_antes) = -1 + 1,8*z; logit(p_depois) = -1 - 1,8*z.
    A reversão é um mecanismo didático conhecido, não hipótese sobre crédito real.
    Os mesmos uniformes acoplam os resultados potenciais antes/depois.
    """
    if sample_size < 100:
        raise ValueError("Concept Drift exige pelo menos 100 registros")
    output_dir = output_dir or PROJECT_ROOT / "reports" / "concept"
    output_dir.mkdir(parents=True, exist_ok=True)
    training, _ = synthetic_cohort(12_000, random_state)
    model, training_metrics = train_baseline(validate_credit_risk(training))
    before, z = synthetic_cohort(sample_size, random_state + 1)
    uniform = np.random.default_rng(random_state + 2).random(sample_size)
    p_before, p_after = expit(-1 + 1.8 * z), expit(-1 - 1.8 * z)
    before["loan_status"] = (uniform < p_before).astype(int)
    after = before.copy()
    after["loan_status"] = (uniform < p_after).astype(int)
    before = validate_credit_risk(before, min_rows=1)
    after = validate_credit_risk(after, min_rows=1)
    pd.testing.assert_frame_equal(before[MODEL_FEATURES], after[MODEL_FEATURES])
    for data in (before, after):
        data["prediction"] = model.predict(data[MODEL_FEATURES])
        data["prediction_probability"] = model.predict_proba(data[MODEL_FEATURES])[:, 1]
    results = DriftDetectionService(
        [PSIDetector(), KolmogorovSmirnovDetector()]
    ).analyze(before, after, NUMERIC_FEATURES)
    performance = _compare_performance(before, after)
    performance["evaluation_protocol"] = "paired_synthetic_conditional_shift"
    error_before = before.prediction != before.loan_status
    error_after = after.prediction != after.loan_status
    worsened = int((~error_before & error_after).sum())
    improved = int((error_before & ~error_after).sum())
    p_value = (
        float(binomtest(worsened, worsened + improved, p=0.5).pvalue)
        if worsened + improved
        else 1.0
    )
    metrics = {
        "prediction_null_ratio": float(after.prediction.isna().mean()),
        "drifted_features_count": float(sum(r.drift_detected for r in results)),
        "model_recall_delta": performance["delta"]["recall"],
        "model_roc_auc": performance["production"]["roc_auc"],
    }
    # Este experimento não mede latência: a métrica fica ausente.
    alerts = AlertPolicyEvaluator().evaluate(metrics)
    summary = {
        "scenario": "synthetic_concept_drift",
        "features_exactly_equal": True,
        "feature_drift_tests_positive": sum(r.drift_detected for r in results),
        "conditional_probability_mean_absolute_change": float(
            np.mean(np.abs(p_after - p_before))
        ),
        "positive_rate_before": float(before.loan_status.mean()),
        "positive_rate_after": float(after.loan_status.mean()),
        "paired_error_test": {
            "method": "exact_mcnemar_binomial",
            "worsened": worsened,
            "improved": improved,
            "p_value": p_value,
        },
        "performance": performance,
        "alerts": [a.to_dict() for a in alerts],
        "training_metrics": training_metrics,
        "random_state": random_state,
        "evaluation_rows": sample_size,
        "mechanism": "logit(p): -1 + 1.8*z -> -1 - 1.8*z",
        "limitation": "Cenário sintético; não demonstra causalidade em crédito real.",
    }
    before.to_parquet(output_dir / "before.parquet", index=False)
    after.to_parquet(output_dir / "after.parquet", index=False)
    joblib.dump(model, output_dir / "synthetic_model.joblib")
    for filename, payload in (
        ("concept_summary.json", summary),
        ("statistical_drift_results.json", [r.to_dict() for r in results]),
    ):
        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    EvidentlyDriftReport().generate(before, after, output_dir / "evidently_inputs.html")
    # Relatório supervisionado simples com valores e limites explicitados.
    rows = "".join(
        f"<tr><td>{name}</td><td>{performance['reference'][name]:.4f}</td>"
        f"<td>{performance['production'][name]:.4f}</td>"
        f"<td>{performance['delta'][name]:+.4f}</td></tr>"
        for name in ("accuracy", "precision", "recall", "f1_score", "roc_auc")
    )
    (output_dir / "concept_report.html").write_text(
        "<!doctype html><html lang='pt-BR'><meta charset='utf-8'>"
        "<title>Concept Drift controlado</title><body>"
        "<h1>Concept Drift: entradas iguais, relação com o alvo alterada</h1>"
        "<p>Cenário inteiramente sintético. Modelo treinado em outra coorte. "
        "logit(p): -1 + 1,8*z antes; -1 - 1,8*z depois. "
        "PSI e KS das entradas: nenhum teste positivo.</p>"
        "<table border='1' cellpadding='8'><tr><th>Métrica</th><th>Antes</th>"
        f"<th>Depois</th><th>Delta</th></tr>{rows}</table>"
        f"<p>McNemar exato: p={p_value:.3g}; erros novos={worsened}, "
        f"erros resolvidos={improved}.</p>"
        "<p>A mudança condicional é conhecida por construção. A queda de métricas "
        "isolada em dados reais não provaria Concept Drift. Os rótulos são "
        "sintéticos e disponíveis imediatamente; crédito real exige maturação.</p>"
        f"<p>Alertas: {', '.join(a.code for a in alerts) or 'nenhum'}.</p>"
        "<a href='evidently_inputs.html'>Relatório de estabilidade das entradas</a>"
        "</body></html>",
        encoding="utf-8",
    )
    return summary

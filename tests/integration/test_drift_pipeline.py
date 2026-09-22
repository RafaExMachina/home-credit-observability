"""Teste de integração do pipeline da Etapa 2."""

from pathlib import Path

import pandas as pd

from home_credit_observability.data.prepare import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from home_credit_observability.models.train import train_baseline
from home_credit_observability.pipelines.drift_pipeline import run_drift_pipeline


def _dataset(rows: int = 200) -> pd.DataFrame:
    """Cria um dataset válido suficiente para treino e drift."""
    index = pd.Series(range(rows))
    data = pd.DataFrame(
        {
            "person_age": 20 + index % 50,
            "person_income": 30_000.0 + index * 500,
            "person_emp_length": 1.0 + index % 20,
            "loan_amnt": 2_000.0 + index * 20,
            "loan_int_rate": 7.0 + index % 10,
            "loan_percent_income": 0.05 + (index % 20) / 100,
            "cb_person_cred_hist_length": 1 + index % 30,
            "person_home_ownership": "RENT",
            "loan_intent": "EDUCATION",
            "loan_grade": "B",
            "cb_person_default_on_file": "N",
            TARGET_COLUMN: index % 2,
        }
    )
    return data[[TARGET_COLUMN] + NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def test_drift_pipeline_creates_expected_artifacts(tmp_path: Path, monkeypatch) -> None:
    """O pipeline deve persistir produção, estatísticas e relatório HTML."""
    reference = _dataset()
    reference_path = tmp_path / "reference.parquet"
    model_path = tmp_path / "model.joblib"
    reference.to_parquet(reference_path, index=False)
    model, _ = train_baseline(reference)

    import joblib

    joblib.dump(model, model_path)

    def fake_generate(self, reference, current, output_path):
        output_path.write_text("<html>drift</html>", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "home_credit_observability.pipelines.drift_pipeline."
        "EvidentlyDriftReport.generate",
        fake_generate,
    )
    result = run_drift_pipeline(
        sample_size=100,
        reference_path=reference_path,
        model_path=model_path,
        production_dir=tmp_path / "production",
        report_dir=tmp_path / "reports",
    )
    assert result["production_rows"] == 100
    assert result["drift_tests"] == len(NUMERIC_FEATURES) * 2
    assert Path(result["production_path"]).exists()
    assert Path(result["statistical_results_path"]).exists()
    assert Path(result["performance_comparison_path"]).exists()
    assert Path(result["evidently_report_path"]).exists()
    assert set(result["performance_delta"]) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }

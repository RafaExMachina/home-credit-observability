"""Verifica independência das coortes e Concept Drift com Evidently real."""

import json

import numpy as np
import pandas as pd

from home_credit_observability.data.prepare import MODEL_FEATURES
from home_credit_observability.pipelines import train_pipeline
from home_credit_observability.pipelines.concept_pipeline import (
    run_concept_pipeline,
    synthetic_cohort,
)
from home_credit_observability.reporting.evidently_report import (
    EvidentlyDriftReport,
    psi_or_ks,
)


def test_concept_changes_conditional_relationship_without_feature_drift(tmp_path):
    result = run_concept_pipeline(sample_size=1000, output_dir=tmp_path)
    before = pd.read_parquet(tmp_path / "before.parquet")
    after = pd.read_parquet(tmp_path / "after.parquet")
    pd.testing.assert_frame_equal(before[MODEL_FEATURES], after[MODEL_FEATURES])
    assert result["feature_drift_tests_positive"] == 0
    assert result["performance"]["delta"]["roc_auc"] < -0.2
    assert result["paired_error_test"]["p_value"] < 0.05
    assert (tmp_path / "evidently_inputs.html").stat().st_size > 1000
    assert (tmp_path / "concept_report.html").exists()
    assert result["conditional_probability_mean_absolute_change"] > 0.2
    metrics = json.loads((tmp_path / "evidently_inputs.json").read_text())["metrics"]
    assert metrics[0]["value"]["count"] == 0


def test_training_reserves_production_before_fitting(tmp_path, monkeypatch):
    data, _ = synthetic_cohort(5000, 42)
    monkeypatch.setattr(train_pipeline, "download_credit_risk", lambda **_: tmp_path)
    monkeypatch.setattr(train_pipeline, "load_application_data", lambda _: data)
    monkeypatch.setattr(train_pipeline, "REFERENCE_DIR", tmp_path / "reference")
    monkeypatch.setattr(train_pipeline, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train_pipeline, "METRICS_DIR", tmp_path / "metrics")
    result = train_pipeline.run_training_pipeline()
    manifest = json.loads((tmp_path / "reference/split_manifest.json").read_text())
    dev_ids = set(manifest["development_row_ids"])
    production_ids = set(manifest["production_row_ids"])
    assert not dev_ids & production_ids
    assert len(dev_ids | production_ids) == 5000
    assert result["metrics"]["train_rows"] + result["metrics"]["test_rows"] == 3500


def test_evidently_callback_matches_operational_policy():
    ref = pd.Series(np.arange(100, dtype=float), name="person_income")
    for current in (ref.copy(), ref + 1000):
        score, detected = psi_or_ks(ref, current, "num", 1.0)
        assert np.isfinite(score)
        assert detected == bool((current != ref).any())


def test_real_evidently_marks_one_changed_feature_as_dataset_drift(tmp_path):
    reference = pd.DataFrame(
        {
            "person_income": np.arange(100, dtype=float),
            # Coluna estável, mas não constante: a correlação do Evidently
            # não está definida para variância zero. PSI constante tem
            # cobertura própria em tests/unit/drift/test_psi.py.
            "person_age": np.linspace(20.0, 60.0, 100),
        }
    )
    current = reference.copy()
    current["person_income"] += 1000
    EvidentlyDriftReport().generate(reference, current, tmp_path / "drift.html")
    metrics = json.loads((tmp_path / "drift.json").read_text())["metrics"]
    assert metrics[0]["value"]["count"] == 1
    assert metrics[0]["value"]["share"] >= metrics[0]["config"]["drift_share"]

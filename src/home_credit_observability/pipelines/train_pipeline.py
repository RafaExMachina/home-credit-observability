from __future__ import annotations

import json
from typing import Any

import joblib
from sklearn.model_selection import train_test_split

from home_credit_observability.config import (
    METRICS_DIR,
    MODELS_DIR,
    REFERENCE_DIR,
)
from home_credit_observability.data.download import download_credit_risk
from home_credit_observability.data.load import load_application_data
from home_credit_observability.data.prepare import (
    clean_reference_data,
    select_model_columns,
)
from home_credit_observability.models.train import train_baseline
from home_credit_observability.validation.validator import validate_credit_risk


def run_training_pipeline(force_download: bool = False) -> dict[str, Any]:
    raw_path = download_credit_risk(force=force_download)
    dataframe = load_application_data(raw_path)
    selected = select_model_columns(dataframe)
    cleaned = clean_reference_data(selected)
    validated = validate_credit_risk(cleaned)

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    # Reserva antes do fitting: a coorte de produção nunca entra no treino
    # nem na validação interna realizada por train_baseline.
    development, production_source = train_test_split(
        validated, test_size=0.30, random_state=42, stratify=validated["loan_status"]
    )
    if set(development.index) & set(production_source.index):
        raise RuntimeError("Sobreposição entre desenvolvimento e produção")
    reference_path = REFERENCE_DIR / "reference.parquet"
    source_path = REFERENCE_DIR / "production_source.parquet"
    development.to_parquet(reference_path, index=False)
    production_source.to_parquet(source_path, index=False)
    manifest_path = REFERENCE_DIR / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "random_state": 42,
                "production_fraction": 0.30,
                "development_row_ids": development.index.tolist(),
                "production_row_ids": production_source.index.tolist(),
                "overlap_count": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model, metrics = train_baseline(development)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "baseline_pipeline.joblib"
    metrics_path = METRICS_DIR / "baseline_metrics.json"
    joblib.dump(model, model_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "raw_path": str(raw_path),
        "reference_path": str(reference_path),
        "production_source_path": str(source_path),
        "split_manifest_path": str(manifest_path),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }

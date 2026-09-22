"""Executa o pipeline de drift a partir da raiz do projeto."""

from home_credit_observability.pipelines.drift_pipeline import run_drift_pipeline

if __name__ == "__main__":
    print(run_drift_pipeline())

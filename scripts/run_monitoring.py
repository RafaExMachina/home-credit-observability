"""Atalho executável para o pipeline de observabilidade."""

import json

from home_credit_observability.pipelines.monitoring_pipeline import (
    run_monitoring_pipeline,
)

if __name__ == "__main__":
    print(json.dumps(run_monitoring_pipeline(), indent=2, ensure_ascii=False))

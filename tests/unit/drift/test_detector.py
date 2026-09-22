"""Testes do serviço agregador de detectores."""

import pandas as pd

from home_credit_observability.drift.detector import DriftDetectionService
from home_credit_observability.drift.ks import KolmogorovSmirnovDetector
from home_credit_observability.drift.psi import PSIDetector


def test_service_combines_features_and_strategies() -> None:
    """O serviço deve produzir um resultado por feature e detector."""
    reference = pd.DataFrame({"a": range(100), "b": range(100, 200)})
    current = reference.copy()
    service = DriftDetectionService([PSIDetector(), KolmogorovSmirnovDetector()])
    results = service.analyze(reference, current, ["a", "b"])
    assert len(results) == 4
    assert {result.method for result in results} == {"psi", "kolmogorov_smirnov"}
    assert not any(result.drift_detected for result in results)

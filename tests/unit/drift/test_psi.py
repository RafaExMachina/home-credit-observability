"""Testes do detector PSI."""

import numpy as np
import pandas as pd

from home_credit_observability.drift.psi import PSIDetector


def test_psi_does_not_detect_equal_distribution() -> None:
    """Distribuições idênticas devem produzir PSI igual a zero."""
    values = pd.Series(np.arange(1, 101, dtype=float))
    result = PSIDetector().detect(values, values.copy(), feature="income")
    assert result.statistic == 0.0
    assert result.drift_detected is False


def test_psi_detects_large_shift() -> None:
    """Uma translação forte deve ultrapassar o limiar de drift."""
    reference = pd.Series(np.arange(1, 101, dtype=float))
    current = reference + 1_000
    result = PSIDetector().detect(reference, current, feature="income")
    assert result.statistic >= 0.25
    assert result.drift_detected is True

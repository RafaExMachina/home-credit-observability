"""Testes do detector PSI."""

import json

import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize("current_value, expected", [(1.0, False), (2.0, True)])
def test_constant_reference_is_json_safe(current_value, expected):
    result = PSIDetector().detect(
        pd.Series([1.0] * 100),
        pd.Series([current_value] * 100),
        feature="constant",
    )
    assert np.isfinite(result.statistic)
    assert result.drift_detected is expected
    json.dumps(result.to_dict(), allow_nan=False)

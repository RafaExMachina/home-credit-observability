"""Testes do detector Kolmogorov-Smirnov."""

import numpy as np
import pandas as pd

from home_credit_observability.drift.ks import KolmogorovSmirnovDetector


def test_ks_does_not_detect_equal_samples() -> None:
    """Amostras idênticas não devem sinalizar drift."""
    values = pd.Series(np.linspace(0, 1, 200))
    result = KolmogorovSmirnovDetector().detect(values, values.copy(), feature="rate")
    assert result.p_value == 1.0
    assert result.drift_detected is False


def test_ks_detects_shifted_samples() -> None:
    """Amostras claramente separadas devem sinalizar drift."""
    reference = pd.Series(np.linspace(0, 1, 200))
    current = pd.Series(np.linspace(10, 11, 200))
    result = KolmogorovSmirnovDetector().detect(reference, current, feature="rate")
    assert result.p_value is not None and result.p_value < 0.05
    assert result.drift_detected is True

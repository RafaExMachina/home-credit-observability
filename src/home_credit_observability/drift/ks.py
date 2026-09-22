"""Detector baseado no teste de Kolmogorov-Smirnov de duas amostras."""

from __future__ import annotations

import pandas as pd
from scipy.stats import ks_2samp

from home_credit_observability.drift.interfaces import DriftResult


class KolmogorovSmirnovDetector:
    """Detecta drift numérico quando o valor-p fica abaixo de alfa."""

    def __init__(self, *, alpha: float = 0.05) -> None:
        """Inicializa o detector com o nível de significância desejado."""
        if not 0 < alpha < 1:
            raise ValueError("alpha deve estar entre 0 e 1")
        self.alpha = alpha

    def detect(
        self,
        reference: pd.Series,
        current: pd.Series,
        *,
        feature: str,
    ) -> DriftResult:
        """Executa o teste KS após remover nulos e coerções inválidas."""
        ref = pd.to_numeric(reference, errors="coerce").dropna()
        cur = pd.to_numeric(current, errors="coerce").dropna()
        if ref.empty or cur.empty:
            raise ValueError(f"{feature}: amostras sem valores numéricos válidos")
        statistic, p_value = ks_2samp(ref, cur, alternative="two-sided", method="auto")
        return DriftResult(
            feature=feature,
            method="kolmogorov_smirnov",
            statistic=float(statistic),
            threshold=self.alpha,
            drift_detected=bool(p_value < self.alpha),
            p_value=float(p_value),
        )

"""Detector baseado no Population Stability Index (PSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from home_credit_observability.drift.interfaces import DriftResult


class PSIDetector:
    """Detecta mudança de distribuição numérica por meio do PSI."""

    def __init__(self, *, threshold: float = 0.25, bins: int = 10) -> None:
        """Inicializa o detector.

        Args:
            threshold: Valor a partir do qual o drift é considerado severo.
            bins: Quantidade máxima de intervalos baseados em quantis.
        """
        if threshold <= 0:
            raise ValueError("threshold deve ser positivo")
        if bins < 2:
            raise ValueError("bins deve ser pelo menos 2")
        self.threshold = threshold
        self.bins = bins

    def detect(
        self,
        reference: pd.Series,
        current: pd.Series,
        *,
        feature: str,
    ) -> DriftResult:
        """Calcula o PSI usando limites determinados pela referência."""
        reference_values = self._finite_values(reference)
        current_values = self._finite_values(current)
        if reference_values.size == 0 or current_values.size == 0:
            raise ValueError(f"{feature}: amostras sem valores numéricos válidos")

        quantiles = np.linspace(0.0, 1.0, self.bins + 1)
        edges = np.unique(np.quantile(reference_values, quantiles))
        if edges.size < 2:
            statistic = 0.0 if np.all(current_values == edges[0]) else float("inf")
        else:
            edges[0], edges[-1] = -np.inf, np.inf
            ref_count, _ = np.histogram(reference_values, bins=edges)
            cur_count, _ = np.histogram(current_values, bins=edges)
            epsilon = 1e-6
            ref_pct = np.clip(ref_count / reference_values.size, epsilon, None)
            cur_pct = np.clip(cur_count / current_values.size, epsilon, None)
            statistic = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

        return DriftResult(
            feature=feature,
            method="psi",
            statistic=statistic,
            threshold=self.threshold,
            drift_detected=statistic >= self.threshold,
        )

    @staticmethod
    def _finite_values(series: pd.Series) -> np.ndarray:
        """Converte uma série em um vetor numérico finito."""
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        return values[np.isfinite(values)]

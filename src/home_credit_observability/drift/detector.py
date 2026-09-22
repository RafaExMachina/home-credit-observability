"""Serviço que combina diferentes estratégias de detecção de drift."""

from __future__ import annotations

import pandas as pd

from home_credit_observability.drift.interfaces import DriftDetector, DriftResult


class DriftDetectionService:
    """Aplica detectores injetados a um conjunto de variáveis numéricas."""

    def __init__(self, detectors: list[DriftDetector]) -> None:
        """Recebe estratégias intercambiáveis de detecção."""
        if not detectors:
            raise ValueError("ao menos um detector deve ser informado")
        self.detectors = detectors

    def analyze(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        features: list[str],
    ) -> list[DriftResult]:
        """Executa todos os detectores para cada variável solicitada."""
        available = set(reference.columns).intersection(current.columns)
        missing = sorted(set(features) - available)
        if missing:
            raise ValueError(f"Colunas ausentes para análise: {missing}")
        return [
            detector.detect(reference[feature], current[feature], feature=feature)
            for feature in features
            for detector in self.detectors
        ]

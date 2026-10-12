"""Adaptador responsável pela geração do relatório Evidently."""

from __future__ import annotations

from math import log10
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.legacy.calculations.stattests.registry import StatTest, register_stattest
from evidently.legacy.core import ColumnType
from evidently.presets import DataDriftPreset

from home_credit_observability.data.prepare import NUMERIC_FEATURES
from home_credit_observability.drift.ks import KolmogorovSmirnovDetector
from home_credit_observability.drift.psi import PSIDetector


def psi_or_ks(reference, current, column_type, threshold):
    """Mesma política operacional: PSI >= 0,25 OU KS p < 0,05.

    Score exibido = max(PSI/0,25, -log10(p)/-log10(0,05)).
    O booleano usa os mesmos detectores do pipeline, inclusive nos limites.
    column_type e threshold integram o contrato de callbacks do Evidently.
    """
    psi = PSIDetector().detect(reference, current, feature=str(reference.name))
    ks = KolmogorovSmirnovDetector().detect(
        reference, current, feature=str(reference.name)
    )
    score = max(
        psi.statistic / 0.25,
        -log10(max(float(ks.p_value), 1e-300)) / -log10(0.05),
    )
    return score, psi.drift_detected or ks.drift_detected


register_stattest(
    StatTest(
        name="project_psi_or_ks",
        display_name="PSI >= 0.25 OR KS p < 0.05 (normalized score)",
        allowed_feature_types=[ColumnType.Numerical],
        default_threshold=1.0,
    ),
    psi_or_ks,
)


class EvidentlyDriftReport:
    """Gera e persiste um relatório HTML de data drift."""

    def generate(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        """Executa o preset de drift e salva a visualização interativa."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        columns = [name for name in NUMERIC_FEATURES if name in reference.columns]
        if not columns:
            raise ValueError("O relatório exige ao menos uma feature numérica")
        # Um preset global positivo se pelo menos uma feature tiver drift.
        # Saídas do modelo não contam como features monitoradas.
        report = Report(
            [
                DataDriftPreset(
                    columns=columns,
                    method="project_psi_or_ks",
                    threshold=1.0,
                    drift_share=1 / len(columns),
                )
            ]
        )
        definition = DataDefinition(numerical_columns=columns)
        snapshot = report.run(
            current_data=Dataset.from_pandas(
                current[columns], data_definition=definition
            ),
            reference_data=Dataset.from_pandas(
                reference[columns], data_definition=definition
            ),
        )
        snapshot.save_html(str(output_path))
        snapshot.save_json(str(output_path.with_suffix(".json")))
        return output_path

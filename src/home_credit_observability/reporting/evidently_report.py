"""Adaptador responsável pela geração do relatório Evidently."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


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
        report = Report([DataDriftPreset()])
        snapshot = report.run(current_data=current, reference_data=reference)
        snapshot.save_html(str(output_path))
        return output_path

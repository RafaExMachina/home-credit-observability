"""Serviço de inferência para os datasets de referência e produção."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from home_credit_observability.data.prepare import MODEL_FEATURES


class PredictionService:
    """Carrega um modelo persistido e acrescenta suas predições aos dados."""

    def __init__(self, model_path: Path) -> None:
        """Carrega o pipeline scikit-learn salvo na Etapa 1."""
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        self.model: Any = joblib.load(model_path)

    def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Retorna uma cópia com classe e probabilidade estimadas."""
        output = dataframe.copy()
        features = output.loc[:, MODEL_FEATURES]
        output["prediction"] = self.model.predict(features)
        output["prediction_probability"] = self.model.predict_proba(features)[:, 1]
        return output

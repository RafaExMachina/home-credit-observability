from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from home_credit_observability.config import DATASET_FILE, OPENML_DATASET_ID, RAW_DIR

TARGET_COLUMN = "loan_status"


def _build_frame(dataset: object) -> pd.DataFrame:
    frame = getattr(dataset, "frame", None)
    if frame is None:
        raise RuntimeError("O OpenML não retornou o dataset como DataFrame.")
    dataframe = frame.copy()
    if TARGET_COLUMN not in dataframe.columns:
        target = getattr(dataset, "target", None)
        if target is None:
            raise RuntimeError(f"A coluna-alvo {TARGET_COLUMN!r} não foi encontrada.")
        dataframe[TARGET_COLUMN] = target
    dataframe[TARGET_COLUMN] = pd.to_numeric(
        dataframe[TARGET_COLUMN], errors="raise"
    ).astype("int64")
    return dataframe


def download_credit_risk(force: bool = False) -> Path:
    """Baixa o Credit Risk Dataset 43454 diretamente do OpenML."""
    destination = RAW_DIR / DATASET_FILE
    if destination.exists() and not force:
        return destination
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset = fetch_openml(
        data_id=OPENML_DATASET_ID,
        as_frame=True,
        parser="auto",
    )
    dataframe = _build_frame(dataset)
    dataframe.to_csv(destination, index=False)
    if len(dataframe) < 5_000:
        raise RuntimeError("O dataset baixado possui menos de 5.000 registros.")
    return destination

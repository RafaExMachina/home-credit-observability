from pathlib import Path

import pandas as pd


def load_application_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    return pd.read_csv(path)

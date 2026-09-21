from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pandera.pandas as pa

from home_credit_observability.config import REPORTS_DIR
from home_credit_observability.validation.schema import build_credit_risk_schema


class DataQualityError(RuntimeError):
    """Indica que o lote foi bloqueado pelo contrato de dados."""


def validate_credit_risk(
    dataframe: pd.DataFrame,
    *,
    min_rows: int = 5_000,
    report_name: str = "validation_failures.csv",
) -> pd.DataFrame:
    schema = build_credit_risk_schema(min_rows=min_rows)
    try:
        return schema.validate(dataframe, lazy=True)
    except pa.errors.SchemaErrors as exc:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / report_name
        failures = exc.failure_cases.copy()
        failures.insert(0, "validated_at_utc", datetime.now(UTC).isoformat())
        failures.to_csv(report_path, index=False)
        raise DataQualityError(
            f"Lote bloqueado: {len(failures)} violações. Relatório: {report_path}"
        ) from exc

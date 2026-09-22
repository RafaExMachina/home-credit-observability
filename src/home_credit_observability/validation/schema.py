from __future__ import annotations

import pandas as pd
import pandera.pandas as pa


def build_credit_risk_schema(min_rows: int = 5_000) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        columns={
            "person_age": pa.Column(
                float, pa.Check.in_range(18, 100), nullable=False, coerce=True
            ),
            "person_income": pa.Column(
                float, pa.Check.gt(0), nullable=False, coerce=True
            ),
            "person_home_ownership": pa.Column(
                str,
                pa.Check.isin(["RENT", "MORTGAGE", "OWN", "OTHER"]),
                nullable=False,
                coerce=True,
            ),
            "person_emp_length": pa.Column(
                float, pa.Check.in_range(0, 80), nullable=True, coerce=True
            ),
            "loan_intent": pa.Column(
                str,
                pa.Check.isin(
                    [
                        "EDUCATION",
                        "MEDICAL",
                        "VENTURE",
                        "PERSONAL",
                        "DEBTCONSOLIDATION",
                        "HOMEIMPROVEMENT",
                    ]
                ),
                nullable=False,
                coerce=True,
            ),
            "loan_grade": pa.Column(
                str, pa.Check.isin(list("ABCDEFG")), nullable=False, coerce=True
            ),
            "loan_amnt": pa.Column(float, pa.Check.gt(0), nullable=False, coerce=True),
            "loan_int_rate": pa.Column(
                float, pa.Check.gt(0), nullable=True, coerce=True
            ),
            "loan_status": pa.Column(
                int, pa.Check.isin([0, 1]), nullable=False, coerce=True
            ),
            "loan_percent_income": pa.Column(
                float, pa.Check.in_range(0, 1), nullable=False, coerce=True
            ),
            "cb_person_default_on_file": pa.Column(
                str, pa.Check.isin(["Y", "N"]), nullable=False, coerce=True
            ),
            "cb_person_cred_hist_length": pa.Column(
                float, pa.Check.in_range(0, 100), nullable=False, coerce=True
            ),
        },
        checks=[
            pa.Check(
                lambda df: len(df) >= min_rows,
                error=f"O lote deve possuir pelo menos {min_rows} registros",
            ),
            pa.Check(
                lambda df: not df.duplicated().any(),
                error="O lote não pode conter linhas completamente duplicadas",
            ),
        ],
        strict=True,
        name="openml_credit_risk_reference",
    )


def make_invalid_batch(dataframe: pd.DataFrame) -> pd.DataFrame:
    if len(dataframe) < 5:
        raise ValueError("São necessários pelo menos cinco registros.")
    invalid = dataframe.head(max(100, min(5_000, len(dataframe)))).copy()
    invalid.loc[invalid.index[0], "loan_status"] = 3
    invalid.loc[invalid.index[1], "person_income"] = -5_000.0
    invalid.loc[invalid.index[2], "person_age"] = 150
    return pd.concat([invalid, invalid.iloc[[3]]], ignore_index=True)

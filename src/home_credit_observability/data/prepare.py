import pandas as pd

TARGET_COLUMN = "loan_status"

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [TARGET_COLUMN] + MODEL_FEATURES


def select_model_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")
    return dataframe.loc[:, REQUIRED_COLUMNS].copy()


def clean_reference_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove violações de domínio; nulos imputáveis são preservados."""
    cleaned = dataframe.drop_duplicates().copy()
    cleaned = cleaned[cleaned[TARGET_COLUMN].isin([0, 1])]
    cleaned = cleaned[cleaned["person_age"].between(18, 100)]
    cleaned = cleaned[cleaned["person_income"] > 0]
    cleaned = cleaned[cleaned["loan_amnt"] > 0]
    cleaned = cleaned[cleaned["loan_percent_income"].between(0, 1)]
    cleaned = cleaned[
        cleaned["person_emp_length"].isna()
        | cleaned["person_emp_length"].between(0, 80)
    ]
    cleaned = cleaned[cleaned["loan_int_rate"].isna() | (cleaned["loan_int_rate"] > 0)]
    cleaned = cleaned[cleaned["cb_person_cred_hist_length"].between(0, 100)]
    return cleaned.reset_index(drop=True)

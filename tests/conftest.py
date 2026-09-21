import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def valid_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    size = 200
    return pd.DataFrame(
        {
            "person_age": rng.integers(21, 70, size).astype(float),
            "person_income": rng.uniform(20_000, 300_000, size),
            "person_home_ownership": rng.choice(
                ["RENT", "MORTGAGE", "OWN", "OTHER"], size
            ),
            "person_emp_length": rng.uniform(0, 35, size),
            "loan_intent": rng.choice(
                [
                    "EDUCATION",
                    "MEDICAL",
                    "VENTURE",
                    "PERSONAL",
                    "DEBTCONSOLIDATION",
                    "HOMEIMPROVEMENT",
                ],
                size,
            ),
            "loan_grade": rng.choice(list("ABCDEFG"), size),
            "loan_amnt": rng.uniform(1_000, 35_000, size),
            "loan_int_rate": rng.uniform(5, 25, size),
            "loan_status": np.tile([0, 0, 0, 1], size // 4),
            "loan_percent_income": rng.uniform(0.01, 0.8, size),
            "cb_person_default_on_file": rng.choice(["Y", "N"], size),
            "cb_person_cred_hist_length": rng.integers(2, 30, size).astype(float),
        }
    )


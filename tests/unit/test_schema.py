import pytest

from home_credit_observability.validation.validator import (
    DataQualityError,
    validate_credit_risk,
)


def test_valid_data_passes(valid_data):
    result = validate_credit_risk(valid_data, min_rows=100)
    assert len(result) == 200


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("loan_status", 3),
        ("person_income", -1.0),
        ("person_age", 150),
    ],
)
def test_invalid_values_fail(valid_data, column, value):
    valid_data.loc[0, column] = value
    with pytest.raises(DataQualityError):
        validate_credit_risk(valid_data, min_rows=100)


def test_duplicate_row_fails(valid_data):
    duplicated = valid_data.copy()
    duplicated.loc[1] = duplicated.loc[0]
    with pytest.raises(DataQualityError):
        validate_credit_risk(duplicated, min_rows=100)

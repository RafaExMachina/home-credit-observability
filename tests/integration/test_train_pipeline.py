from home_credit_observability.models.train import train_baseline
from home_credit_observability.validation.validator import validate_credit_risk


def test_baseline_trains_with_valid_data(valid_data):
    validated = validate_credit_risk(valid_data, min_rows=100)
    model, metrics = train_baseline(validated)
    assert hasattr(model, "predict")
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["train_rows"] == 160
    assert metrics["test_rows"] == 40

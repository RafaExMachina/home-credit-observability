from home_credit_observability.features.preprocessing import build_preprocessor


def test_preprocessor_transforms_data(valid_data):
    transformed = build_preprocessor().fit_transform(valid_data)
    assert transformed.shape[0] == len(valid_data)
    assert transformed.shape[1] > 0

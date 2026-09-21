from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from home_credit_observability.data.prepare import MODEL_FEATURES, TARGET_COLUMN
from home_credit_observability.features.preprocessing import build_preprocessor
from home_credit_observability.models.evaluate import evaluate_binary_classifier


def train_baseline(
    dataframe: pd.DataFrame,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    X = dataframe.loc[:, MODEL_FEATURES]
    y = dataframe[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary_classifier(y_test, predictions, probabilities)
    metrics.update(
        {
            "model": "logistic_regression",
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
        }
    )
    return pipeline, metrics

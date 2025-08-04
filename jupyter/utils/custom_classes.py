import pickle
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import boxcox

from dataclasses import dataclass, field
from sklearn.metrics import classification_report

from typing import Dict, Any

import pandas as pd


@dataclass
class CustomModel:
    name: str
    estimator: BaseEstimator
    params: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    val_scores: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    test_scores: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def test(self, X, y):
        # generate predictions
        y_pred = self.estimator.predict(X)

        # generate classification report
        report_dict = classification_report(y, y_pred, output_dict=True)

        # convert report to frame
        report_frame = pd.DataFrame(report_dict).transpose().drop(columns=["support"], index=["accuracy", "macro avg"])

        # save test scores
        self.test_scores = report_frame

        # Return the test_scores
        return self.test_scores


class CustomBoxCoxTransformer(BaseEstimator, TransformerMixin):
    """Applies predefined Box-Cox transformations to specific columns."""

    def __init__(self):
        self.lambda_dict = {
            "age": 1.188462507146636,
            "trestbps": -0.566962062306727,
            "chol": -0.12552639286383113,
            "thalach": 2.445456150508822,
            "oldpeak": 0.17759774299109296,
        }

    def fit(self, X, y=None):
        return self  # No fitting required, returns self

    def transform(self, X):
        """Transforms the data using Box-Cox with predefined lambdas."""
        X_transformed = X.copy()
        for col, lambda_ in self.lambda_dict.items():
            X_transformed[col] = boxcox(X_transformed[col], lmbda=lambda_)
        return X_transformed

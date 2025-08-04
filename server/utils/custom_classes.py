from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import boxcox


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

import numpy as np

from robust_regression import RobustLinearRegression
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor


def test_positive_incline():
    rng = np.random.RandomState(0)
    n_features = 1
    X, y = make_regression(n_samples=20, n_features=n_features, random_state=0, noise=4.0,
                           bias=100.0)

    X_outliers = rng.normal(0, 0.5, size=(4, n_features))
    y_outliers = rng.normal(0, 2.0, size=4)
    X_outliers[:2, :] += X.max() + X.mean()
    X_outliers[2:, :] += X.min() - X.mean()
    y_outliers[:2] += y.min() - y.mean()
    y_outliers[2:] += y.max() * 2 + y.mean()
    X = np.vstack((X, X_outliers))
    y = np.concatenate((y, y_outliers))

    robust_regression_estimator = RobustLinearRegression()
    huber_estimator = HuberRegressor()

    robust_regression_estimator.fit(X, y)
    huber_estimator.fit(X, y)

    assert (huber_estimator.coef_ * robust_regression_estimator.coef_[:-1] > 0).all()


def test_negative_incline():
    rng = np.random.RandomState(0)
    n_features = 1
    X, y = make_regression(n_samples=20, n_features=n_features, random_state=0, noise=4.0,
                           bias=100.0)

    X_outliers = rng.normal(0, 0.5, size=(4, n_features))
    y_outliers = rng.normal(0, 2.0, size=4)
    X_outliers[:2, :] += X.max() + X.mean()
    X_outliers[2:, :] += X.min() - X.mean()
    y_outliers[:2] += y.min() - y.mean()
    y_outliers[2:] += y.max() * 2 + y.mean()
    X = np.vstack((X, X_outliers))
    X = X.max() - X
    y = np.concatenate((y, y_outliers))

    robust_regression_estimator = RobustLinearRegression()
    huber_estimator = HuberRegressor()

    robust_regression_estimator.fit(X, y)
    huber_estimator.fit(X, y)

    # Use [:-1] to remove last coefficient (bias)
    assert (huber_estimator.coef_ * robust_regression_estimator.coef_[:-1] > 0).all()
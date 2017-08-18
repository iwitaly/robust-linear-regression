from sklearn.utils.estimator_checks import check_estimator
from robust_regression import RobustLinearRegression


def test_estimator():
    return check_estimator(RobustLinearRegression)

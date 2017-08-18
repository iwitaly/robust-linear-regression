import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def _l1_loss(w, *args):
    '''
    :param w: weights
    :param args: iterable that has (X, y)
    :return: l1 loss for linear regression model
    '''
    X = args[0]
    y = args[1]
    return np.sum(
        np.abs(y - X.dot(w))
    )


def _add_ones_column_to_matrix(X):
    '''
    Add a column of ones to a matrix X
    :param X: Any 2d matrix
    :return: X with a added column of ones
    '''
    ones = np.ones(shape=(X.shape[0], 1))
    return np.c_[X, np.ones(X.shape[0])]


class RobustLinearRegression(BaseEstimator):
    def __init__(self, max_iter=None, tol=1e-05):
        '''
        :param max_iter: param for optimization method
        :param tol: param for optimization method
        '''
        self.max_iter = max_iter
        self.tol = tol


    def fit(self, X, y):
        '''
        Fit linear regression model with L1 loss
        :param X: 2d matrix of features
        :param y: target variable
        :return: fitted model
        '''
        X, y = check_X_y(X, y)
        X_with_ones = _add_ones_column_to_matrix(X)

        rng = np.random.RandomState(0)
        w_initial = rng.normal(size=X_with_ones.shape[1])

        weights = optimize.fmin(_l1_loss, w_initial, args=(X_with_ones, y),
                                ftol=self.tol, maxiter=self.max_iter, disp=0)
        self.coef_ = weights
        return self


    def predict(self, X):
        check_is_fitted(self, ['coef_'])
        X = check_array(X)
        X_with_ones = _add_ones_column_to_matrix(X)
        return X_with_ones.dot(self.coef_)

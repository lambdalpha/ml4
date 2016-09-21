from .Regression import Regression
import numpy as np


class LinearRegression(Regression):
    def __init__(self):
        self.w = None

    def train(self, X, y):
        # self.w = np.dot(np.linalg.inv(x.T.dot(x)), x.T, y)
        # x = np.column_stack((np.ones(len(x)), x))
        self.w = np.dot(np.linalg.pinv(X), y)

    def predict(self, x):
        # x = np.array([1] + list(x))
        return np.dot(self.w, x)

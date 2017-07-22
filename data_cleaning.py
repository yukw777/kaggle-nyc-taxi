import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

class DistanceAdder(BaseEstimator, TransformerMixin):

    PICKUP_LONG_COL = 0
    PICKUP_LAT_COL = 1
    DROPOFF_LONG_COL = 2
    DROPOFF_LAT_COL = 3

    def __init__(self, add_distance=False):
        self.add_distance = add_distance

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.add_distance:
            x_diff = X[:, DROPOFF_LAT_COL] - X[:, PICKUP_LAT_COL]
            y_diff = X[:, DROPOFF_LONG_COL] - X[:, PICKUP_LONG_COL]
            distance = np.sqrt(np.square(x_diff) + np.square(y_diff))

            return np.c_[X, distance]
        return X

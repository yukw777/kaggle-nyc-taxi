import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

class DateTimeToTimeStampTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, time_attrs):
        self.time_attrs = time_attrs
        self.helper = np.vectorize(lambda t: t.timestamp())

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for attr in self.time_attrs:
            X[attr] = self.helper(X[attr].dt.to_pydatetime())

        return X

class DistanceAdder(BaseEstimator, TransformerMixin):

    EARTH_RADIUS = 6371     # Earth radius in km

    def __init__(self, add_distance=True):
        self.add_distance = add_distance

    def fit(self, X, y=None):
        return self

    def haversine_np(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

        c = 2 * np.arcsin(np.sqrt(a))
        km = self.EARTH_RADIUS * c

        return km

    def transform(self, X):
        if self.add_distance:
            distance = self.haversine_np(
                X['dropoff_longitude'],
                X['dropoff_latitude'],
                X['pickup_longitude'],
                X['pickup_latitude']
            )
            X['distance'] = distance
        return X

time_attrs = [
    'pickup_datetime',
    'dropoff_datetime',
]

attrs = [
    'pickup_datetime',
    'dropoff_datetime',
    'passenger_count',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
]

pipeline = Pipeline([
    ('dist_adder', DistanceAdder()),
    ('datetime_to_timestamp', DateTimeToTimeStampTransformer(time_attrs)),
    ('selector', DataFrameSelector(attrs)),
    ('std_scaler', StandardScaler()),
])

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

        a = np.sin(dlat/2.0)**2 + np.cos(lat1) \
            * np.cos(lat2) * np.sin(dlon/2.0)**2

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


class OutlierRemover(BaseEstimator, TransformerMixin):

    def __init__(self, attrs):
        self.attrs = attrs
        self.helper = np.vectorize(lambda t: t.timestamp())

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for attr in self.attrs:
            X = self.reject_outliers(X, attr)
        return X

    def reject_outliers(self, data, attr, m=2.):
        abs_distance = np.abs(data[attr]-data[attr].median())
        return data[abs_distance <= (m * data[attr].std())]


class CustomStandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self, attrs=[]):
        self.attrs = attrs
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.attrs], y)
        return self

    def transform(self, X):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(
            self.scaler.transform(X[self.attrs]),
            columns=self.attrs
        )
        X_not_scaled = X.ix[:, ~X.columns.isin(self.attrs)]

        # we need to reset the indices...
        X_scaled.reset_index(drop=True, inplace=True)
        X_not_scaled.reset_index(drop=True, inplace=True)

        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


class DataCleaner():

    def __init__(self):
        self.time_attrs = [
            'pickup_datetime',
            'dropoff_datetime',
        ]

        self.attrs = [
            'pickup_datetime',
            'dropoff_datetime',
            'passenger_count',
            'pickup_longitude',
            'pickup_latitude',
            'dropoff_longitude',
            'dropoff_latitude',
            'distance',
        ]

        self.label_attr = 'trip_duration'

        self.all_attrs = self.attrs + [self.label_attr]

        self.pipeline = Pipeline([
            ('dist_adder', DistanceAdder()),
            ('datetime_to_timestamp',
                DateTimeToTimeStampTransformer(self.time_attrs)),
            ('outlier_remover', OutlierRemover(self.all_attrs)),
            ('std_scaler', CustomStandardScaler(self.attrs)),
            ('selector', DataFrameSelector(self.all_attrs)),
        ])

    def clean(self, data_path):
        raw_data = pd.read_csv(data_path, parse_dates=self.time_attrs)
        return self.pipeline.fit_transform(raw_data)

    def pickle(self, data_to_pickle, pickle_path):
        np.save(pickle_path, data_to_pickle)

    def clean_and_pickle(self, data_path, pickle_path):
        cleaned_data = self.clean(data_path)
        self.pickle(cleaned_data, pickle_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_to_clean')
    parser.add_argument('pickle_file')
    args = parser.parse_args()

    dc = DataCleaner()
    dc.clean_and_pickle(args.data_file_to_clean, args.pickle_file)

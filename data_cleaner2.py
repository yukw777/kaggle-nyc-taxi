# import pandas as pd

from data_cleaner import DataCleaner
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class PickupDatetimeFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        prefix = 'pickup_'
        pickup_datetime = prefix + 'datetime'
        X[prefix + 'weekday'] = X[pickup_datetime].dt.weekday
        X[prefix + 'hour_weekofyear'] = X['pickup_datetime'].dt.weekofyear
        X[prefix + 'hour'] = X['pickup_datetime'].dt.hour
        X[prefix + 'minute'] = X['pickup_datetime'].dt.minute
        from_min = X['pickup_datetime'] - X['pickup_datetime'].min()
        X[prefix + 'dt'] = (from_min).dt.total_seconds()
        X[prefix + 'week_hour'] = X['pickup_weekday'] * 24 + X['pickup_hour']
        return X


class DataCleaner2(DataCleaner):

    def __init__(self):
        self.time_attrs = ['pickup_datetime']
        self.pipeline = Pipeline([
            ('pickup_datetime_features', PickupDatetimeFeatures())
        ])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_to_clean')
    parser.add_argument('pickle_file')
    args = parser.parse_args()

    dc = DataCleaner2()
    print(dc.clean(args.data_file_to_clean).head())
    # dc.clean_and_pickle(args.data_file_to_clean, args.pickle_file)

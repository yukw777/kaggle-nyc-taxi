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

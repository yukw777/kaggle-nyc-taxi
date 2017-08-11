# import pandas as pd
import numpy as np

from data_cleaner import DataCleaner
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


class NoFitEstimator(BaseEstimator):

    def fit(self, X, y=None):
        return self


class PickupDatetimeFeatures(NoFitEstimator, TransformerMixin):

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


class StoreAndFwdFlagToInt(NoFitEstimator, TransformerMixin):

    def transform(self, X):
        X['store_and_fwd_flag'] = 1 * (X.store_and_fwd_flag.values == 'Y')
        return X


class LogTripDuration(NoFitEstimator, TransformerMixin):

    def transform(self, X):
        X['log_trip_duration'] = np.log(X.trip_duration.values + 1)
        return X


class PCACoords(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        coords = np.vstack((
            X[['pickup_latitude', 'pickup_longitude']].values,
            X[['dropoff_latitude', 'dropoff_longitude']].values
        ))
        self.pca = PCA().fit(coords)

        return self

    def transform(self, X):
        pickup_pca_coords = self.pca.transform(
            X[['pickup_latitude', 'pickup_longitude']])
        X['pickup_pca0'] = pickup_pca_coords[:, 0]
        X['pickup_pca1'] = pickup_pca_coords[:, 1]
        dropoff_pca_coords = self.pca.transform(
            X[['dropoff_latitude', 'dropoff_longitude']])
        X['dropoff_pca0'] = dropoff_pca_coords[:, 0]
        X['dropoff_pca1'] = dropoff_pca_coords[:, 1]

        # Manhattan distances for PCA coordinates
        # Proved to be useful
        pca0_abs_diff = np.abs(X['dropoff_pca0'] - X['pickup_pca0'])
        pca1_abs_diff = np.abs(X['dropoff_pca1'] - X['pickup_pca1'])
        X['pca_manhattan'] = pca0_abs_diff + pca1_abs_diff

        return X


class DataCleaner2(DataCleaner):

    def __init__(self):
        self.time_attrs = ['pickup_datetime']
        self.pipeline = Pipeline([
            ('pickup_datetime_features', PickupDatetimeFeatures()),
            ('sf_flag_to_int', StoreAndFwdFlagToInt()),
            ('log_trip_duration', LogTripDuration()),
            ('pca_features', PCACoords()),
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

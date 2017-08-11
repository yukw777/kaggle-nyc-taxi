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


class HaversineDistance(NoFitEstimator, TransformerMixin):

    EARTH_RADIUS = 6371     # Earth radius in km

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
        distance = self.haversine_np(
            X['dropoff_longitude'],
            X['dropoff_latitude'],
            X['pickup_longitude'],
            X['pickup_latitude']
        )
        X['distance_haversine'] = distance
        return X


class Direction(NoFitEstimator, TransformerMixin):

    def bearing_array(self, lat1, lng1, lat2, lng2):
        lng_delta_rad = np.radians(lng2 - lng1)
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) \
            * np.cos(lng_delta_rad)
        return np.degrees(np.arctan2(y, x))

    def transform(self, X):
        X['direction'] = self.bearing_array(
            X['pickup_latitude'].values,
            X['pickup_longitude'].values,
            X['dropoff_latitude'].values,
            X['dropoff_longitude'].values
        )
        return X


class CenterCoords(NoFitEstimator, TransformerMixin):

    def transform(self, X):
        X['center_latitude'] = (X['pickup_latitude'].values
                                + X['dropoff_latitude'].values) / 2
        X['center_longitude'] = (X['pickup_longitude'].values
                                 + X['dropoff_longitude'].values) / 2
        return X


class DataCleaner2(DataCleaner):

    def __init__(self):
        self.time_attrs = ['pickup_datetime']
        self.pipeline = Pipeline([
            ('pickup_datetime_features', PickupDatetimeFeatures()),
            ('sf_flag_to_int', StoreAndFwdFlagToInt()),
            ('log_trip_duration', LogTripDuration()),
            ('pca_features', PCACoords()),
            ('haversine_distance', HaversineDistance()),
            ('direction', Direction()),
            ('center_coords', CenterCoords()),
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

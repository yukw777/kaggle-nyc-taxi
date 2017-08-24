import pandas as pd
import numpy as np

from data_cleaner import DataCleaner
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans


class NoFitEstimator(BaseEstimator):

    def fit(self, X, y=None):
        return self


class PickupDatetimeFeatures(NoFitEstimator, TransformerMixin):

    def transform(self, X):
        prefix = 'pickup_'
        pickup_datetime = prefix + 'datetime'
        X[prefix + 'date'] = X[pickup_datetime].dt.date
        X[prefix + 'weekday'] = X[pickup_datetime].dt.weekday
        X[prefix + 'hour_weekofyear'] = X[pickup_datetime].dt.weekofyear
        X[prefix + 'hour'] = X[pickup_datetime].dt.hour
        X[prefix + 'minute'] = X[pickup_datetime].dt.minute
        from_min = X[pickup_datetime] - X[pickup_datetime].min()
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


class CoordKMeans(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        coords = np.vstack((
            X[['pickup_latitude', 'pickup_longitude']].values,
            X[['dropoff_latitude', 'dropoff_longitude']].values
        ))
        self.kmeans = MiniBatchKMeans(
            n_clusters=100, batch_size=10000).fit(coords)

        return self

    def transform(self, X):
        X['pickup_cluster'] = self.kmeans.predict(
            X[['pickup_latitude', 'pickup_longitude']])
        X['dropoff_cluster'] = self.kmeans.predict(
            X[['dropoff_latitude', 'dropoff_longitude']])
        return X


class GeospatialAggregate(NoFitEstimator, TransformerMixin):

    def mean_avg_speed_log_trip_duration(self, X):
        # the means of average_speed and log_trip_duration
        # grouped by the given columns
        for gby_col in ['pickup_hour', 'pickup_date', 'pickup_dt_bin',
                        'pickup_week_hour', 'pickup_cluster',
                        'dropoff_cluster']:
            gby = X.groupby(
                gby_col).mean()[['avg_speed_h', 'log_trip_duration']]
            gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]
            X = pd.merge(
                X, gby, how='left', left_on=gby_col, right_index=True)
        return X

    def mean_count_avg_speed(self, X):
        # mean average speed and counts over provided columns
        for gby_cols in [['center_lat_bin', 'center_long_bin'],
                         ['pickup_hour', 'center_lat_bin', 'center_long_bin'],
                         ['pickup_hour', 'pickup_cluster'],
                         ['pickup_hour', 'dropoff_cluster'],
                         ['pickup_cluster', 'dropoff_cluster']]:
            coord_speed = X \
                .groupby(gby_cols) \
                .mean()[['avg_speed_h']] \
                .reset_index()
            coord_count = X \
                .groupby(gby_cols) \
                .count()[['id']] \
                .reset_index()
            coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)
            coord_stats = coord_stats[coord_stats['id'] > 100]
            col_names = []
            col_names.append('avg_speed_h_%s' % '_'.join(gby_cols))
            col_names.append('cnt_%s' % '_'.join(gby_cols))
            coord_stats.columns = gby_cols + col_names
            X = pd.merge(X, coord_stats, how='left', on=gby_cols)
        return X

    def transform(self, X):
        # some round ups.. won't be used for training
        X['pickup_lat_bin'] = np.round(X['pickup_latitude'], 2)
        X['pickup_long_bin'] = np.round(X['pickup_longitude'], 2)
        X['center_lat_bin'] = np.round(X['center_latitude'], 2)
        X['center_long_bin'] = np.round(X['center_longitude'], 2)
        X['pickup_dt_bin'] = (X['pickup_dt'] // (3 * 3600))
        X['avg_speed_h'] = 1000 * X['distance_haversine'] / X['trip_duration']

        X = self.mean_avg_speed_log_trip_duration(X)
        X = self.mean_count_avg_speed(X)

        X.drop('pickup_lat_bin', axis=1, inplace=True)
        X.drop('pickup_long_bin', axis=1, inplace=True)
        X.drop('center_lat_bin', axis=1, inplace=True)
        X.drop('center_long_bin', axis=1, inplace=True)
        X.drop('pickup_dt_bin', axis=1, inplace=True)
        X.drop('avg_speed_h', axis=1, inplace=True)

        return X


class CountFeatures(BaseEstimator, TransformerMixin):

    GROUP_FREQ = '60min'

    def count_over_group_freq(self, X):
        # Count trips over 60min
        copy_counts = self.copy \
            .set_index('pickup_datetime')[['id']].sort_index()
        copy_counts['count_60min'] = copy_counts \
            .isnull() \
            .rolling(self.GROUP_FREQ).count()['id']
        return X.merge(copy_counts, on='id', how='left')

    def trip_count_between_clusters(self, X):
        # Count how many trips are going to each cluster over time
        dropoff_counts = self.copy \
            .set_index('pickup_datetime') \
            .groupby([pd.TimeGrouper(self.GROUP_FREQ), 'dropoff_cluster']) \
            .agg({'id': 'count'}) \
            .reset_index().set_index('pickup_datetime') \
            .groupby('dropoff_cluster').rolling('240min').mean() \
            .drop('dropoff_cluster', axis=1) \
            .reset_index().set_index('pickup_datetime') \
            .shift(freq='-120min').reset_index() \
            .rename(columns={
                'pickup_datetime': 'pickup_datetime_group',
                'id': 'dropoff_cluster_count'})

        return X[['pickup_datetime_group', 'dropoff_cluster']] \
            .merge(
                dropoff_counts,
                on=['pickup_datetime_group', 'dropoff_cluster'],
                how='left'
            )['dropoff_cluster_count'].fillna(0)

    def fit(self, X, y=None):
        self.copy = X.filter(
            ['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster'],
            axis=1
        )
        return self

    def transform(self, X):
        # Count trips over 60min
        X['pickup_datetime_group'] = X['pickup_datetime'] \
            .dt.round(self.GROUP_FREQ)

        X = self.count_over_group_freq(X)
        X['dropoff_cluster_count'] = self.trip_count_between_clusters(X)

        X.drop('pickup_datetime_group', axis=1, inplace=True)

        return X


class FilterFeatures(NoFitEstimator, TransformerMixin):

    def __init__(self):
        self.do_not_use_for_training = [
            'id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime',
            'pickup_date', 'avg_speed_h'
        ]

    def transform(self, X):
        feature_names = [f for f in X.columns
                         if f not in self.do_not_use_for_training]

        return X[feature_names]


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
            ('coord_kmeans', CoordKMeans()),
            ('geospatial_agg', GeospatialAggregate()),
            ('count_features', CountFeatures()),
            ('filter_features', FilterFeatures()),
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

import pandas as pd
import data_cleaning

time_attrs = [
    'pickup_datetime',
    'dropoff_datetime',
]

taxi = pd.read_csv('data/train.csv', parse_dates=time_attrs)
taxi_cleaned = data_cleaning.pipeline.fit_transform(taxi)
print(taxi_cleaned[:10])

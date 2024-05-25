import pandas as pd

data = pd.read_csv("dataset.csv")

# convert date columns to proper datetime formats
data['tpep_pickup_datetime'] = pd.to_datetime(
    data['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
data['tpep_dropoff_datetime'] = pd.to_datetime(
    data['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')

# remove data with missing pickup and dropoff times
data = data.dropna(subset=['tpep_pickup_datetime'])
data = data.dropna(subset=['tpep_dropoff_datetime'])

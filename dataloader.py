import pandas as pd
import datetime 
import numpy as np
from scipy import stats
import os
import torch

def load_data():
    df = pd.read_csv(
    'ChargePoint Data CY20Q4.csv', dtype={'Station Name': str}, 
    parse_dates=['Start Date', 'Total Duration (hh:mm:ss)'], infer_datetime_format=True, low_memory=False)

    # Make a unique id for each row
    df['Id'] = df.index
    # Drop columns that are not needed
    df.drop(
        ['User ID', 'County', 'City', 'Postal Code', 'Driver Postal Code',
        'State/Province', 'Currency', 'EVSE ID', 'Transaction Date (Pacific Time)',
        'GHG Savings (kg)', 'Gasoline Savings (gallons)', 'Org Name',
        'Address 1', 'System S/N', 'Ended By',
        'End Time Zone', 'Start Time Zone', 'Country', 'Model Number'
        ], axis=1, inplace=True)

    df['Total Duration (min)'] = pd.to_timedelta(df['Total Duration (hh:mm:ss)']).dt.total_seconds()/60
    df['Charging Time (min)'] = pd.to_timedelta(df['Charging Time (hh:mm:ss)']).dt.total_seconds()/60
    df.drop(['Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)'], axis=1, inplace=True)

    df['End Date'] = df['End Date'].apply(convert_to_datetime)

    # Create clusters 
    df['Cluster'] = df['Station Name'].str.split(' ').str[4]
    return df

def convert_to_datetime(serial):
    # There's Excel formatting in the `End Data` column, so we need to convert from Excel serial to datetime
    # https://stackoverflow.com/questions/6706231/fetching-datetime-from-float-in-python
    try:
        return pd.to_datetime(serial)
    except:
        seconds = (float(serial) - 25569) * 86400.0
        return datetime.datetime.utcfromtimestamp(seconds)


def create_count_data(df, interval_length=30, save=False):
    """ Create counts for number of sessions in each interval. The `Period` defines when the period starts and runs until the next period in the dataframe."""
    df_combined = pd.DataFrame()
    for cluster in df['Cluster'].unique():
        # Collect each period that the charging session covers in the 'Period' column. This column will contain a list of datetimes :)
        df.loc[df.Cluster == cluster, 'Period'] = np.array([
            pd.date_range(s, e, freq='T') for s, e in zip(df[df.Cluster == cluster]['Start Date'], df[df.Cluster == cluster]['End Date'])
        ], dtype=object)
        # The the count of 
        res = df[df.Cluster == cluster].explode('Period').groupby(pd.Grouper(key='Period', freq=f'{interval_length}T'))['Id'].nunique()
        # rename value column to `Sessions``
        res.rename('Sessions', inplace=True)
        data = res.to_frame()
        data = data.reset_index()
        data.Period = pd.to_datetime(data.Period)
        data["Cluster"] = cluster
        df_combined = pd.concat([df_combined, data])
    df_combined.reset_index(drop=True, inplace=True)
    df_pivot = df_combined.pivot_table(index='Period', columns='Cluster', values='Sessions')
    # Cut timeseries off at the latest timepoint of the cluster with the earliest last timepoint
    df_pivot_reduced = df_pivot.loc[:df_combined.groupby('Cluster').agg({'Period': 'max'}).Period.min()].copy()
    # Cluster "SHERMAN" has no data before late 2021, so we drop it
    df_pivot_reduced.drop('SHERMAN', axis=1, inplace=True)
    # Fill missing values with 0
    df_pivot_reduced.fillna(0, inplace=True)
    if save:
        df_pivot_reduced.to_csv(f'charging_session_count_{interval_length}.csv')

    return df_pivot_reduced


def generate_dataset(data, seq_len, pred_len, time_len=None, split_ratio=0.8, normalize=False):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pred_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """

    # print parameters
    print('seq_len: ', seq_len)
    print('pred_len: ', pred_len)
    print('time_len: ', time_len)  
    print('split_ratio: ', split_ratio)

    # each row is a timepoint and each column is a cluster
    data = data
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()

    for i in range(len(train_data) - seq_len - pred_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pred_len]))

    for i in range(len(test_data) - seq_len - pred_len):
        test_X.append(np.array(test_data[i : i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pred_len]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(data, seq_len, pred_len, time_len=None, split_ratio=0.8, normalize=False):
    normalize=False
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pred_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset
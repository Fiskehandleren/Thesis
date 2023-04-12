import numpy as np
import pandas as pd
import datetime 

from torch.utils.data import Dataset
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import torch 
import networkx as nx
from haversine import haversine, Unit
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def load_data():
    path_processed = os.path.join(ROOT_PATH, '../data/ChargePoint Data CY20Q4_fixed_dates.csv')
    if os.path.exists(path_processed):
        df = pd.read_csv(path_processed,
                         dtype={
                            'Station Name' : str,
                            'Port Number': 'int8',
                            'Plug Type': str,
                            'Latitude': 'float32',
                            'Longitude':'float32',
                            'Total Duration (min)': 'float16',
                            'Charging Time (min)': 'float16',
                            'Cluster': str
                        },
                        parse_dates=['Start Date', 'End Date'], 
                        infer_datetime_format=True,
                        )
    else:
        path_unprocessed = os.path.join(ROOT_PATH, '../data/ChargePoint Data CY20Q4.csv')
        required_columns = ['Station Name', 'Start Date', 'End Date',
                            'Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)',
                            'Longitude', 'Latitude', 'Plug Type', 'Port Number'
                            ]
        df = pd.read_csv(path_unprocessed, 
                        dtype={'Station Name': str, 'Longitude': 'float32', 'Latitude': 'float32','Port Number': 'int8'}, 
                        parse_dates=['Start Date', 'Total Duration (hh:mm:ss)'], 
                        infer_datetime_format=True,
                        usecols=required_columns,
                        low_memory=False)

        # Make a unique id for each row
        df['Id'] = df.index

        df['Total Duration (min)'] = pd.to_timedelta(df['Total Duration (hh:mm:ss)']).dt.total_seconds()/60
        df['Charging Time (min)'] = pd.to_timedelta(df['Charging Time (hh:mm:ss)']).dt.total_seconds()/60
        df.drop(['Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)'], axis=1, inplace=True)

        df['End Date'] = df['End Date'].apply(convert_to_datetime)

        # Create clusters 
        df['Cluster'] = df['Station Name'].str.split(' ').str[4]
        df.to_csv(path_processed)

    return df

def convert_to_datetime(serial):
    # There's Excel formatting in the `End Data` column, so we need to convert from Excel serial to datetime
    # https://stackoverflow.com/questions/6706231/fetching-datetime-from-float-in-python
    try:
        return pd.to_datetime(serial)
    except:
        seconds = (float(serial) - 25569) * 86400.0
        return datetime.datetime.utcfromtimestamp(seconds)

def cyclical_encode(data: pd.DataFrame, col: str, max_val: int):
    """ Create cyclical cos and sin encoding for a column.
    Args:
        data: The dataframe to encode
        col: The column to encode
        max_val: The maximum value of the data type
    Returns:
        data: The dataframe with the new columns
        cols: The names of the new columns
    """

    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    data.drop(col, axis=1, inplace=True)
    return data, [col + '_sin', col + '_cos']


def create_count_data(df, interval_length=30, save=False, cap_recordings = False, censored = False):
    """ Create counts for number of sessions in each interval. The `Period` defines when the period starts and runs until the next period in the dataframe."""
    df_combined = pd.DataFrame()
    # Cluster "SHERMAN" has no data before late 2021, so we drop it
    df = df[df.Cluster != 'SHERMAN']
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
    # Fill missing values with 0
    df_pivot_reduced.fillna(0, inplace=True)

    if cap_recordings == True:
        raise NotImplementedError("Capping recordings is not implemented yet")

    if censored == True:
        df_pivot_reduced['CensoredSessions'] = np.zeros(len(df_pivot_reduced))
        df_pivot_reduced['CensoredSessions'][df_pivot_reduced['Sessions'] >= 4] = df_pivot_reduced['Sessions']-2
        df_pivot_reduced.to_csv(f'charging_session_count_{interval_length}_censored.csv')

    if save:
        df_pivot_reduced.to_csv(f'charging_session_count_{interval_length}.csv')

    return df_pivot_reduced

def get_graph(df, adjecency_threshold_km=3):
    G = nx.Graph()

    for idx, cluster in enumerate(df['Cluster'].unique()):
        if 'SHERMAN' in cluster:
            continue
        G.add_node(cluster)
        G.nodes[cluster]['ID'] = idx
        G.nodes[cluster]['lat'] = df[df['Cluster'] == cluster]['Latitude'].mean()
        G.nodes[cluster]['long'] = df[df['Cluster'] == cluster]['Longitude'].mean()
        G.nodes[cluster]['pos'] = (G.nodes[cluster]['long'], G.nodes[cluster]['lat'])

    for node_x in G.nodes:
        for node_y in G.nodes:
            dist = haversine(
                (G.nodes[node_x]['lat'], G.nodes[node_x]['long']),
                (G.nodes[node_y]['lat'], G.nodes[node_y]['long']),
                unit=Unit.KILOMETERS)
            # Assume that if nodes are further than adjecency_threshold_km km apart, their usage are not correlated
            if (dist > adjecency_threshold_km):
                continue
            # We might have to avoid setting self-connections here, e.g if node_x == node_y then continue
            # This is because the GCN requires A and not A_hat
            G.add_edge(node_x, node_y)
            G[node_x][node_y]['weight'] = np.exp(-dist)

    adj = nx.adjacency_matrix(G)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    return G, adj, edge_index, edge_weight.float()   


def get_targets_and_features_tgcn(
        df, node_names, sequence_length=30, forecast_lead=1,
        censored=True, add_month=True, add_hour=True, add_day_of_week=True, add_year=True):
    num_nodes = len(node_names)
    # By default we already shift the target by 1 timestep, so we only have to shift by additionaly 
    # forecast_leard - 1 steps
    forecast_lead -= 1

    df_test = df.copy()
    features, new_cols = [], []
    if add_month:
        df_test['month'] = df.Period.dt.month
        df_test, new_cols = cyclical_encode(df_test, 'month', 12)
        features = features + new_cols
    if add_day_of_week:
        df_test['dayofweek'] = df.Period.dt.dayofweek
        df_test, new_cols = cyclical_encode(df_test, 'dayofweek', 7)
        features = features + new_cols
    if add_hour:
        df_test['hour'] = df.Period.dt.hour
        df_test, new_cols = cyclical_encode(df_test, 'hour', 24)
        features = features + new_cols
    if add_year:
        # We subtract the minimum year to make the year start at 0
        df_test['year'] = df.Period.dt.year - df.Period.dt.year.min()
        features.append('year')
    
    # Get initial lagged features by taking the first `sequence_length` observations and treat
    # the `sequence_length`+1 observation as the target
    sessions_array = df_test[node_names].to_numpy(dtype=int)

    lag_feats = np.array([
        sessions_array[i : i + sequence_length, :].T
        for i in range(sessions_array.shape[0] - sequence_length)
    ])
    # Reshape to fit being concatenated with the datetime features
    lag_feats = lag_feats.reshape(-1, num_nodes, 1, sequence_length)

    sessions_array_shifted = df_test[node_names].shift(-forecast_lead, fill_value=-1).to_numpy(dtype=int) # -1 because the next line shifts by 1 by default
    y = np.array([
        sessions_array_shifted[i + sequence_length + forecast_lead, :].T
        for i in range(sessions_array_shifted.shape[0] - sequence_length - forecast_lead)
    ])

    time_features = df_test[features].to_numpy(dtype=int)

    times = np.array([
        [time_features[i : i + sequence_length, :].T]
        for i in range(time_features.shape[0] - sequence_length)
    ])
    # Repeat the time features 8 times because we have 8 nodes, and the 
    # period is the same across all nodes
    times = times.repeat(num_nodes, axis=1)

    if censored:
        _tau = df_test.filter(like='_TAU').shift(-forecast_lead, fill_value=-1).to_numpy(dtype=int)
        tau = np.array([
            _tau[i + sequence_length + forecast_lead, :].T
            for i in range(_tau.shape[0] - sequence_length - forecast_lead)
        ])
        _y_true = df_test.filter(like='_TRUE').shift(-forecast_lead, fill_value=-1).to_numpy(dtype=int)
        y_true = np.array([
            _y_true[i + sequence_length + forecast_lead, :].T
            for i in range(_y_true.shape[0] - sequence_length - forecast_lead)
        ])

    else:
        tau = None
        y_true = None
    
    # The `feat` matrix will go from (time_length, nodes, lags) to (time_length, nodes, number of features, lags)
    # We repeat the date-specific features 8 times because we have 8 nodes. 
    X = np.concatenate((lag_feats, times), axis=2).astype(np.float16)

    return X, y, tau, y_true

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def get_datasets_NN(target, forecast_lead, add_month=True, add_hour=True, add_day_of_week=True, add_year=True, train_start='2016-07-01', 
                    train_end='2017-07-01', test_end = '2017-08-01', val_end = '2017-09-01', is_censored = False, multiple_stations = False, censorship_level = 1):
    
    ## Function to load data sets, add covariates and split into training and test set. Has option to censor the input data (arg. is_censored) and 
    ## has option to use several stations to predict demand of one station (arg. multiple_stations)

    ## Output: training set, test set, list of explanatory variables (features) and list of targets (target)
    
    ## TODO: add option for validation set
    target_var = target

    ## if censored:
    path = os.path.join(ROOT_PATH, f'../data/charging_session_count_1_to_30_censored_{censorship_level}.csv')
    df = pd.read_csv(path, parse_dates=['Period'])

    df_test = df.copy()
    df_test[target_var + '_TAU'] = df_test[target_var + '_TAU'].shift(-forecast_lead)
    df_test[target_var + '_TRUE'] = df_test[target_var + '_TRUE'].shift(-forecast_lead)

    if multiple_stations:
        
        ## keep data from other stations, the period and threshold tau for target variable
        features = [v for v in df_test.columns if target + '_TAU' in v]
        other_stations = [v for v in df_test.columns if '_TAU' not in v]
        features.extend(other_stations)

        df_test = df_test[features]

        ## Remove tau so it isnt and input feature
        features.remove(target + '_TAU')

    else:
        features = [v for v in df_test.columns if target in v]
        features.append('Period')
        df_test = df_test[features]

        print(features)
        ## Remove tau so it isnt and input feature
        features.remove(target + '_TAU')

    '''
    else:
        ## keep everything from input dataframe
        path = os.path.join(ROOT_PATH, '../data/charging_session_count_1_to_30.csv')
        df = pd.read_csv(path, parse_dates=['Period'])
        
        df_test = df.copy()

        features = df_test.columns.values

        if not multiple_stations:
            ## Keep only data from target station and the period 
            features = [station for station in df_test.columns if target in station]
            features.append('Period')
            df_test = df_test[features]
'''
   
    ## create end points for dataset
    test_start = train_end + " 00:30:00"
    val_start = test_end + " 00:30:00"

    if (type(train_end) != int):
        train_start = df_test[df_test['Period'] == train_start].index.values[0]
        train_end = df_test[df_test['Period'] == train_end].index.values[0]
        test_start = df_test[df_test['Period'] == test_start].index.values[0]
        test_end = df_test[df_test['Period'] == test_end].index.values[0]
        val_start = df_test[df_test['Period'] == val_start].index.values[0]
        val_end = df_test[df_test['Period'] == val_end].index.values[0]

    # Create target variable. We might have more targets if we're running 
    # multivariate models

    if isinstance(target_var, list):
        target = [f"{var}_lead{forecast_lead}" for var in target_var]
    else:
        target = f"{target_var}_lead{forecast_lead}"
    
    ## Shift target variable(s)
    df_test[target] = df_test[target_var].shift(-forecast_lead)
    df_test = df_test.iloc[:-forecast_lead]


    new_cols = []
    if add_month:
        df_test['month'] = df.Period.dt.month
        df_test, new_cols = cyclical_encode(df_test, 'month', 12)
        features.extend(new_cols)
        #features = features + new_cols
    if add_day_of_week:
        df_test['dayofweek'] = df.Period.dt.dayofweek
        df_test, new_cols = cyclical_encode(df_test, 'dayofweek', 7)
        features.extend(new_cols)
        #features = features + new_cols
    if add_hour:
        df_test['hour'] = df.Period.dt.hour
        df_test, new_cols = cyclical_encode(df_test, 'hour', 24)
        features.extend(new_cols)
        #features = features + new_cols
    if add_year:
        df_test['year'] = df.Period.dt.year - df.Period.dt.year.min()
        features.append('year')


    ## Create train/test set
    df_train = df_test.loc[train_start:train_end].copy()
    df_val = df_test.loc[val_start:val_end].copy()
    df_test = df_test.loc[test_start:test_end].copy()

    features.remove('Period')

    #print("Test set fraction:", len(df_test) / len(df_train))
    return df_train, df_test, df_val, features, target


class SequenceDataset(Dataset):
    ## Class to retrieve time series elements appropirately with CENSORED target variable y
    def __init__(self, dataframe, target, features, threshold=None, true_target=None, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

        if threshold is not None:
            self.tau = torch.tensor(dataframe[threshold].values).float()
            self.y_true = torch.tensor(dataframe[true_target].values).float()
        else:
            self.tau = None
        
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        if self.tau is not None:
            return x, self.y[i], self.tau[i], self.y_true[i]
        else:
            return x, self.y[i]
import numpy as np
import pandas as pd
import datetime

from torch.utils.data import Dataset
import utils.constants
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import torch
import networkx as nx
from haversine import haversine, Unit
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def load_data():
    path_processed = os.path.join(
        ROOT_PATH, "../data/ChargePoint Data CY20Q4_fixed_dates.csv"
    )
    if os.path.exists(path_processed):
        df = pd.read_csv(
            path_processed,
            dtype={
                "Station Name": str,
                "Port Number": "int8",
                "Plug Type": str,
                "Latitude": "float32",
                "Longitude": "float32",
                "Total Duration (min)": "float16",
                "Charging Time (min)": "float16",
                "Cluster": str,
            },
            parse_dates=["Start Date", "End Date"],
            infer_datetime_format=True,
        )
    else:
        path_unprocessed = os.path.join(
            ROOT_PATH, "../data/ChargePoint Data CY20Q4.csv"
        )
        required_columns = [
            "Station Name",
            "Start Date",
            "End Date",
            "Total Duration (hh:mm:ss)",
            "Charging Time (hh:mm:ss)",
            "Longitude",
            "Latitude",
            "Plug Type",
            "Port Number",
        ]
        df = pd.read_csv(
            path_unprocessed,
            dtype={
                "Station Name": str,
                "Longitude": "float32",
                "Latitude": "float32",
                "Port Number": "int8",
            },
            parse_dates=["Start Date", "Total Duration (hh:mm:ss)"],
            infer_datetime_format=True,
            usecols=required_columns,
            low_memory=False,
        )

        # Make a unique id for each row
        df["Id"] = df.index

        df["Total Duration (min)"] = (
            pd.to_timedelta(df["Total Duration (hh:mm:ss)"]).dt.total_seconds() / 60
        )
        df["Charging Time (min)"] = (
            pd.to_timedelta(df["Charging Time (hh:mm:ss)"]).dt.total_seconds() / 60
        )
        df.drop(
            ["Total Duration (hh:mm:ss)", "Charging Time (hh:mm:ss)"],
            axis=1,
            inplace=True,
        )

        df["End Date"] = df["End Date"].apply(convert_to_datetime)

        # Create clusters
        df["Cluster"] = df["Station Name"].str.split(" ").str[4]
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
    """Create cyclical cos and sin encoding for a column.
    Args:
        data: The dataframe to encode
        col: The column to encode
        max_val: The maximum value of the data type
    Returns:
        data: The dataframe with the new columns
        cols: The names of the new columns
    """

    data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)
    data.drop(col, axis=1, inplace=True)
    return data, [col + "_sin", col + "_cos"]


def create_count_data(df, interval_length=30, save=False, cap_recordings=False):
    """Create counts for number of sessions in each interval. The `Period` defines when the period starts and runs until the next period in the dataframe."""
    df_combined = pd.DataFrame()
    # Cluster "SHERMAN" has no data before late 2021, so we drop it
    df = df[df.Cluster != "SHERMAN"]
    for cluster in df["Cluster"].unique():
        # Collect each period that the charging session covers in the 'Period' column. This column will contain a list of datetimes :)
        df.loc[df.Cluster == cluster, "Period"] = np.array(
            [
                pd.date_range(s, e, freq="T")
                for s, e in zip(
                    df[df.Cluster == cluster]["Start Date"],
                    df[df.Cluster == cluster]["End Date"],
                )
            ],
            dtype=object,
        )
        # The the count of
        res = (
            df[df.Cluster == cluster]
            .explode("Period")
            .groupby(pd.Grouper(key="Period", freq=f"{interval_length}T"))["Id"]
            .nunique()
        )
        # rename value column to `Sessions``
        res.rename("Sessions", inplace=True)
        data = res.to_frame()
        data = data.reset_index()
        data.Period = pd.to_datetime(data.Period)
        data["Cluster"] = cluster
        df_combined = pd.concat([df_combined, data])
    df_combined.reset_index(drop=True, inplace=True)
    df_pivot = df_combined.pivot_table(
        index="Period", columns="Cluster", values="Sessions"
    )
    # Cut timeseries off at the latest timepoint of the cluster with the earliest last timepoint
    df_pivot_reduced = df_pivot.loc[
        : df_combined.groupby("Cluster").agg({"Period": "max"}).Period.min()
    ].copy()
    # Fill missing values with 0
    df_pivot_reduced.fillna(0, inplace=True)

    if cap_recordings == True:
        raise NotImplementedError("Capping recordings is not implemented yet")

    if save:
        df_pivot_reduced.to_csv(f"charging_session_count_{interval_length}.csv")

    return df_pivot_reduced


def get_graph(df: pd.DataFrame, adjecency_threshold_km: float):
    """
    This function is used to generate a graph structure from a given DataFrame, where each node
    in the graph represents a cluster, and edges are formed between nodes if the clusters are
    within a certain distance threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the cluster data. Each row corresponds to a data point,
        and the DataFrame must include the columns "Cluster", "Latitude", and "Longitude",
        which represent the cluster name and its geographic coordinates respectively.

    adjecency_threshold_km : float
        The distance threshold (in kilometers) for establishing edges between nodes. If two
        clusters are separated by a distance greater than this threshold, no edge will be
        formed between them.

    Returns
    -------
    tuple
        A tuple containing four elements:
        G (nx.Graph): The generated graph, where nodes represent clusters and edges represent
            close proximity between clusters.
        adj (scipy.sparse.csr_matrix): The adjacency matrix of the graph.
        edge_index (torch.Tensor): The edge index tensor, which indicates the indices of
            the nodes that form each edge.
        edge_weight (torch.Tensor): The edge weight tensor, containing the weight of each edge,
            which is calculated as the exponential of the negative distance between two clusters.
    """
    G = nx.Graph()
    CLUSTERS = utils.constants.cluster_names
    for idx, cluster in enumerate(CLUSTERS):
        G.add_node(cluster)
        G.nodes[cluster]["ID"] = idx
        G.nodes[cluster]["lat"] = df[df["Cluster"] == cluster]["Latitude"].mean()
        G.nodes[cluster]["long"] = df[df["Cluster"] == cluster]["Longitude"].mean()
        G.nodes[cluster]["pos"] = (G.nodes[cluster]["long"], G.nodes[cluster]["lat"])

    for node_x in G.nodes:
        for node_y in G.nodes:
            dist = haversine(
                (G.nodes[node_x]["lat"], G.nodes[node_x]["long"]),
                (G.nodes[node_y]["lat"], G.nodes[node_y]["long"]),
                unit=Unit.KILOMETERS,
            )
            # Assume that if nodes are further than adjecency_threshold_km km apart, their usage are not correlated
            if dist > adjecency_threshold_km:
                continue
            # We might have to avoid setting self-connections here, e.g if node_x == node_y then continue
            # This is because the GCN requires A and not A_hat
            G.add_edge(node_x, node_y)
            G[node_x][node_y]["weight"] = np.exp(-dist)

    adj = nx.adjacency_matrix(G)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    return G, adj, edge_index, edge_weight.float()


def get_targets_and_features_tgcn(
    df,
    node_names,
    forecast_lead=1,
    add_month=True,
    add_hour=True,
    add_day_of_week=True,
    add_year=True,
):
    """
    This function generates targets and features for a Temporal Graph Convolutional Network (T-GCN).
    It takes as input a DataFrame and list of node names, and applies various transformations to
    generate the required input format for the TGCN. These transformations include shifting the target,
    adding cyclical time features such as month, day of the week, hour and year, and reshaping the data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must contain a 'Period' column of type datetime.

    node_names : list
        List of names of nodes in the graph.

    forecast_lead : int, optional
        Number of timesteps to shift the target data. Default is 1.

    add_month : bool, optional
        If True, add the month as a cyclical feature. Default is True.

    add_hour : bool, optional
        If True, add the hour as a cyclical feature. Default is True.

    add_day_of_week : bool, optional
        If True, add the day of the week as a cyclical feature. Default is True.

    add_year : bool, optional
        If True, add the year as a feature. Default is True.

    Returns
    -------
    tuple
        A tuple containing four elements:
        X (np.array): Input features for the TGCN. This array includes the cyclical time features
            and the previous sessions' data for each node.
        y (np.array): Target variable. This is the sessions data shifted by the forecast lead.
        tau (np.array): The Tau parameter shifted by the forecast lead.
        y_true (np.array): The true values of the target variable shifted by the forecast lead.
    """
    num_nodes = len(node_names)
    # By default we already shift the target by 1 timestep, so we only have to shift by additionaly
    # forecast_leard - 1 steps

    df_test = df.copy()
    features, new_cols = [], []
    if add_month:
        df_test["month"] = df.Period.dt.month
        df_test, new_cols = cyclical_encode(df_test, "month", 12)
        features = features + new_cols
    if add_day_of_week:
        df_test["dayofweek"] = df.Period.dt.dayofweek
        df_test, new_cols = cyclical_encode(df_test, "dayofweek", 7)
        features = features + new_cols
    if add_hour:
        df_test["hour"] = df.Period.dt.hour
        df_test, new_cols = cyclical_encode(df_test, "hour", 24)
        features = features + new_cols
    if add_year:
        # We subtract the minimum year to make the year start at 0
        df_test["year"] = df.Period.dt.year - df.Period.dt.year.min()
        features.append("year")
    # We shift the target by forecast_lead timesteps and remove the last forecast_lead timesteps
    # as we don't have the target for these timesteps
    y = df_test[node_names].shift(-forecast_lead).to_numpy(dtype=int)[:-forecast_lead].T

    tau = (
        df_test.filter(like="_TAU")
        .shift(-forecast_lead)
        .to_numpy(dtype=int)[:-forecast_lead]
        .T
    )

    y_true = (
        df_test.filter(like="_TRUE")
        .shift(-forecast_lead)
        .to_numpy(dtype=np.float32)[:-forecast_lead]
        .T
    )

    # Get the sessions for each node so we have [num_nodes, num_timesteps]
    sessions_array = df_test[node_names].to_numpy(dtype=int)[:-forecast_lead].T

    # Drop the last forecast_lead timesteps as we don't have the target for these timesteps
    time_features = df_test[features].to_numpy(dtype=int)[:-forecast_lead].T

    # Repeat the time features 8 times because we have 8 nodes, and the
    # period is the same across all nodes
    node_time_features = np.expand_dims(
        time_features, axis=0
    )  # Add new 1nd axis, so we can repeat this dim 8 times
    node_time_features = node_time_features.repeat(num_nodes, axis=0)

    # Reshape to fit being concatenated with the datetime features
    lag_feats = np.expand_dims(sessions_array, axis=1)

    # We repeat the date-specific features 8 times because we have 8 nodes.
    X = np.concatenate((lag_feats, node_time_features), axis=1).astype(np.float32)

    return X, y, tau, y_true


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def get_datasets_NN(
    target,
    forecast_lead,
    add_month=True,
    add_hour=True,
    add_day_of_week=True,
    add_year=True,
    train_start="2016-07-01",
    train_end="2017-07-01",
    test_end="2017-08-01",
    val_end="2017-09-01",
    multiple_stations=False,
    censorship_level=1,
    censor_dynamic=False,
):
    ## Function to load data sets, add covariates and split into training and test set. Has option to censor the input data (arg. is_censored) and
    ## has option to use several stations to predict demand of one station (arg. multiple_stations)

    ## Output: training set, test set, list of explanatory variables (features) and list of targets (target)

    ## TODO: add option for validation set
    target_var = target

    if censor_dynamic:
        path = os.path.join(
            ROOT_PATH,
            f"../data/charging_session_count_1_to_30_censored_{censorship_level}_dynamic.csv",
        )
    else:
        path = os.path.join(
            ROOT_PATH,
            f"../data/charging_session_count_1_to_30_censored_{censorship_level}.csv",
        )

    df = pd.read_csv(path, parse_dates=["Period"])
    df_time = df.copy()

    df[target_var + "_TAU"] = df[target_var + "_TAU"].shift(-forecast_lead)
    df[target_var + "_TRUE"] = df[target_var + "_TRUE"].shift(-forecast_lead)

    if multiple_stations:
        ## keep data from other stations, the period and threshold tau for target variable
        features = [v for v in df.columns if target + "_TAU" in v]
        other_stations = [v for v in df.columns if "_TAU" not in v]
        features.extend(other_stations)

        df = df[features]

        ## Remove tau so it isnt an input feature
        features.remove(target + "_TAU")
        features.remove(target + "_TRUE")

    else:
        features = [v for v in df.columns if target in v]
        features.append("Period")
        df = df[features]

        print(features)
        ## Remove tau and true valueso it isnt an input feature
        features.remove(target + "_TAU")
        features.remove(target + "_TRUE")

    ## create end points for dataset
    val_start = train_end + " 00:30:00"
    test_start = val_end + " 00:30:00"

    if type(train_end) != int:
        train_start = df[df["Period"] == train_start].index.values[0]
        train_end = df[df["Period"] == train_end].index.values[0]
        val_start = df[df["Period"] == val_start].index.values[0]
        val_end = df[df["Period"] == val_end].index.values[0]
        test_start = df[df["Period"] == test_start].index.values[0]
        test_end = df[df["Period"] == test_end].index.values[0]

    # Create target variable. We might have more targets if we're running
    # multivariate models

    if isinstance(target_var, list):
        target = [f"{var}_lead{forecast_lead}" for var in target_var]
    else:
        target = f"{target_var}_lead{forecast_lead}"

    ## Shift target variable(s)
    df[target] = df[target_var].shift(-forecast_lead)
    df = df.iloc[:-forecast_lead]

    new_cols = []
    if add_month:
        df["month"] = df_time.Period.dt.month
        df, new_cols = cyclical_encode(df, "month", 12)
        features.extend(new_cols)
    if add_day_of_week:
        df["dayofweek"] = df_time.Period.dt.dayofweek
        df, new_cols = cyclical_encode(df, "dayofweek", 7)
        features.extend(new_cols)
    if add_hour:
        df["hour"] = df_time.Period.dt.hour
        df, new_cols = cyclical_encode(df, "hour", 24)
        features.extend(new_cols)
    if add_year:
        df["year"] = df_time.Period.dt.year - df_time.Period.dt.year.min()
        features.append("year")

    ## Create train/test set
    df_train = df.loc[train_start:train_end].copy()
    df_val = df.loc[val_start:val_end].copy()
    df_test = df.loc[test_start:test_end].copy()

    features.remove("Period")

    return df_train, df_test, df_val, features, target


class SequenceDataset(Dataset):
    ## Class to retrieve time series elements appropirately with CENSORED target variable y
    def __init__(
        self,
        dataframe,
        target,
        features,
        threshold=None,
        true_target=None,
        sequence_length=5,
        forecast_horizon=2,
    ):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

        self.tau = torch.tensor(dataframe[threshold].values).float()
        self.y_true = torch.tensor(dataframe[true_target].values).float()

    def __len__(self):
        return self.X.shape[0] - self.sequence_length

    def __getitem__(self, i):
        i += self.sequence_length - 1
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
        else:
            print("WARNING! USING PADDINGs")
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        y_start = i
        y_end = y_start + self.forecast_horizon

        if y_end > self.y.shape[0]:
            pad_length = y_end - self.y.shape[0]

            y_values = self.y[y_start:]
            y_padding = y_values[-1].repeat(pad_length)
            y_values = torch.cat((y_values, y_padding))

            tau_values = self.tau[y_start:]
            tau_padding = tau_values[-1].repeat(pad_length)
            tau_values = torch.cat((tau_values, tau_padding))

            y_true_values = self.y_true[y_start:]
            y_true_padding = y_true_values[-1].repeat(pad_length)
            y_true_values = torch.cat((y_true_values, y_true_padding))
        else:
            y_values = self.y[y_start:y_end]
            tau_values = self.tau[y_start:y_end]
            y_true_values = self.y_true[y_start:y_end]

        return x, y_values, tau_values, y_true_values

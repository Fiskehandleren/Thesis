from torch.utils.data import DataLoader, Dataset, TensorDataset
import pytorch_lightning as pl
import pandas as pd
import os
import utils.dataloader as dataloader
import argparse
import torch

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

class EVChargersDatasetSpatial(pl.LightningDataModule):
    def __init__(
        self,
        covariates: bool,
        batch_size: int,
        sequence_length: int,
        cluster: str,
        train_start: str,
        train_end: str,
        test_end: str,
        val_end: str,
        censored: bool,
        **kwargs
    ):
        super().__init__()
        self.coverariates = covariates
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = train_end + " 00:30:00"
        self.test_end = test_end
        self.val_start = test_end + " 00:30:00"
        self.val_end = val_end
        self.censored = censored

        self.df = dataloader.load_data()
        # TODO add specific censor strategy
        dataset_name = f'../data/charging_session_count_1_to_30{"_censored_4" if self.censored else ""}.csv'
        if not os.path.exists(os.path.join(ROOT_PATH, dataset_name)):
            print(f'Dataset "{dataset_name}" not found locally. Creating dataset...')
            self._feat = dataloader.create_count_data(self.df, 30, save=True, censored=self.censored)
        else:
            self._feat = pd.read_csv(os.path.join(ROOT_PATH, dataset_name), parse_dates=['Period'])

        if cluster is not None:
            self._feat = self._feat[cluster]
            self.cluster_names = [cluster]
        else:
            self.cluster_names = self.df['Cluster'].unique()
            self.cluster_names = self.cluster_names[self.cluster_names != 'SHERMAN']

        # Load node features
        X, y, tau = dataloader.get_targets_and_features_tgcn(
            self._feat,
            lags=self.sequence_length, # TODO not sure if this is the same way as nonspatial is done? 
            censored=self.censored,
            add_month=self.coverariates,
            add_day_of_week=self.coverariates,
            add_hour=self.coverariates,
            add_year=self.coverariates)

        train_start_index = self._feat[(self._feat.Period >= self.train_start)].index[0]
        train_end_index = self._feat[(self._feat.Period >= self.train_end)].index[0]
        test_end_index = self._feat[self._feat.Period >= self.test_end].index[0]
        val_start_index = self._feat[(self._feat.Period >= self.val_start)].index[0]
        val_end_index = self._feat[(self._feat.Period >= self.val_end)].index[0]

        # Grab training data from the start of the dataset to the start of the test set
        self.X_train, self.y_train = X[train_start_index : train_end_index], y[train_start_index : train_end_index]
        self.X_val, self.y_val = X[val_start_index : val_end_index], y[val_start_index : val_end_index]
        self.X_test, self.y_test = X[train_end_index : test_end_index], y[train_end_index : test_end_index]

        if self.censored and tau is not None:
            self.tau_train, self.tau_test = tau[train_start_index:train_end_index], tau[train_end_index:]

        G, adj, self.edge_index, self.edge_weight = dataloader.get_graph(self.df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def setup(self, stage=None):
        if self.censored:
            self.train_dataset = CensoredSpatialDataset(
                torch.FloatTensor(self.X_train), torch.FloatTensor(self.y_train), torch.FloatTensor(self.tau_train)
            )
            self.val_dataset = CensoredSpatialDataset(
                torch.FloatTensor(self.X_test), torch.FloatTensor(self.y_test), torch.FloatTensor(self.tau_test)
            )
        else:
            self.train_dataset = TensorDataset(
                torch.FloatTensor(self.X_train), torch.FloatTensor(self.y_train)
            )
            self.val_dataset = TensorDataset(
                torch.FloatTensor(self.X_test), torch.FloatTensor(self.y_test)
            )

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--covariates", help="Add covariates to the dataset", default=False, action='store_true')
        parser.add_argument("--cluster", type=str, help="Which cluster to fit an AR model to")
        parser.add_argument("--censored", default=False, action='store_true')
        parser.add_argument("--censor_level", default = 1, help = "Choose censorship level")
        parser.add_argument("--sequence_length",  type=int, default = 72)
        parser.add_argument("--train_start", type=str, required=True)
        parser.add_argument("--train_end", type=str, required=True)
        parser.add_argument("--test_end", type=str, required=True)
        parser.add_argument("--val_end", type=str, required=True)

        return parser


class CensoredSpatialDataset(Dataset):
    def __init__(self, X, y, tau):
        self.X = X
        self.y = y
        self.tau = tau

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.tau[i]
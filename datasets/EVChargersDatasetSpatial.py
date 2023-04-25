from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
import os
import argparse
import numpy as np
import torch
import multiprocessing as mp

import utils.dataloader as dataloader

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

class EVChargersDatasetSpatial(pl.LightningDataModule):
    def __init__(
        self,
        covariates: bool,
        batch_size: int,
        sequence_length: int,
        forecast_lead: int,
        cluster: str,
        train_start: str,
        train_end: str,
        test_end: str,
        val_end: str,
        censored: bool,
        censor_level: int,
        censor_dynamic: bool,
        **kwargs
    ):
        super().__init__()
        self.coverariates = covariates
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.forecast_lead = forecast_lead
        self.train_start = train_start
        self.train_end = train_end

        self.val_start = train_end + " 00:30:00"
        self.val_end = val_end
        
        self.test_start = val_end + " 00:30:00"
        self.test_end = test_end

        self.censored = censored
        self.censor_dynamic = censor_dynamic

        self.num_workers = mp.cpu_count()

        self.df = dataloader.load_data()
        if censor_dynamic:
            dataset_name = f'../data/charging_session_count_1_to_30{f"_censored_{censor_level}" if self.censored else ""}_dynamic.csv'
        else: 
            dataset_name = f'../data/charging_session_count_1_to_30{f"_censored_{censor_level}" if self.censored else ""}.csv'
        if not os.path.exists(os.path.join(ROOT_PATH, dataset_name)):
            print(f'Dataset "{dataset_name}" not found locally. Creating dataset...')
            self._feat = dataloader.create_count_data(self.df, 30, save=True)
        else:
            self._feat = pd.read_csv(os.path.join(ROOT_PATH, dataset_name), parse_dates=['Period'])

        if cluster is not None:
            self._feat = self._feat[cluster].to_frame()
            self.cluster_names = np.array([cluster])
        else:
            self.cluster_names = self._feat.columns[~(self._feat.columns.str.contains('_TAU')) &  ~(self._feat.columns.str.contains('_TRUE'))]
            self.cluster_names = self.cluster_names[(self.cluster_names != 'SHERMAN') & (self.cluster_names != 'Period')]
            self.cluster_names = self.cluster_names.to_numpy(dtype=str)

        # Load node features
        X, y, tau, y_true = dataloader.get_targets_and_features_tgcn(
            self._feat,
            node_names=self.cluster_names,
            sequence_length=self.sequence_length,
            forecast_lead=self.forecast_lead,
            add_month=self.coverariates,
            add_day_of_week=self.coverariates,
            add_hour=self.coverariates,
            add_year=self.coverariates)

        train_start_index = self._feat[(self._feat.Period == self.train_start)].index[0]
        train_end_index = self._feat[(self._feat.Period == self.train_end)].index[0]

        val_start_index = self._feat[(self._feat.Period == self.val_start)].index[0]
        val_end_index = self._feat[(self._feat.Period == self.val_end)].index[0]

        test_start_index = self._feat[self._feat.Period == self.test_start].index[0]
        test_end_index = self._feat[self._feat.Period == self.test_end].index[0]
        
        # Grab training data from the start of the dataset to the start of the test set
        self.X_train, self.y_train = X[:, :, train_start_index : train_end_index], y[:, train_start_index : train_end_index]
        self.X_val, self.y_val = X[:, :, val_start_index : val_end_index], y[:, val_start_index : val_end_index]
        self.X_test, self.y_test = X[:, :, test_start_index : test_end_index], y[:, test_start_index : test_end_index]

        self.y_dates = self._feat[test_start_index : test_end_index].Period.to_numpy()

        self.tau_train, self.tau_test = tau[:, train_start_index:train_end_index], tau[:, train_end_index:]
        self.y_train_true, self.y_test_true = y_true[:, train_start_index:train_end_index], y_true[:, train_end_index:]

        _, _, self.edge_index, self.edge_weight = dataloader.get_graph(self.df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def setup(self, stage=None):
        self.train_dataset = SequenceSpatialDataset(
            torch.FloatTensor(self.X_train), torch.FloatTensor(self.y_train),
            torch.FloatTensor(self.tau_train), torch.FloatTensor(self.y_train_true),
            self.sequence_length
        )
        self.val_dataset = SequenceSpatialDataset(
            torch.FloatTensor(self.X_test), torch.FloatTensor(self.y_test),
            torch.FloatTensor(self.tau_test), torch.FloatTensor(self.y_test_true),
            self.sequence_length
        )
        self.test_dataset = SequenceSpatialDataset(
            torch.FloatTensor(self.X_test), torch.FloatTensor(self.y_test),
            torch.FloatTensor(self.tau_test), torch.FloatTensor(self.y_test_true),
            self.sequence_length
        )
       
    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        return parser


class SequenceSpatialDataset(Dataset):
    ## Class to retrieve time series elements appropirately with CENSORED target variable y
    def __init__(self, X, y, tau, y_true, sequence_length):
        self.sequence_length = sequence_length
        self.X = X
        self.y = y
        self.tau = tau
        self.y_true = y_true

    def __len__(self):
        return self.X.shape[2]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[:, :, i_start:(i + 1)]
        else:
            # If we are at the beginning of the time series, we need to pad the sequence with the first element
            # until we have a total sequence of length sequence_length 
            padding = self.X[:, :, 0].unsqueeze(2).repeat_interleave(self.sequence_length - i - 1, dim=2)
            x = self.X[:, :, 0:(i + 1)]
            x = torch.cat((padding, x), 2)

        y_start = i
        y_end = y_start + self.forecast_horizon

        # If we are at the end of the time series, we need to pad the sequence with the last element
        if y_end > self.y.shape[1]:
            pad_length = y_end - self.y.shape[1]

            y_values = self.y[:, y_start:]
            y_padding = y_values[:, -1].unsqueeze(1).repeat_interleave(pad_length, dim=1)
            y_values = torch.cat((y_values, y_padding), 1)

            tau_values = self.tau[:, y_start:]
            tau_padding = tau_values[:, -1].unsqueeze(1).repeat_interleave(pad_length, dim=1)
            tau_values = torch.cat((tau_values, tau_padding), 1)

            y_true_values = self.y_true[:, y_start:]
            y_true_padding = y_true_values[:, -1].unsqueeze(1).repeat_interleave(pad_length, dim=1)
            y_true_values = torch.cat((y_true_values, y_true_padding), 1)
        else:
            # If we are not at the end of the time series, we can just take the next forecast_horizon values
            y_values = self.y[:, y_start:y_end]
            tau_values = self.tau[:, y_start:y_end]
            y_true_values = self.y_true[:, y_start:y_end]

        return x, y_values, tau_values, y_true_values

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
        forecast_horizon: int,
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
        self.covariates = covariates
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.forecast_lead = forecast_lead
        self.forecast_horizon = forecast_horizon
        self.censor_level = censor_level
        # Set the train, val and test dates
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
        dataset_name = self.get_dataset_name()

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
            forecast_lead=self.forecast_lead,
            add_month=self.covariates,
            add_day_of_week=self.covariates,
            add_hour=self.covariates,
            add_year=self.covariates)

        train_start_index, train_end_index = self.get_indices(self.train_start, self.train_end)
        val_start_index, val_end_index = self.get_indices(self.val_start, self.val_end)
        test_start_index, test_end_index = self.get_indices(self.test_start, self.test_end)
        
        # Grab training data from the start of the dataset to the start of the test set
        self.X_train, self.y_train = X[:, :, train_start_index : train_end_index], y[:, train_start_index : train_end_index]
        self.X_val, self.y_val = X[:, :, val_start_index : val_end_index], y[:, val_start_index : val_end_index]
        self.X_test, self.y_test = X[:, :, test_start_index : test_end_index], y[:, test_start_index : test_end_index]

        self.y_dates = self._feat[test_start_index : test_end_index].Period.to_numpy()

        self.tau_train, self.tau_test, self.tau_val = tau[:, train_start_index : train_end_index], tau[:, test_start_index : test_end_index], tau[:, val_start_index : val_end_index]
        self.y_train_true, self.y_test_true, self.y_val_true = y_true[:, train_start_index : train_end_index], y_true[:, test_start_index : test_end_index], y_true[:, val_start_index : val_end_index]

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
            self.sequence_length, self.forecast_horizon
        )
        self.val_dataset = SequenceSpatialDataset(
            torch.FloatTensor(self.X_val), torch.FloatTensor(self.y_val),
            torch.FloatTensor(self.tau_val), torch.FloatTensor(self.y_val_true),
            self.sequence_length, self.forecast_horizon
        )
        self.test_dataset = SequenceSpatialDataset(
            torch.FloatTensor(self.X_test), torch.FloatTensor(self.y_test),
            torch.FloatTensor(self.tau_test), torch.FloatTensor(self.y_test_true),
            self.sequence_length, self.forecast_horizon
        )
       
    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        return parser
    
    def get_indices(self, start_date, end_date):
        start_index = self._feat[(self._feat.Period == start_date)].index[0]
        end_index = self._feat[(self._feat.Period == end_date)].index[0]
        return start_index, end_index
    
    def get_dataset_name(self):
        if self.censored:
            censor_level = f'{self.censor_level}' 
        else:
            censor_level = ''
        if self.censor_dynamic:
            return f'../data/charging_session_count_1_to_30{f"_censored_{censor_level}" if self.censored else ""}_dynamic.csv'
        else:
            return f'../data/charging_session_count_1_to_30{f"_censored_{censor_level}" if self.censored else ""}.csv'


class SequenceSpatialDataset(Dataset):
    def __init__(self, X, y, tau, y_true, sequence_length, forecast_horizon):
        self.sequence_length = sequence_length
        self.X = X
        self.y = y
        self.tau = tau
        self.y_true = y_true
        self.forecast_horizon = forecast_horizon

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

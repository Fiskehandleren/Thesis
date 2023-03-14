from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import os
import utils.dataloader as dataloader
import argparse
import torch


class EVChargersDataset(pl.LightningDataModule):
    def __init__(
        self,
        feat_path: str,
        covariates: bool,
        batch_size: int,
        lags: int,
        session_minutes: int,
        cluster: str,
        spatial: bool,
        train_start: str,
        test_start: str,
        **kwargs
    ):
        super().__init__()
        self._feat_path = feat_path
        self.coverariates = covariates
        self.batch_size = batch_size
        self.lags = lags
        self.spatial = spatial
        self.train_start = train_start
        self.test_start = test_start

        dataset_name = f'charging_session_count_{session_minutes}.csv'
        if not os.path.exists(os.path.join(feat_path, dataset_name)):
            print('Dataset not found locally. Creating dataset...')
            self.df = dataloader.load_data()
            self._feat = dataloader.create_count_data(self.df, session_minutes, save=True)
        else:
            self._feat = pd.read_csv(os.path.join(feat_path, dataset_name), parse_dates=['Period'])

        if cluster is not None:
            self._feat = self._feat[cluster]

        if self.spatial:
            # Load node features
            X, y = dataloader.get_targets_and_features_tgcn(
                self._feat,
                lags=self.lags,
                add_month=self.coverariates,
                add_day_of_week=self.coverariates,
                add_hour=self.coverariates,
                add_year=self.coverariates)
            train_start_index = self._feat[self._feat.Period >= self.train_start].index[0]
            test_start_index = self._feat[self._feat.Period >= self.test_start].index[0]
            
            # Grab training data from the start of the dataset to the start of the test set
            self.X_train, self.y_train = X[train_start_index:test_start_index], y[train_start_index:test_start_index]
            self.X_test, self.y_test = X[test_start_index:], y[test_start_index:]
            self.df = dataloader.load_data()
            G, adj, self.edge_index, self.edge_weight = dataloader.get_graph(self.df)

        else:
            raise NotImplementedError('Non-spatial data-loading not implemented yet.')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=False)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)


    def setup(self, stage=None):
        self.train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.X_train), torch.FloatTensor(self.y_train)
        )
        self.val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.X_test), torch.FloatTensor(self.y_test)
        )

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--covariates", help="Add covariates to the dataset", type=bool, default=False)
        parser.add_argument("--cluster", type=str, help="Which cluster to fit an AR model to")
        parser.add_argument("--spatial", type=bool, default=True)
        parser.add_argument("--lags", type=int, default=30)
        parser.add_argument("--session_minutes", type=int, default=30)
        parser.add_argument("--train_start", type=str, required=True)
        parser.add_argument("--test_start", type=str, required=True)

        return parser

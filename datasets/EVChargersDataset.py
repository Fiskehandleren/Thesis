from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import os
import dataloader
import argparse


class EVChargersDataset(pl.LightningDataModule):
    def __init__(
        self,
        feat_path: str,
        adj_path: str,
        covariates: bool,
        batch_size: int,
        seq_len: int,
        pred_len: int,
        split_ratio: float,
        normalize: bool,
        session_minutes: int,
        cluster: str,
        **kwargs
    ):
        super(EVChargersDataset, self).__init__()
        self._feat_path = feat_path
        self._adj_path = adj_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self.coverariates = covariates

        dataset_name = f'charging_session_count_{session_minutes}.csv'
        if not os.path.exists(os.path.join(feat_path, dataset_name)):
            print('Dataset not found locally. Creating dataset...')
            self._feat = dataloader.create_count_data(dataloader.load_data(), session_minutes, save=True)
        else:
            self._feat = pd.read_csv(os.path.join(feat_path, dataset_name)).drop('Period', axis=1)

        if cluster is not None:
            self._feat = self._feat[cluster]
        self._feat = self._feat.to_numpy()[:2*24*60] # 60 days TMP FIX

        if covariates == True:
            raise NotImplementedError('Covariates not implemented yet.')

        # Find max value in the data for normalization
        self._feat_max_val = np.max(self._feat)
        if self.normalize:
            self._feat = self._feat / self._feat_max_val
        self.adj = np.load(os.path.join(self._adj_path, 'distance_matrix.npy'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=False)

    def setup(self, stage=None):
        (self.train_dataset, self.val_dataset) = dataloader.generate_torch_datasets(
            self._feat,
            self.seq_len,
            self.pred_len,
            split_ratio=self.split_ratio,
            normalize=self.normalize,
        )    

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--seq_len", type=int, default=2*24*7)
        parser.add_argument("--pred_len", type=int, default=2*24)
        parser.add_argument("--split_ratio", type=float, default=0.8)
        parser.add_argument("--normalize", type=bool, default=True)
        parser.add_argument("--covariates", help="Add covariates to the dataset", type=bool, default=False)
        parser.add_argument("--session-minutes", type=int, help="The length of a charging session in minutes", default=30)
        parser.add_argument("--cluster", type=str, help="Which cluster to fit an AR model to")

        return parser

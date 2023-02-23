import numpy as np
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import logging
import argparse
import os
import pytorch_lightning as pl
from dataloader import generate_torch_datasets
import models
from tasks import AR_Task

logger = logging.getLogger('Thesis.Train')

class TimeSeriesDataset(pl.LightningDataModule):
    def __init__(
        self,
        feat_path: str,
        adj_path: str,
        batch_size: int = 1,
        seq_len: int = 2*24*7,
        pred_len: int = 2*24,
        split_ratio: float = 0.8,
        normalize: bool = True,
        session_minutes: int = 30,
        **kwargs
    ):
        super(TimeSeriesDataset, self).__init__()
        self._feat_path = feat_path
        self._adj_path = adj_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pred_len
        self.split_ratio = split_ratio
        self.normalize = normalize

        self._feat = pd.read_csv(os.path.join(feat_path, f'charging_session_count_{session_minutes}.csv')).drop('Period', axis=1).to_numpy()
        self._feat_max_val = np.max(self._feat)
        self._adj = None #utils.data.functions.load_adjacency_matrix(self._adj_path)

        #covariates = generate_covariates(df[train_start:test_end].index, 4)

        #train_data = df[train_start:train_end].values
        #test_data = df[test_start:test_end].values

        # data_start = (train_data!=0).argmax(axis=0) #find first nonzero value in each time series

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

    def setup(self, stage=None):
        (self.train_dataset, self.val_dataset) = generate_torch_datasets(
            self._feat,
            self.seq_len,
            self.pre_len,
            split_ratio=self.split_ratio,
            normalize=self.normalize,
        )    

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--seq_len", type=int, default=2*24*7)
        parser.add_argument("--pre_len", type=int, default=2*24)
        parser.add_argument("--split_ratio", type=float, default=0.8)
        parser.add_argument("--normalize", type=bool, default=True)
        return parser

def get_model(args, dm):
    model = None
    if args.model_name == "AR":
        model = models.AR(input_dim=2*24*7, output_dim=2*24)
    elif args.model_name == "AR-Net":
        model = models.AR_Net(input_dim=8, output_dim=args.output_dim, hidden_dim=args.hidden_dim)

    #if args.model_name == "GRU":
        #model = models.GRU(input_dim=dm.adj.shape[0], hidden_dim=args.hidden_dim)
    #if args.model_name == "TGCN":
       #model = models.TGCN(adj=dm.adj, hidden_dim=args.hidden_dim)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--data",
        type=str,
        help="The name of the dataset",
        choices=("temporal", "temporal+spatial"),
        default="temporal"
    )
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("AR", "AR-Net", "DeepAR", "T-GCN"),
        default="AR",
    )
    temp_args, _ = parser.parse_known_args()
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)

    parser = TimeSeriesDataset.add_data_specific_arguments(parser)

    args = parser.parse_args()

    dm = TimeSeriesDataset(
        feat_path='data', adj_path="TODO", **vars(args)
    )
    model = get_model(args, dm)
    
    task = AR_Task(model=model, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(task, dm)

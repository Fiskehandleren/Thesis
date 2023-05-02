import argparse
import torch.optim
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN2
from pytorch_lightning import LightningModule
import numpy as np
from utils.losses import get_loss_metrics

class TGCN(LightningModule):
    def __init__(
        self,
        loss_fn,
        edge_index,
        edge_weight,
        node_features: int,
        forecast_horizon:int,
        sequence_length: int,
        hidden_dim: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        censored= False,
        no_self_loops=False,
        **kwargs
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.censored = censored
        # Hyperparameters
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        TGCN.get_loss_metrics = get_loss_metrics

        # We add improved self-loops for each node, to make sure that the nodes are weighing themselves
        # more than their neighbors. `improved=True` means that A_hat = A + 2I, so the diagonal is 3.
        self.tgcn_cell = TGCN2(node_features, hidden_dim, add_self_loops=True, improved=not no_self_loops, batch_size=batch_size)
        self.linear = torch.nn.Linear(hidden_dim, forecast_horizon)

        # To save predictions and their true values for visualizations
        self.test_y = np.empty((0, 8, forecast_horizon))
        self.test_y_hat = np.empty((0, 8, forecast_horizon))
        self.test_y_true = np.empty((0, 8, forecast_horizon))

        self.save_hyperparameters(ignore=["edge_index", "edge_weight", "node_features", "loss_fn"])

    def forward(self, x, edge_index, edge_weight):
        # X is shape (Batch Size, Nodes, Features, Sequence Length)
        h = None # Maybe initialize randomly?
        # Go over each 
        for i in range(self.sequence_length):
            # Each X_t is of shape (Batch Size, Nodes, Features)
            h = self.tgcn_cell(x[:,:,:,i], edge_index, edge_weight, h)

        y = self.linear(h)
        return y.exp(), h
    
    def _get_preds_loss_metrics(self, batch, stage):        
        y_hat = self._get_preds(batch)
        return self.get_loss_metrics(batch, y_hat, stage)
    
    def _get_preds(self, batch):
        x = batch[0]
        # Transfer graph stuff to device
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)
        # Make predictions
        y_hat, _ = self(x, self.edge_index, self.edge_weight)
        return y_hat

    def training_step(self, batch, batch_idx):
        loss_metrics, _, _, _ = self._get_preds_loss_metrics(batch, "train")
        self.log_dict(loss_metrics, prog_bar=True)
        return loss_metrics["train_loss"]
    
    def validation_step(self, batch, batch_idx):
        loss_metrics, _, _, _ = self._get_preds_loss_metrics(batch, "val")
        self.log_dict(loss_metrics, prog_bar=True)
        return loss_metrics["val_loss"]

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        y_hat = self._get_preds(batch)
        _, y, _, y_true = batch
        self.test_y = np.concatenate((self.test_y, y.cpu().detach().numpy()))
        self.test_y_hat = np.concatenate((self.test_y_hat, y_hat.cpu().detach().numpy()))
        if self.censored:
            self.test_y_true = np.concatenate((self.test_y_true, y_true.cpu().detach().numpy()))
        return y_true

    def test_step(self, batch, batch_idx):
        loss_metrics, y, y_true, y_hat = self._get_preds_loss_metrics(batch, "test")
        self.log_dict(loss_metrics, on_epoch=True, on_step=False, prog_bar=True)
        self.test_y = np.concatenate((self.test_y, y.cpu().detach().numpy()))
        self.test_y_hat = np.concatenate((self.test_y_hat, y_hat.cpu().detach().numpy()))
        if self.censored:
            self.test_y_true = np.concatenate((self.test_y_true, y_true.cpu().detach().numpy()))
        return loss_metrics["test_loss"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--no_self_loops", action='store_true', default = False, help= "Censor data at cap. tau")
        return parser
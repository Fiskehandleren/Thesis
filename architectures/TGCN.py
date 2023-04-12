import argparse
import torch.optim
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN2
from pytorch_lightning import LightningModule
from torch import sqrt
import numpy as np
from torch import nn
import pandas as pd

class TGCN(LightningModule):
    def __init__(
        self,
        loss_fn,
        edge_index,
        edge_weight,
        node_features: int,
        sequence_length: int,
        hidden_dim: int,
        batch_size: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        censored= False,
        **kwargs
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.censored = censored
        # Hyperparameters
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # We add improved self-loops for each node, to make sure that the nodes are weighing themselves
        # more than their neighbors. `improved=True` means that A_hat = A + 2I, so the diagonal is 3.
        self.tgcn_cell = TGCN2(node_features, self.hidden_dim, add_self_loops=True, improved=True, batch_size=batch_size)
        self.linear = torch.nn.Linear(hidden_dim, 1)

        # To save predictions and their true values for visualizations
        self.test_y = np.empty((0, 8))
        self.test_y_hat = np.empty((0, 8))

        self.save_hyperparameters(ignore=["loss_fn", "edge_index", "edge_weight"])

    def forward(self, x, edge_index, edge_weight):
        h = None # Maybe initialize randomly?
        # Go over each 
        for i in range(self.sequence_length):
            # Each X_t is of shape (Batch Size, Nodes, Features)
            h = self.tgcn_cell(x[:,:,:,i], edge_index, edge_weight, h)

        y = F.relu(h)
        y = self.linear(h)
        return y.exp(), h
    
    def _get_preds_loss_metrics(self, batch, stage):
        if self.censored:
            x, y, tau, y_true = batch
        else:
            x, y = batch

        # Transfer to device
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)
        
        y_hat, _ = self(x, self.edge_index, self.edge_weight)
        y_hat = y_hat.view(-1, x.shape[1])
        if self.censored:
            loss = self.loss_fn(y_hat, y, tau)
            loss_uncen = nn.PoissonNLLLoss(log_input=False)
            loss_true = loss_uncen(y_hat, y_true)

            mse = F.mse_loss(y_hat, y_true)
            mae = F.l1_loss(y_hat, y_true) 
        else:
            loss = self.loss_fn(y_hat, y)
            loss_true = loss
            mse = F.mse_loss(y_hat, y)
            mae = F.l1_loss(y_hat, y)  

        return {
            f"{stage}_loss": loss,
            f"{stage}_loss_true": loss_true,
            f"{stage}_mae": mae,
            f"{stage}_rmse": sqrt(mse),
            f"{stage}_mse": mse,
        }, y if not self.censored else y_true, y_hat
    
    def training_step(self, batch, batch_idx):
        loss_metrics, _, _= self._get_preds_loss_metrics(batch, "train")
        #self.log_dict(loss_metrics, prog_bar=True, on_epoch=True, on_step=False)
        return loss_metrics["train_loss"]
    
    def validation_step(self, batch, batch_idx):
        loss_metrics, _, _ = self._get_preds_loss_metrics(batch, "val")
        self.log_dict(loss_metrics, prog_bar=True, on_epoch=True, on_step=False)
        return loss_metrics["val_loss"]
    
    def test_step(self, batch, batch_idx):
        loss_metrics, y, y_hat = self._get_preds_loss_metrics(batch, "test")
        self.log_dict(loss_metrics, on_epoch=True, on_step=False, prog_bar=True)
        self.test_y = np.concatenate((self.test_y, y.cpu().detach().numpy()))
        self.test_y_hat = np.concatenate((self.test_y_hat, y_hat.cpu().detach().numpy()))

        return loss_metrics["test_loss"]
    
    def training_epoch_end(self, outputs) -> None:
        loss = np.mean([output['loss'].cpu().numpy() for output in outputs])
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser
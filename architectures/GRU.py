from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim
from torch import sqrt
import numpy as np
import argparse 


class GRU(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        forecast_horizon,
        hidden_dim,
        loss_fn,
        censored,
        num_layers: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_layers = num_layers
        self.censored = censored
        self._loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.feat_max_val = feat_max_val
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=hidden_dim, out_features=forecast_horizon)
        
        # To save predictions and their true values for visualizations
        self.test_y = np.empty((0, forecast_horizon))
        self.test_y_hat = np.empty((0, forecast_horizon))
        self.test_y_true = np.empty((0, forecast_horizon))

    def forward(self, x):
        _, hn = self.gru(x)
        out = self.linear(hn[-1])#.flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out.exp()

    def _get_preds_loss_metrics(self, batch, stage):
        x, y, tau, y_true = batch
        y_hat = self(x)#.view(-1)
        
        if self.censored: 
            loss = self._loss_fn(y_hat, y, tau)
            loss_uncen = nn.PoissonNLLLoss(log_input=False)
            loss_true = loss_uncen(y_hat, y_true)

            mse = F.mse_loss(y_hat, y_true)
            mae = F.l1_loss(y_hat, y_true) 
        else:
            loss = self._loss_fn(y_hat, y)
            loss_true = self._loss_fn(y_hat, y_true)
            mse = F.mse_loss(y_hat, y)
            mae = F.l1_loss(y_hat, y)  

        return {
            f"{stage}_loss": loss,
            f"{stage}_loss_true": loss_true.item(),
            f"{stage}_mae": mae.item(),
            f"{stage}_rmse": sqrt(mse).item(),
            f"{stage}_mse": mse.item(),
        }, y, y_hat, y_true
    
    def training_step(self, batch, batch_idx):
        loss_metrics, _, _, _= self._get_preds_loss_metrics(batch, "train")
        self.log_dict(loss_metrics, prog_bar=True, on_epoch=True, on_step=False)
        return loss_metrics["train_loss"]

    def validation_step(self, batch, batch_idx):
        loss_metrics, _, _, _ = self._get_preds_loss_metrics(batch, "val")
        self.log_dict(loss_metrics, prog_bar=True, on_epoch=True, on_step=False)
        return loss_metrics["val_loss"]
    
    def test_step(self, batch, batch_idx):
        loss_metrics, y, y_hat, y_true = self._get_preds_loss_metrics(batch, "test")
        self.log_dict(loss_metrics, on_epoch=True, on_step=False, prog_bar=True)
        self.test_y = np.concatenate((self.test_y, y.cpu().detach().numpy()))
        self.test_y_hat = np.concatenate((self.test_y_hat, y_hat.cpu().detach().numpy()))
        self.test_y_true = np.concatenate((self.test_y_true, y_true.cpu().detach().numpy()))

        return loss_metrics["test_loss"]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=72)
        parser.add_argument("--num_layers", type=int, default=1)
        #parser.add_argument("--output_dim", type=int, default=1)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        
        return parser

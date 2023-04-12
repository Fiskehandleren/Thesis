from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from torch import sqrt
import numpy as np
import argparse 
import torch


class LSTM(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim, 
        loss_fn,
        censored,
        hidden_dim,
        num_layers: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.censored = censored
        self._loss_fn = loss_fn
        self.feat_max_val = feat_max_val
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=output_dim)
        
        
        # To save predictions and their true values for visualizations
        self.test_y = np.empty(0)
        self.test_y_hat = np.empty(0)
        self.test_y_true = np.empty(0)

    def forward(self, x):
        batch_size = x.shape[0]
        # If no h0 and c0 is given to the model, they're initialized to zeros
        # These two caused issues so outcommented. Why am I writing in English?

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()
        
        _, (hn, _) = self.lstm(x)

        #:math:`(\text{num\_layers}, N, H_{out})` containing the
        # final hidden state for each element in the sequence.
        out = self.linear(hn[-1]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        # Should we always grab the last layer? [-1] indicese last eleement

        return out.exp()    
    
    def _get_preds_loss_metrics(self, batch, stage):
        '''
        if self.censored:
            x, y, tau, y_true = batch
        else:
            x, y = batch
        '''
        x, y, tau, y_true = batch
        y_hat = self(x).view(-1)
        
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
            f"{stage}_loss_true": loss_true,
            f"{stage}_mae": mae,
            f"{stage}_rmse": sqrt(mse),
            f"{stage}_mse": mse,
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
        #optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=72)
        parser.add_argument("--num_layers", type=int, default=1)
        #parser.add_argument("--output_dim", type=int, default=1)
        return parser


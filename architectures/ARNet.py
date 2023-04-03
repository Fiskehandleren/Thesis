from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from torch import sqrt
import numpy as np
import argparse 

class ARNet(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        loss_fn,
        censored,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.censored = censored
        self._loss_fn = loss_fn
        self.feat_max_val = feat_max_val
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.activation = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # To save predictions and their true values for visualizations
        self.test_y = np.empty(0)
        self.test_y_hat = np.empty(0)


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)

        return out.exp()
    
    def training_step(self, batch, batch_idx):
        # Run forward calculation
        if self.censored == True:
            x, y, tau = batch
        else:
            x, y = batch

        y_predict = self(x).view(-1)

        # Compute loss.
        if self.censored == True:
            loss = self._loss_fn(y_predict, y, tau)
        else:
            loss = self._loss_fn(y_predict, y)
        
        mse = F.mse_loss(y_predict, y)
        mae = F.l1_loss(y_predict, y)
        metrics = {
            "train_loss": loss,
            "train_mse": mse,
            "train_rmse": sqrt(mse),
            "train_mae": mae
        }

        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        return loss
   
    def validation_step(self, batch, batch_idx):
        if self.censored:
            x, y, tau = batch
        else:
            x, y = batch

        y_predict = self(x).view(-1)

        # Compute loss.
        if self.censored:
            loss = self._loss_fn(y_predict, y, tau)
        else:
            loss = self._loss_fn(y_predict, y)
        
        mse = F.mse_loss(y_predict, y)
        mae = F.l1_loss(y_predict, y)       
        metrics = {
            "val_loss": loss,
            "val_rmse": sqrt(mse),
            "val_mse": mse,
            "val_mae": mae
        }
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    

    def test_step(self, batch, batch_idx):
        if self.censored:
            x, y, tau = batch
        else:
            x, y = batch

        y_predict = self(x).view(-1)

        # Compute loss.
        if self.censored:
            loss = self._loss_fn(y_predict, y, tau)
        else:
            loss = self._loss_fn(y_predict, y)
        mse = F.mse_loss(y_predict, y)
        mae = F.l1_loss(y_predict, y)
        metrics = {
            "test_loss": loss,
            "test_mse": mse,
            "test_rmse": sqrt(mse),
            "test_mae": mae
        }

        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        
        self.test_y = np.concatenate((self.test_y, y.cpu().detach().numpy()))
        self.test_y_hat = np.concatenate((self.test_y_hat, y_predict.cpu().detach().numpy()))

        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=72)
        #parser.add_argument("--output_dim", type=int, default=8)
        return parser
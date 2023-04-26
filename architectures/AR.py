from torch import nn
import pytorch_lightning as pl
from torch import optim
import numpy as np
import argparse 

from utils.losses import get_loss_metrics

class AR(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        forecast_horizon,
        loss_fn,
        censored,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.censored = censored
        self.loss_fn = loss_fn

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        AR.get_loss_metrics = get_loss_metrics
        self.fc1 = nn.Linear(input_dim, forecast_horizon) 

        # To save predictions and their true values for visualizations
        self.test_y = np.empty((0, forecast_horizon))
        self.test_y_hat = np.empty((0, forecast_horizon))
        self.test_y_true = np.empty((0, forecast_horizon))

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.fc1(x)
        return out.exp()
    
    def _get_preds(self, batch):
        x = batch[0]
        return self(x)#.view(-1)

    def _get_preds_loss_metrics(self, batch, stage):
        y_hat = self._get_preds(batch)
        return self.get_loss_metrics(batch, y_hat, stage)
    
    def training_step(self, batch, batch_idx):
        loss_metrics, _, _, _ = self._get_preds_loss_metrics(batch, "train")
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
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        #parser.add_argument("--output_dim", type=int, default=8)
        return parser